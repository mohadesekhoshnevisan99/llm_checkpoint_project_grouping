from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nodes.node import EventLogger
from simulation.backend import SimulationBackend
from simulation.progress import Slowdown, advance_work

from .base import CheckpointWorker, bump_network_tasks, ckpt_fabric_of

if TYPE_CHECKING:
    from jobs.config import ClusterConfig, JobConfig


@dataclass(frozen=True, slots=True)
class CheckpointCopy:
    owner_rank: int
    location_rank: int
    tier: str
    iteration: int
    size_gb: float
    completed_at: float


class BilledCopies(dict):
    """PURE-ACCOUNTING wrapper around the `copies` ledger (resource bill,
    2026-07-28): whenever a CheckpointCopy is destroyed or overwritten, its
    residency (GB x seconds since completed_at) is booked into the owning
    strategy's stats under `ckpt_{tier}_gb_seconds`. Mutation semantics are
    IDENTICAL to dict — no dynamics, no RNG, no env interaction. `bill_gb`
    holds per-key cluster-wide GB overrides for the sites whose stored
    size_gb is per-rank rather than per-cohort (capture / snapshot / restore
    on an m-node cohort worker: resident bytes are m x shard_gb while the
    ledger keeps the per-rank shard for recovery semantics)."""

    __slots__ = ("_now", "_stats", "_on_end", "bill_gb")

    def __init__(self, now_fn, stats, on_end=None):
        super().__init__()
        self._now = now_fn
        self._stats = stats
        self._on_end = on_end          # optional trace hook (key, copy, gb)
        self.bill_gb: dict = {}

    def _book(self, key, copy) -> None:
        gb = self.bill_gb.pop(key, None)
        if gb is None:
            gb = copy.size_gb
        self._stats[f"ckpt_{copy.tier}_gb_seconds"] += gb * max(
            0.0, self._now() - copy.completed_at)
        if self._on_end is not None:
            self._on_end(key, copy, gb)

    def __setitem__(self, key, value) -> None:
        old = self.get(key)
        if old is not None:
            self._book(key, old)
        super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        old = self.get(key)
        if old is not None:
            self._book(key, old)
        super().__delitem__(key)

    def pop(self, key, *default):
        if key in self:
            self._book(key, self[key])
        return super().pop(key, *default)

    def finalize(self) -> None:
        """End of run: book residency of still-live copies up to the horizon.
        Idempotent — each copy's completed_at is advanced to now."""
        now = self._now()
        for key, copy in list(self.items()):
            gb = self.bill_gb.get(key, copy.size_gb)
            self._stats[f"ckpt_{copy.tier}_gb_seconds"] += gb * max(
                0.0, now - copy.completed_at)
            super().__setitem__(
                key, dataclasses.replace(copy, completed_at=now))


class TieredCheckpointStrategy:
    """Keep completed pipeline shards in local or rank-paired DRAM/SSD."""

    contention_quantum_seconds = 0.05

    # NON-BLOCKING CHECKPOINT CAPTURE (sim_validation_protocol.md CL-017).
    # OFF (default): the GPU->DRAM capture holds the worker's exclusive `gpu`
    #   lock for the whole D2H copy, so a training iteration that re-requests
    #   the GPU stalls behind it — the +0.2937 pp capture floor CL-012.A(d)
    #   derived arithmetically from the sim's own dt_s mass. Every committed
    #   scenario, staircase, rack table and parallelism table was produced
    #   under this model and is byte-identical with the flag absent (the hard
    #   regression gate, same as CL-014/CL-016).
    # ON  (`model.nonblocking_capture: true`): the capture takes NO GPU lock,
    #   so it charges training exactly zero. exp1 hardware refuted the lock:
    #   capture 413 ms vs 485 ms iteration, measured p99 stretch 1.002 — "The
    #   capture does not stall training at all"
    #   (meetings_summary/2026-07-31_exp1_results.md section 5, point 3;
    #   PAPER_NOTES.md 2026-08-01). The capture DURATION is unchanged and
    #   still gates the checkpoint pipeline: the DRAM copy is placed and the
    #   persist/stripe starts only after the D2H timeout completes. Only the
    #   training-stall coupling is removed. ZERO free parameters: the flag
    #   deletes a lock; it sets nothing.
    # Set per run by run_scenario.run_arm() (on CrossJobPeerStrategy, exactly
    # parallel to fabric_aware_coupling / intensity_coupling); a class
    # attribute so arms cannot leak into each other and a test can flip it
    # without rebuilding a scenario.
    nonblocking_capture = False

    def __init__(
        self,
        *,
        run_id: int,
        cluster: ClusterConfig,
        logger: EventLogger,
        object_store: Any,
        workers: dict[int, CheckpointWorker],
        paired: bool,
        persist_to_ssd: bool,
        backend: SimulationBackend,
        strategy_name: str | None = None,
    ) -> None:
        del object_store
        self.backend = backend
        self.run_id = run_id
        self.cluster = cluster
        self.logger = logger
        self.workers = workers
        self.paired = paired
        self.persist_to_ssd = persist_to_ssd
        self.strategy_name = strategy_name or (
            "paired_tiered"
            if paired
            else ("ssd_tiered" if persist_to_ssd else "cpu_tiered")
        )
        self.copies: dict[tuple[int, int, str], CheckpointCopy] = {}
        self.checkpoint_locks = {
            rank: backend.resource() for rank in workers
        }
        self.stage_completion_events: dict[tuple[int, int], Any] = {}
        self.peer_transfer_slots = {
            min(rank, self.partner_rank(rank)): backend.resource()
            for rank in workers
            if self.paired
        }

    def stage_completion_event(
        self,
        worker: CheckpointWorker,
        iteration: int,
    ) -> Any:
        """Return the one-shot event fired after foreground GPU->DRAM capture.

        The measured DeepSpeed async API returns only after this capture; the
        configured runtime uses this hook when checkpoint.backpressure=true.
        Persistence and cleanup remain background work.
        """

        key = (worker.rank, iteration)
        event = self.stage_completion_events.get(key)
        if event is None:
            event = self.backend.event()
            self.stage_completion_events[key] = event
        return event

    def gpu_slowdown(
        self,
        worker: CheckpointWorker,
        job: JobConfig,
    ) -> float:
        if worker.checkpoint_uploads <= 0:
            return 1.0
        return job.checkpoint_upload_gpu_slowdown

    def network_slowdown(self, worker: CheckpointWorker) -> float:
        """Fair-share a rank's network with active peer checkpoint traffic."""

        return 1.0 + worker.checkpoint_network_tasks

    def ckpt_fabric(self, job_id: str) -> str:
        """Which fabric this job's checkpoint STREAMS ride (CL-014).

        Read off the per-job `rates:` block that CrossJobPeerStrategy already
        carries (`rates_by_job`); strategies without one are single-fabric, so
        every transfer is tagged with the default fabric and the fabric-scoped
        consumer behaves exactly like the fabric-blind one."""
        return ckpt_fabric_of(getattr(self, "rates_by_job", {}) or {}, job_id)

    def partner_rank(self, rank: int) -> int:
        if not self.paired:
            return rank
        return rank ^ 1

    def handle_failure(
        self,
        worker: CheckpointWorker,
        failure_type: str,
    ) -> dict[str, float]:
        # event engine: this worker's failure_generation has just been bumped, so
        # any in-flight capacity transfer that watches it is now invalid — abort it
        # promptly (the observable equivalent of the polling loop's per-quantum
        # valid() check). No-op unless a capacity registry is attached in event mode.
        cap = getattr(self, "capacity", None)
        if cap is not None and getattr(self, "engine", None) == "event":
            cap.ev_notify_worker_change(worker)
        if failure_type == "process":
            return {"dram_flushed_gb": 0.0, "ssd_flushed_gb": 0.0}

        flushed = {"dram_flushed_gb": 0.0, "ssd_flushed_gb": 0.0}
        tiers = {"dram"} if failure_type == "node" else {"dram", "ssd"}
        for key, copy in list(self.copies.items()):
            if copy.location_rank != worker.rank or copy.tier not in tiers:
                continue
            flushed[f"{copy.tier}_flushed_gb"] += copy.size_gb
            del self.copies[key]
        return flushed

    def checkpoint(
        self,
        worker: CheckpointWorker,
        job: JobConfig,
        *,
        iteration: int,
        checkpoint_epoch: int | None = None,
    ):
        if checkpoint_epoch is None:
            checkpoint_epoch = getattr(worker, "checkpoint_epoch", 0)
        shard_gb = job.checkpoint_gb / job.pipeline_stages
        checkpoint_group = f"{job.job_id}-iteration-{iteration}"
        lock = self.checkpoint_locks[worker.rank]
        with lock.request() as lock_request:
            yield lock_request
            worker.active_checkpoint_gb = shard_gb
            try:
                while True:
                    if getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch:
                        return False
                    yield from self._wait_until_healthy(worker)
                    if getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch:
                        return False
                    generation = worker.failure_generation
                    # capture acquires the GPU FIRST, then the CPU: requesting
                    # both together parks the CPU grant for the whole GPU wait
                    # (an entire training iteration), blocking the async flusher
                    # and pinning its donor reservations — found by hand-check2
                    # (imbalanced 2/3/4 scenario), missed by the 31-agent audit.
                    # CL-017: under nonblocking_capture the GPU lock is not
                    # taken at all — hardware shows the D2H copy overlaps
                    # training (exp1 p99 stretch 1.002) — while the CPU grant
                    # and the capture timeout below are unchanged, so the
                    # persist pipeline still waits for the full capture.
                    if self.nonblocking_capture:
                        gpu_request = None
                    else:
                        gpu_request = worker.gpu.request()
                        yield gpu_request
                    cpu_request = worker.cpu.request(priority=0)
                    yield cpu_request
                    if (
                        worker.failed
                        or worker.failure_generation != generation
                    ):
                        if gpu_request is not None:
                            worker.gpu.release(gpu_request)
                        worker.cpu.release(cpu_request)
                        if getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch:
                            return False
                        continue

                    start = self.backend.now
                    duration = (
                        shard_gb / self.cluster.gpu_cpu_bandwidth_gbps
                    )
                    completed = yield from self._checkpoint_timeout(
                        (worker,),
                        generations=(generation,),
                        duration=duration,
                    )
                    if gpu_request is not None:
                        worker.gpu.release(gpu_request)
                    worker.cpu.release(cpu_request)
                    if not completed:
                        if getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch:
                            return False
                        continue
                    if getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch:
                        return False

                    self._record(
                        worker,
                        start=start,
                        category="Checkpoint",
                        operation="checkpoint_stage_gpu_to_dram",
                        resources=["CPU", "GPU"],
                        iteration=iteration,
                        data_gb=shard_gb,
                        details=self._checkpoint_details(
                            worker,
                            job,
                            checkpoint_group,
                            duration,
                        ),
                    )
                    self._put_copy(
                        owner_rank=worker.rank,
                        location_rank=worker.rank,
                        tier="dram",
                        iteration=iteration,
                        size_gb=shard_gb,
                        bill_gb=shard_gb * getattr(worker, "represents", 1),
                    )
                    stage_event = getattr(
                        self,
                        "stage_completion_events",
                        {},
                    ).get(
                        (worker.rank, iteration)
                    )
                    if stage_event is not None and not stage_event.triggered:
                        stage_event.succeed(True)
                    persisted = yield from self._persist_checkpoint(
                        worker,
                        job,
                        iteration=iteration,
                        generation=generation,
                        shard_gb=shard_gb,
                        checkpoint_group=checkpoint_group,
                        checkpoint_epoch=checkpoint_epoch,
                    )
                    if persisted:
                        cleanup_seconds = (
                            getattr(job, "async_checkpoint_cleanup_seconds", 0.0)
                            if getattr(job, "checkpoint_mode", "asynchronous")
                            == "asynchronous"
                            else 0.0
                        )
                        if cleanup_seconds > 0:
                            cleanup_start = self.backend.now
                            cleanup_completed = yield from self._checkpoint_timeout(
                                (worker,),
                                generations=(generation,),
                                duration=cleanup_seconds,
                            )
                            if not cleanup_completed:
                                if (
                                    getattr(worker, "checkpoint_epoch", 0)
                                    != checkpoint_epoch
                                ):
                                    return False
                                continue
                            self._record(
                                worker,
                                start=cleanup_start,
                                category="Checkpoint",
                                operation="checkpoint_async_cleanup",
                                resources=[],
                                iteration=iteration,
                                data_gb=0.0,
                                details={
                                    "checkpoint_strategy": self.strategy_name,
                                    "checkpoint_mode": getattr(
                                        job,
                                        "checkpoint_mode",
                                        "asynchronous",
                                    ),
                                    "checkpoint_group": checkpoint_group,
                                    "base_duration": cleanup_seconds,
                                    "effective_slowdown": 1.0,
                                },
                            )
                        return
            finally:
                worker.active_checkpoint_gb = 0.0

    def _persist_checkpoint(
        self,
        worker: CheckpointWorker,
        job: JobConfig,
        *,
        iteration: int,
        generation: int,
        shard_gb: float,
        checkpoint_group: str,
        checkpoint_epoch: int | None = None,
    ):
        if (
            checkpoint_epoch is not None
            and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
        ):
            return False
        common = {
            "checkpoint_strategy": self.strategy_name,
            "checkpoint_group": checkpoint_group,
            "checkpoint_owner_rank": worker.rank,
            "checkpoint_partner_rank": self.partner_rank(worker.rank),
            "shard": worker.pipeline_stage,
            "shard_count": job.pipeline_stages,
        }
        if self.persist_to_ssd:
            local_ssd_ok = yield from self._checkpoint_chunks(
                actor=worker,
                watched=(worker,),
                generations=(generation,),
                iteration=iteration,
                size_gb=shard_gb,
                chunk_gb=job.checkpoint_chunk_gb,
                bandwidth_gbps=self.cluster.local_ssd_bandwidth_gbps,
                operation="checkpoint_dram_to_local_ssd_chunk",
                source=f"{worker.name}/dram",
                destination=f"{worker.name}/ssd",
                resources=["CPU"],
                details=common,
            )
            if not local_ssd_ok:
                return False
            if (
                checkpoint_epoch is not None
                and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
            ):
                return False
            self._put_copy(
                owner_rank=worker.rank,
                location_rank=worker.rank,
                tier="ssd",
                iteration=iteration,
                size_gb=shard_gb,
            )

        if not self.paired:
            return True

        partner = self.workers[self.partner_rank(worker.rank)]
        partner_generation = partner.failure_generation
        peer_dram_ok = yield from self._checkpoint_chunks(
            actor=worker,
            watched=(worker, partner),
            generations=(generation, partner_generation),
            iteration=iteration,
            size_gb=shard_gb,
            chunk_gb=job.checkpoint_chunk_gb,
            bandwidth_gbps=self.cluster.network_bandwidth_gbps,
            operation="checkpoint_dram_to_peer_dram_chunk",
            source=f"{worker.name}/dram",
            destination=f"{partner.name}/dram",
            resources=["CPU", "NETWORK"],
            details={
                **common,
                "link": (
                    f"{job.job_id}/checkpoint-pair/"
                    f"{min(worker.rank, partner.rank)}-"
                    f"{max(worker.rank, partner.rank)}"
                ),
            },
        )
        if not peer_dram_ok:
            return False
        if (
            checkpoint_epoch is not None
            and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
        ):
            return False
        self._put_copy(
            owner_rank=worker.rank,
            location_rank=partner.rank,
            tier="dram",
            iteration=iteration,
            size_gb=shard_gb,
        )
        if not self.persist_to_ssd:
            return True

        peer_ssd_ok = yield from self._checkpoint_chunks(
            actor=partner,
            watched=(worker, partner),
            generations=(generation, partner_generation),
            iteration=iteration,
            size_gb=shard_gb,
            chunk_gb=job.checkpoint_chunk_gb,
            bandwidth_gbps=self.cluster.local_ssd_bandwidth_gbps,
            operation="checkpoint_peer_dram_to_peer_ssd_chunk",
            source=f"{partner.name}/dram",
            destination=f"{partner.name}/ssd",
            resources=["CPU"],
            details=common,
        )
        if not peer_ssd_ok:
            return False
        if (
            checkpoint_epoch is not None
            and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
        ):
            return False
        self._put_copy(
            owner_rank=worker.rank,
            location_rank=partner.rank,
            tier="ssd",
            iteration=iteration,
            size_gb=shard_gb,
        )
        return True

    def _checkpoint_chunks(
        self,
        *,
        actor: CheckpointWorker,
        watched: tuple[CheckpointWorker, ...],
        generations: tuple[int, ...],
        iteration: int,
        size_gb: float,
        chunk_gb: float,
        bandwidth_gbps: float,
        operation: str,
        source: str,
        destination: str,
        resources: list[str],
        details: dict[str, Any],
    ):
        chunk_count = math.ceil(size_gb / chunk_gb)
        remaining = size_gb
        for chunk_index in range(1, chunk_count + 1):
            current_gb = min(chunk_gb, remaining)
            uses_network = "NETWORK" in resources
            cpu_workers = (actor,)
            network_workers = watched if uses_network else ()
            pair_slot = None
            pair_slot_request = None
            if uses_network:
                pair_key = min(worker.rank for worker in watched)
                pair_slot = self.peer_transfer_slots[pair_key]
                pair_slot_request = pair_slot.request()
                yield pair_slot_request
            cpu_requests = [
                worker.cpu.request(priority=10) for worker in cpu_workers
            ]
            yield self.backend.all_of(cpu_requests)
            if not self._checkpoint_generation_valid(watched, generations):
                for worker, request in zip(cpu_workers, cpu_requests):
                    worker.cpu.release(request)
                if pair_slot is not None and pair_slot_request is not None:
                    pair_slot.release(pair_slot_request)
                return False
            start = self.backend.now
            duration = current_gb / bandwidth_gbps
            for worker in cpu_workers:
                worker.checkpoint_uploads += 1
            # CL-014: the stream rides the OWNER's checkpoint fabric end to end
            # (the arm shapes one wire; both endpoints sit on it), so the donor's
            # counter is tagged with the owner's fabric, not the donor's.
            # CL-016: rate_gbps = the SAME `bandwidth_gbps` this fixed-rate path
            # times the chunk with (`duration = current_gb / bandwidth_gbps`
            # above) — a scenario rate constant, not a fitted value.
            chunk_fabric = self.ckpt_fabric(actor.job_id)
            bump_network_tasks(network_workers, chunk_fabric, +1,
                               rate_gbps=bandwidth_gbps)
            try:
                if uses_network:
                    completed, bandwidth_contention = (
                        yield from self._checkpoint_network_timeout(
                            watched,
                            generations=generations,
                            duration=duration,
                        )
                    )
                else:
                    if "ssd" in operation:
                        completed, bandwidth_contention = (
                            yield from self._checkpoint_local_ssd_timeout(
                                actor,
                                watched,
                                generations=generations,
                                duration=duration,
                            )
                        )
                    else:
                        completed = yield from self._checkpoint_timeout(
                            watched,
                            generations=generations,
                            duration=duration,
                        )
                        bandwidth_contention = 0.0
            finally:
                for worker in cpu_workers:
                    worker.checkpoint_uploads -= 1
                bump_network_tasks(network_workers, chunk_fabric, -1,
                                   rate_gbps=bandwidth_gbps)
                for worker, request in zip(cpu_workers, cpu_requests):
                    worker.cpu.release(request)
                if pair_slot is not None and pair_slot_request is not None:
                    pair_slot.release(pair_slot_request)
            if not completed:
                return False
            actual_duration = self.backend.now - start
            chunk_details = {
                **details,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "base_duration": duration,
                "bandwidth_contention_seconds": bandwidth_contention,
                "effective_slowdown": actual_duration / duration,
                "peer_pair_checkpoint_gb": (
                    size_gb * 2 if uses_network else None
                ),
            }
            self._record(
                actor,
                start=start,
                category="Checkpoint",
                operation=operation,
                resources=resources,
                iteration=iteration,
                source=source,
                destination=destination,
                data_gb=current_gb,
                details=chunk_details,
            )
            remaining -= current_gb
        return True

    def recover(
        self,
        worker: CheckpointWorker,
        job: JobConfig,
    ):
        generation = worker.failure_generation
        shard_gb = job.checkpoint_gb / job.pipeline_stages
        source = self._latest_available_copy(worker)
        checkpoint_iteration = source.iteration if source is not None else 0
        source_tier = source.tier if source is not None else "initial_state"
        source_rank = (
            source.location_rank if source is not None else None
        )

        if source is None:
            loaded = yield from self._recovery_chunks(
                worker,
                generation=generation,
                iteration=0,
                size_gb=shard_gb,
                chunk_gb=job.checkpoint_chunk_gb,
                bandwidth_gbps=self.cluster.object_store_bandwidth_gbps,
                operation="initial_state_to_dram_chunk",
                source="job-initial-state",
                destination=f"{worker.name}/dram",
                resources=["CPU", "NETWORK"],
                source_tier=source_tier,
                source_rank=source_rank,
            )
        elif source.tier == "dram" and source.location_rank == worker.rank:
            loaded = True
        else:
            is_peer = source.location_rank != worker.rank
            operation = (
                f"checkpoint_peer_{source.tier}_to_dram_recovery_chunk"
                if is_peer
                else "checkpoint_ssd_to_dram_recovery_chunk"
            )
            bandwidth = (
                min(
                    self.cluster.local_ssd_bandwidth_gbps,
                    self.cluster.network_bandwidth_gbps,
                )
                if is_peer and source.tier == "ssd"
                else (
                    self.cluster.network_bandwidth_gbps
                    if is_peer
                    else self.cluster.local_ssd_bandwidth_gbps
                )
            )
            loaded = yield from self._recovery_chunks(
                worker,
                generation=generation,
                iteration=checkpoint_iteration,
                size_gb=shard_gb,
                chunk_gb=job.checkpoint_chunk_gb,
                bandwidth_gbps=bandwidth,
                operation=operation,
                source=(
                    f"{self.workers[source.location_rank].name}/"
                    f"{source.tier}"
                ),
                destination=f"{worker.name}/dram",
                resources=["CPU", *(["NETWORK"] if is_peer else [])],
                source_tier=source_tier,
                source_rank=source_rank,
            )
        if not loaded or worker.failure_generation != generation:
            return False

        self._put_copy(
            owner_rank=worker.rank,
            location_rank=worker.rank,
            tier="dram",
            iteration=checkpoint_iteration,
            size_gb=shard_gb,
            bill_gb=shard_gb * getattr(worker, "represents", 1),
            record_begin=True,
        )
        cpu_request = worker.cpu.request(priority=-10)
        gpu_request = worker.gpu.request()
        yield self.backend.all_of([cpu_request, gpu_request])
        if worker.failure_generation != generation:
            worker.cpu.release(cpu_request)
            worker.gpu.release(gpu_request)
            return False

        start = self.backend.now
        duration = shard_gb / self.cluster.gpu_cpu_bandwidth_gbps
        completed = yield from self._recovery_timeout(
            worker,
            generation=generation,
            duration=duration,
        )
        if completed:
            self._record(
                worker,
                start=start,
                category="Recovery",
                operation="dram_to_gpu_restore",
                resources=["CPU", "GPU"],
                iteration=checkpoint_iteration,
                source=f"{worker.name}/dram",
                destination=worker.name,
                data_gb=shard_gb,
                details={
                    "checkpoint_strategy": self.strategy_name,
                    "checkpoint_iteration": checkpoint_iteration,
                    "checkpoint_source_tier": source_tier,
                    "checkpoint_source_rank": source_rank,
                    "base_duration": duration,
                    "effective_slowdown": 1.0,
                },
            )
        worker.cpu.release(cpu_request)
        worker.gpu.release(gpu_request)
        return completed

    def _recovery_chunks(
        self,
        worker: CheckpointWorker,
        *,
        generation: int,
        iteration: int,
        size_gb: float,
        chunk_gb: float,
        bandwidth_gbps: float,
        operation: str,
        source: str,
        destination: str,
        resources: list[str],
        source_tier: str,
        source_rank: int | None,
    ):
        chunk_count = math.ceil(size_gb / chunk_gb)
        remaining = size_gb
        for chunk_index in range(1, chunk_count + 1):
            request = worker.cpu.request(priority=-10)
            yield request
            if worker.failure_generation != generation:
                worker.cpu.release(request)
                return False
            start = self.backend.now
            current_gb = min(chunk_gb, remaining)
            duration = current_gb / bandwidth_gbps
            completed = yield from self._recovery_timeout(
                worker,
                generation=generation,
                duration=duration,
            )
            worker.cpu.release(request)
            if not completed:
                return False
            self._record(
                worker,
                start=start,
                category="Recovery",
                operation=operation,
                resources=resources,
                iteration=iteration,
                source=source,
                destination=destination,
                data_gb=current_gb,
                details={
                    "checkpoint_strategy": self.strategy_name,
                    "checkpoint_iteration": iteration,
                    "checkpoint_source_tier": source_tier,
                    "checkpoint_source_rank": source_rank,
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "base_duration": duration,
                    "effective_slowdown": 1.0,
                },
            )
            remaining -= current_gb
        return True

    def _latest_available_copy(
        self,
        worker: CheckpointWorker,
    ) -> CheckpointCopy | None:
        tier_preference = {
            (False, "dram"): 4,
            (True, "dram"): 3,
            (False, "ssd"): 2,
            (True, "ssd"): 1,
        }
        candidates = []
        max_iteration = getattr(worker, "current_iteration", None)
        for copy in self.copies.values():
            if copy.owner_rank != worker.rank:
                continue
            if max_iteration is not None and copy.iteration > max_iteration:
                continue
            is_peer = copy.location_rank != worker.rank
            if is_peer and self.workers[copy.location_rank].failed:
                continue
            candidates.append(copy)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda copy: (
                copy.iteration,
                tier_preference[
                    (copy.location_rank != worker.rank, copy.tier)
                ],
                copy.completed_at,
            ),
        )

    def _put_copy(
        self,
        *,
        owner_rank: int,
        location_rank: int,
        tier: str,
        iteration: int,
        size_gb: float,
        bill_gb: float | None = None,
        record_begin: bool = False,
    ) -> None:
        # bill_gb (resource bill, 2026-07-28): cluster-wide resident GB for the
        # accounting ledger when it differs from size_gb (cohort capture /
        # snapshot / restore store the PER-RANK shard as size_gb — recovery
        # semantics — while m nodes actually hold the bytes). Pure accounting.
        # record_begin: emit a zero-duration Accounting `ckpt_copy_begin` trace
        # event when the copy actually lands — used by the RESTORE sites, whose
        # copy is registered at restore-START (an aborted restore tail keeps
        # the copy but records no Recovery event, so without this the piece's
        # birth would be invisible to the trace-side extractor).
        owner = self.workers.get(owner_rank)
        if owner is not None and iteration > getattr(
            owner, "current_iteration", iteration
        ):
            return
        key = (owner_rank, location_rank, tier)
        current = self.copies.get(key)
        if current is not None and current.iteration > iteration:
            return
        self.copies[key] = CheckpointCopy(
            owner_rank=owner_rank,
            location_rank=location_rank,
            tier=tier,
            iteration=iteration,
            size_gb=size_gb,
            completed_at=self.backend.now,
        )
        if isinstance(self.copies, BilledCopies):
            if bill_gb is not None:
                self.copies.bill_gb[key] = bill_gb
            if record_begin:
                w = self.workers.get(location_rank)
                self.logger.record(
                    start=self.backend.now, end=self.backend.now,
                    run_id=self.run_id, job_id=getattr(w, "job_id", None),
                    rank=owner_rank,
                    node=getattr(w, "name", f"rank-{location_rank}"),
                    category="Accounting", operation="ckpt_copy_begin",
                    iteration=iteration, data_gb=size_gb,
                    details={"tier": tier, "location_rank": location_rank})

    def _checkpoint_generation_valid(
        self,
        workers: tuple[CheckpointWorker, ...],
        generations: tuple[int, ...],
    ) -> bool:
        return (
            not any(worker.failed for worker in workers)
            and tuple(worker.failure_generation for worker in workers)
            == generations
        )

    def _checkpoint_timeout(
        self,
        workers: tuple[CheckpointWorker, ...],
        *,
        generations: tuple[int, ...],
        duration: float,
    ):
        result = yield from advance_work(
            self.backend,
            base_duration=duration,
            aborted=lambda: not self._checkpoint_generation_valid(
                workers,
                generations,
            ),
            quantum=self.contention_quantum_seconds,
        )
        return result.completed

    def _checkpoint_network_timeout(
        self,
        workers: tuple[CheckpointWorker, ...],
        *,
        generations: tuple[int, ...],
        duration: float,
    ):
        def current_slowdown() -> Slowdown:
            factor = max(
                worker.checkpoint_network_tasks
                + worker.training_network_tasks
                for worker in workers
            )
            return Slowdown(
                factor=factor,
                reasons=(
                    frozenset({"shared_bandwidth"})
                    if factor > 1.0
                    else frozenset()
                ),
            )

        result = yield from advance_work(
            self.backend,
            base_duration=duration,
            slowdown=current_slowdown,
            aborted=lambda: not self._checkpoint_generation_valid(
                workers,
                generations,
            ),
            quantum=self.contention_quantum_seconds,
        )
        return result.completed, result.contention("shared_bandwidth")

    def _checkpoint_local_ssd_timeout(
        self,
        actor: CheckpointWorker,
        workers: tuple[CheckpointWorker, ...],
        *,
        generations: tuple[int, ...],
        duration: float,
    ):
        state = getattr(actor, "physical_state", None)
        if state is None or self.cluster.gpus_per_node <= 1:
            completed = yield from self._checkpoint_timeout(
                workers,
                generations=generations,
                duration=duration,
            )
            return completed, 0.0

        state.local_ssd_tasks += 1
        try:
            yield self.backend.timeout(0)
            result = yield from advance_work(
                self.backend,
                base_duration=duration,
                slowdown=lambda: Slowdown(
                    factor=max(1.0, float(state.local_ssd_tasks)),
                    reasons=(
                        frozenset({"shared_local_ssd"})
                        if state.local_ssd_tasks > 1
                        else frozenset()
                    ),
                ),
                aborted=lambda: not self._checkpoint_generation_valid(
                    workers,
                    generations,
                ),
                quantum=self.contention_quantum_seconds,
            )
        finally:
            state.local_ssd_tasks -= 1
        return result.completed, result.contention("shared_local_ssd")

    def _recovery_timeout(
        self,
        worker: CheckpointWorker,
        *,
        generation: int,
        duration: float,
    ):
        result = yield from advance_work(
            self.backend,
            base_duration=duration,
            aborted=lambda: worker.failure_generation != generation,
            quantum=self.contention_quantum_seconds,
        )
        return result.completed

    def _wait_until_healthy(self, worker: CheckpointWorker):
        while worker.failed:
            recovered_event = worker.recovered_event
            if recovered_event is not None:
                yield recovered_event
            else:
                yield self.backend.timeout(0)

    def _checkpoint_details(
        self,
        worker: CheckpointWorker,
        job: JobConfig,
        checkpoint_group: str,
        duration: float,
    ) -> dict[str, Any]:
        return {
            "checkpoint_strategy": self.strategy_name,
            "checkpoint_group": checkpoint_group,
            "checkpoint_owner_rank": worker.rank,
            "checkpoint_partner_rank": self.partner_rank(worker.rank),
            "shard": worker.pipeline_stage,
            "shard_count": job.pipeline_stages,
            "base_duration": duration,
            "failure_wait_seconds": 0.0,
            "effective_slowdown": 1.0,
        }

    def _record(
        self,
        worker: CheckpointWorker,
        *,
        start: float,
        category: str,
        operation: str,
        resources: list[str],
        iteration: int,
        source: str | None = None,
        destination: str | None = None,
        data_gb: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.logger.record(
            start=start,
            end=self.backend.now,
            run_id=self.run_id,
            job_id=worker.job_id,
            rank=worker.rank,
            node=worker.name,
            physical_node=worker.physical_node,
            pipeline_stage=worker.pipeline_stage,
            data_parallel_rank=worker.data_parallel_rank,
            category=category,
            operation=operation,
            resources=resources,
            iteration=iteration,
            source=source,
            destination=destination,
            data_gb=data_gb,
            details=details,
        )
