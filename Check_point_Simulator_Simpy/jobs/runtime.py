from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable

import simpy

# NOTE(sam): imported lazily below — checkpointing<->simulation<->jobs
# form an import cycle; module-level import breaks `import run_scenario`
from nodes.node import EventLogger
from simulation.backend import SimulationBackend, SimPyBackend
from simulation.progress import Slowdown, advance_work, combine_slowdowns

from .aggregate_runtime import AggregatedDataParallelRuntime
from .chunked_runtime import ChunkedDataParallelRuntime
from .config import JobConfig, SimulatorConfig
from .failures import FailureController


@dataclass(slots=True)
class PhysicalNodeState:
    """Resources and activity counters shared by ranks on one physical host."""

    cpu: Any
    network_tasks: int = 0
    local_ssd_tasks: int = 0


@dataclass(slots=True)
class JobWorker:
    job_id: str
    rank: int
    data_parallel_rank: int
    pipeline_stage: int
    physical_node: str
    gpu: Any
    cpu: Any
    gpu_index: int = 0
    physical_state: PhysicalNodeState | None = None
    checkpoint_uploads: int = 0
    checkpoint_network_tasks: int = 0
    # FABRIC-SCOPED COUPLING (sim_validation_protocol.md CL-012.D S1/S2,
    # implemented by CL-014): the SAME in-flight-transfer count as the scalar
    # above, split by the fabric each transfer rides. Maintained unconditionally
    # by checkpointing.base.bump_network_tasks — it is a label on an existing
    # event, not a new dynamic — and consumed only when
    # `model.fabric_aware_coupling` is on. With no fabric declared anywhere the
    # dict is just {"default": <the scalar>}.
    checkpoint_network_tasks_by_fabric: dict[str, int] = field(default_factory=dict)
    # INTENSITY-BASED COUPLING (sim_validation_protocol.md CL-012.D S3,
    # implemented by CL-016): the aggregate OFFERED rate (GB/s) of the same
    # in-flight transfers the count above tracks, split by fabric. Each entry is
    # the sum of the in-flight streams' rate caps — the shaped/measured rates the
    # scenario already declares, never a fitted value. Maintained unconditionally
    # by checkpointing.base.bump_network_tasks (a second value on an existing
    # bookkeeping event, not a new dynamic) and consumed only when
    # `model.intensity_coupling` is on.
    checkpoint_network_demand_by_fabric: dict[str, float] = field(
        default_factory=dict)
    training_network_tasks: int = 0
    failed: bool = False
    failure_type: str | None = None
    failure_started: float | None = None
    failed_until: float = 0.0
    failure_signal_count: int = 0
    failure_batch_size: int = 0
    failure_generation: int = 0
    recovered_event: Any | None = None
    active_checkpoint_gb: float = 0.0
    active: bool = True
    current_iteration: int = 0
    # Incremented for every job-wide rollback. Queued checkpoint work captures
    # the value and must not publish state after the timeline changes.
    checkpoint_epoch: int = 0
    failure_types_seen: list[str] = field(default_factory=list)
    failure_dram_flushed_gb: float = 0.0
    failure_ssd_flushed_gb: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.job_id}-rank-{self.rank}"


def _all_processes(
    backend: SimulationBackend,
    generators: Iterable,
):
    return backend.all_of(
        [backend.process(generator) for generator in generators]
    )


class DistributedJobRuntime:
    """Shared timing engine for data, pipeline, and hybrid parallel jobs."""

    def __init__(
        self,
        *,
        run_id: int,
        config: JobConfig,
        simulator_config: SimulatorConfig,
        logger: EventLogger,
        node_offset: int,
        object_store: Any,
        verbose: bool,
        backend: SimulationBackend,
        physical_nodes: dict[str, PhysicalNodeState] | None = None,
    ) -> None:
        self.backend = backend
        self.run_id = run_id
        self.config = config
        self.cluster = simulator_config.cluster
        self.iterations = simulator_config.simulation.iterations
        self.logger = logger
        self.object_store = object_store
        self.verbose = verbose
        self.checkpoint_processes: list[Any] = []
        self.pipeline_ready_at: dict[tuple[int, int], float] = {}
        self.workers: dict[tuple[int, int], JobWorker] = {}
        if physical_nodes is None and self.cluster.gpus_per_node > 1:
            raise ValueError(
                "multi-GPU configured runs require the shared physical-node "
                "state supplied by ExecutionPlanner"
            )
        if physical_nodes is None:
            physical_nodes = {
                f"node-{index}": PhysicalNodeState(
                    cpu=backend.priority_resource(
                        self.cluster.cpu_cores_per_node
                    )
                )
                for index in range(self.cluster.node_count)
            }
        placement_width = simulator_config.required_nodes
        for dp_rank in range(config.data_parallel_replicas):
            for stage in range(config.pipeline_stages):
                rank = config.global_rank(dp_rank, stage)
                slot = node_offset + rank
                physical_node = f"node-{slot % placement_width}"
                gpu_index = slot // placement_width
                physical_state = physical_nodes[physical_node]
                self.workers[(dp_rank, stage)] = JobWorker(
                    job_id=config.job_id,
                    rank=rank,
                    data_parallel_rank=dp_rank,
                    pipeline_stage=stage,
                    physical_node=physical_node,
                    gpu_index=gpu_index,
                    physical_state=physical_state,
                    gpu=backend.resource(1),
                    cpu=physical_state.cpu,
                )

        self.checkpoint_strategy = __import__('checkpointing').create_checkpoint_strategy(
            config.checkpoint_strategy,
            run_id=run_id,
            cluster=self.cluster,
            logger=logger,
            object_store=object_store,
            backend=backend,
            workers={
                worker.rank: worker for worker in self.workers.values()
            },
        )

        self.pipeline_links = {
            (dp_rank, stage): backend.resource()
            for dp_rank in range(config.data_parallel_replicas)
            for stage in range(config.pipeline_stages - 1)
        }

    def placement(self) -> list[dict[str, Any]]:
        return [
            {
                "job_id": worker.job_id,
                "rank": worker.rank,
                "data_parallel_rank": worker.data_parallel_rank,
                "pipeline_stage": worker.pipeline_stage,
                "physical_node": worker.physical_node,
                "gpu_index": worker.gpu_index,
                "checkpoint_partner_rank": (
                    worker.rank ^ 1
                    if self.config.checkpoint_strategy
                    in {"cpu_tiered", "ssd_tiered", "paired_tiered"}
                    else None
                ),
            }
            for worker in self.workers.values()
        ]

    def _record(
        self,
        worker: JobWorker,
        *,
        start: float,
        end: float | None = None,
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
            end=self.backend.now if end is None else end,
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

    def _gpu_work(
        self,
        worker: JobWorker,
        *,
        duration: float,
        iteration: int,
        operation: str,
        microbatch: int | None = None,
    ):
        attempt = 0
        while True:
            attempt += 1
            yield from self._wait_for_workers_healthy((worker,))
            with worker.gpu.request() as request:
                yield request
                if worker.failed:
                    continue
                generation = worker.failure_generation
                start = self.backend.now
                if (
                    attempt == 1
                    and operation == "pipeline_optimizer"
                    and self.config.data_parallel_replicas == 1
                ):
                    self._record_synchronization_wait(
                        worker,
                        start=self.pipeline_ready_at.get(
                            (iteration, worker.rank),
                            start,
                        ),
                        end=start,
                        iteration=iteration,
                        operation="pipeline_flush_wait",
                        wait_for="pipeline_iteration_flush",
                    )
                completed, contention = (
                    yield from self._run_contended_gpu_work(
                        worker,
                        base_duration=duration,
                        generation=generation,
                    )
                )

            status = "completed" if completed else "interrupted"
            self._record(
                worker,
                start=start,
                category="Compute",
                operation=operation,
                resources=["GPU"],
                iteration=iteration,
                details={
                    "microbatch": microbatch,
                    "base_duration": duration,
                    **contention,
                    "status": status,
                    "attempt": attempt,
                    "pipeline_stage": worker.pipeline_stage,
                    "data_parallel_rank": worker.data_parallel_rank,
                },
            )
            if not completed:
                yield from self._wait_for_workers_healthy((worker,))
                continue

            if operation in {
                "data_parallel_forward",
                "data_parallel_backward",
                "pipeline_forward",
                "pipeline_backward",
            }:
                self.pipeline_ready_at[(iteration, worker.rank)] = (
                    self.backend.now
                )
            return

    def _run_contended_gpu_work(
        self,
        worker: JobWorker,
        *,
        base_duration: float,
        generation: int,
    ):
        quantum = self.checkpoint_strategy.contention_quantum_seconds
        result = yield from advance_work(
            self.backend,
            base_duration=base_duration,
            slowdown=lambda: Slowdown(
                factor=self.checkpoint_strategy.gpu_slowdown(
                    worker,
                    self.config,
                ),
                reasons=(
                    frozenset({"checkpoint_cpu"})
                    if self.checkpoint_strategy.gpu_slowdown(
                        worker,
                        self.config,
                    )
                    > 1.0
                    else frozenset()
                ),
            ),
            aborted=lambda: (
                worker.failed
                or worker.failure_generation != generation
            ),
            quantum=quantum,
        )
        contention_seconds = result.contention("checkpoint_cpu")
        return result.completed, {
            "contention_seconds": contention_seconds,
            "checkpoint_upload_contention_seconds": contention_seconds,
            "failure_wait_seconds": 0.0,
            "effective_slowdown": result.effective_slowdown,
            "completed_fraction": result.completed_fraction,
        }

    def _record_synchronization_wait(
        self,
        worker: JobWorker,
        *,
        start: float,
        end: float,
        iteration: int,
        operation: str,
        wait_for: str,
    ) -> None:
        if end - start <= 1e-12:
            return

        busy_intervals = sorted(
            (
                event.start,
                event.end,
            )
            for event in self.logger.events
            if event.job_id == worker.job_id
            and event.rank == worker.rank
            and "GPU" in event.resources
            and event.category != "Synchronization"
            and event.end > start
            and event.start < end
        )
        cursor = start
        for busy_start, busy_end in busy_intervals:
            idle_end = min(busy_start, end)
            if idle_end - cursor > 1e-12:
                self._record_wait_interval(
                    worker,
                    start=cursor,
                    end=idle_end,
                    iteration=iteration,
                    operation=operation,
                    wait_for=wait_for,
                )
            cursor = max(cursor, busy_end)
            if cursor >= end:
                return
        if end - cursor > 1e-12:
            self._record_wait_interval(
                worker,
                start=cursor,
                end=end,
                iteration=iteration,
                operation=operation,
                wait_for=wait_for,
            )

    def _record_wait_interval(
        self,
        worker: JobWorker,
        *,
        start: float,
        end: float,
        iteration: int,
        operation: str,
        wait_for: str,
    ) -> None:
        self.logger.record(
            start=start,
            end=end,
            run_id=self.run_id,
            job_id=worker.job_id,
            rank=worker.rank,
            node=worker.name,
            physical_node=worker.physical_node,
            pipeline_stage=worker.pipeline_stage,
            data_parallel_rank=worker.data_parallel_rank,
            category="Synchronization",
            operation=operation,
            resources=["GPU"],
            iteration=iteration,
            details={
                "status": "idle_wait",
                "base_duration": end - start,
                "effective_slowdown": 1.0,
                "wait_for": wait_for,
            },
        )

    def _pipeline_transfer(
        self,
        source: JobWorker,
        destination: JobWorker,
        *,
        iteration: int,
        microbatch: int,
        operation: str,
        data_gb: float,
    ):
        edge = min(source.pipeline_stage, destination.pipeline_stage)
        link = self.pipeline_links[(source.data_parallel_rank, edge)]
        workers = (source, destination)
        duration = data_gb / self.cluster.network_bandwidth_gbps
        launch_duration = self.cluster.communication_launch_seconds
        receive_operation = operation.replace("_send", "_receive")

        while True:
            yield from self._wait_for_workers_healthy(workers)
            generations = tuple(
                worker.failure_generation for worker in workers
            )

            cpu_requests = [worker.cpu.request() for worker in workers]
            yield self.backend.all_of(cpu_requests)
            launch_start = self.backend.now
            launch_completed = yield from self._worker_group_timeout(
                workers,
                generations=generations,
                duration=launch_duration,
            )
            launch_end = self.backend.now
            for worker, request in zip(workers, cpu_requests):
                worker.cpu.release(request)
            if not launch_completed:
                continue

            link_request = link.request()
            gpu_requests = [worker.gpu.request() for worker in workers]
            yield self.backend.all_of([link_request, *gpu_requests])
            if (
                any(worker.failed for worker in workers)
                or tuple(
                    worker.failure_generation for worker in workers
                )
                != generations
            ):
                link.release(link_request)
                for worker, request in zip(workers, gpu_requests):
                    worker.gpu.release(request)
                continue

            start = self.backend.now
            for worker in workers:
                worker.training_network_tasks += 1
            try:
                transfer_completed, bandwidth_contention = (
                    yield from self._worker_group_network_timeout(
                        workers,
                        generations=generations,
                        duration=duration,
                    )
                )
            finally:
                for worker in workers:
                    worker.training_network_tasks -= 1
            end = self.backend.now
            link.release(link_request)
            for worker, request in zip(workers, gpu_requests):
                worker.gpu.release(request)
            if not transfer_completed:
                continue

            link_name = (
                f"{self.config.job_id}/dp-{source.data_parallel_rank}/"
                f"stage-{edge}-stage-{edge + 1}"
            )
            launch_details = {
                "microbatch": microbatch,
                "transfer_operation": operation,
                "base_duration": launch_duration,
                "effective_slowdown": 1.0,
            }
            for worker in workers:
                self._record(
                    worker,
                    start=launch_start,
                    end=launch_end,
                    category="Communication",
                    operation="pipeline_transfer_launch",
                    resources=["CPU"],
                    iteration=iteration,
                    source=source.name,
                    destination=destination.name,
                    data_gb=data_gb,
                    details=launch_details,
                )
            self._record(
                source,
                start=start,
                end=end,
                category="Communication",
                operation=operation,
                resources=["GPU", "NETWORK"],
                iteration=iteration,
                source=source.name,
                destination=destination.name,
                data_gb=data_gb,
                details={
                    "microbatch": microbatch,
                    "link": link_name,
                    "base_duration": duration,
                    "bandwidth_contention_seconds": bandwidth_contention,
                    "failure_wait_seconds": 0.0,
                    "effective_slowdown": (end - start) / duration,
                },
            )
            self._record(
                destination,
                start=start,
                end=end,
                category="Communication",
                operation=receive_operation,
                resources=["GPU"],
                iteration=iteration,
                source=source.name,
                destination=destination.name,
                data_gb=data_gb,
                details={
                    "microbatch": microbatch,
                    "peer_link": link_name,
                    "base_duration": duration,
                    "bandwidth_contention_seconds": bandwidth_contention,
                    "failure_wait_seconds": 0.0,
                    "effective_slowdown": (end - start) / duration,
                },
            )
            return

    def _worker_group_timeout(
        self,
        workers: tuple[JobWorker, ...],
        *,
        generations: tuple[int, ...],
        duration: float,
    ):
        result = yield from advance_work(
            self.backend,
            base_duration=duration,
            aborted=lambda: (
                any(worker.failed for worker in workers)
                or tuple(
                    worker.failure_generation for worker in workers
                )
                != generations
            ),
        )
        return result.completed

    def _worker_group_network_timeout(
        self,
        workers: tuple[JobWorker, ...],
        *,
        generations: tuple[int, ...],
        duration: float,
    ):
        def current_slowdown() -> Slowdown:
            factor = max(
                self.checkpoint_strategy.network_slowdown(worker)
                for worker in workers
            )
            return Slowdown(
                factor=factor,
                reasons=(
                    frozenset({"checkpoint_network"})
                    if factor > 1.0
                    else frozenset()
                ),
            )

        result = yield from advance_work(
            self.backend,
            base_duration=duration,
            slowdown=current_slowdown,
            aborted=lambda: (
                any(worker.failed for worker in workers)
                or tuple(
                    worker.failure_generation for worker in workers
                )
                != generations
            ),
        )
        return result.completed, result.contention("checkpoint_network")

    def _microbatch(
        self,
        *,
        data_parallel_rank: int,
        iteration: int,
        microbatch: int,
    ):
        stage_count = self.config.pipeline_stages
        microbatch_count = self.config.microbatches
        forward_duration = (
            self.config.forward_seconds / stage_count / microbatch_count
        )
        backward_duration = (
            self.config.backward_seconds / stage_count / microbatch_count
        )
        activation_gb = self.config.activation_gb / microbatch_count
        forward_operation = (
            "data_parallel_forward"
            if self.config.strategy == "data_parallel"
            else "pipeline_forward"
        )
        backward_operation = (
            "data_parallel_backward"
            if self.config.strategy == "data_parallel"
            else "pipeline_backward"
        )

        for stage in range(stage_count):
            worker = self.workers[(data_parallel_rank, stage)]
            yield self.backend.process(
                self._gpu_work(
                    worker,
                    duration=forward_duration,
                    iteration=iteration,
                    operation=forward_operation,
                    microbatch=microbatch,
                )
            )
            if stage + 1 < stage_count:
                destination = self.workers[(data_parallel_rank, stage + 1)]
                yield self.backend.process(
                    self._pipeline_transfer(
                        worker,
                        destination,
                        iteration=iteration,
                        microbatch=microbatch,
                        operation="pipeline_activation_send",
                        data_gb=activation_gb,
                    )
                )

        for stage in reversed(range(stage_count)):
            worker = self.workers[(data_parallel_rank, stage)]
            yield self.backend.process(
                self._gpu_work(
                    worker,
                    duration=backward_duration,
                    iteration=iteration,
                    operation=backward_operation,
                    microbatch=microbatch,
                )
            )
            if stage > 0:
                destination = self.workers[(data_parallel_rank, stage - 1)]
                yield self.backend.process(
                    self._pipeline_transfer(
                        worker,
                        destination,
                        iteration=iteration,
                        microbatch=microbatch,
                        operation="pipeline_gradient_send",
                        data_gb=activation_gb,
                    )
                )

    def _data_parallel_all_reduce(
        self,
        *,
        pipeline_stage: int,
        iteration: int,
    ):
        replica_count = self.config.data_parallel_replicas
        if replica_count <= 1:
            return
        workers = tuple(
            self.workers[(dp_rank, pipeline_stage)]
            for dp_rank in range(replica_count)
        )
        gradient_shard_gb = (
            self.config.gradient_gb / self.config.pipeline_stages
        )
        duration = (
            2
            * (replica_count - 1)
            / replica_count
            * gradient_shard_gb
            / self.cluster.network_bandwidth_gbps
        )

        while True:
            yield from self._wait_for_workers_healthy(workers)
            requests = [worker.gpu.request() for worker in workers]
            yield self.backend.all_of(requests)
            start = self.backend.now
            generations = tuple(
                worker.failure_generation for worker in workers
            )

            def current_slowdown() -> Slowdown:
                cpu_factor = max(
                    self.checkpoint_strategy.gpu_slowdown(
                        worker,
                        self.config,
                    )
                    for worker in workers
                )
                network_factor = max(
                    self.checkpoint_strategy.network_slowdown(worker)
                    for worker in workers
                )
                physical_nic_factor = (
                    max(
                        (
                            worker.physical_state.network_tasks
                            for worker in workers
                            if worker.physical_state is not None
                        ),
                        default=1,
                    )
                    if self.cluster.gpus_per_node > 1
                    else 1.0
                )
                return combine_slowdowns(
                    (
                        (cpu_factor, "checkpoint_cpu"),
                        (network_factor, "checkpoint_network"),
                        (physical_nic_factor, "physical_nic"),
                    )
                )

            for worker in workers:
                worker.training_network_tasks += 1
                if worker.physical_state is not None:
                    worker.physical_state.network_tasks += 1
            try:
                if self.cluster.gpus_per_node > 1:
                    # Let collectives from concurrently starting jobs register
                    # before the first fair-share decision.
                    yield self.backend.timeout(0)
                result = yield from advance_work(
                    self.backend,
                    base_duration=duration,
                    slowdown=current_slowdown,
                    aborted=lambda: (
                        any(worker.failed for worker in workers)
                        or tuple(
                            worker.failure_generation for worker in workers
                        )
                        != generations
                    ),
                )
            finally:
                for worker in workers:
                    worker.training_network_tasks -= 1
                    if worker.physical_state is not None:
                        worker.physical_state.network_tasks -= 1
                for worker, request in zip(workers, requests):
                    worker.gpu.release(request)

            if not result.completed:
                yield from self._wait_for_workers_healthy(workers)
                continue

            actual_duration = result.wall_elapsed
            bandwidth_contention_seconds = (
                result.contention("checkpoint_network")
                + result.contention("physical_nic")
            )
            contention_seconds = result.contention("any")
            for worker in workers:
                self._record_synchronization_wait(
                    worker,
                    start=self.pipeline_ready_at.get(
                        (iteration, worker.rank),
                        start,
                    ),
                    end=start,
                    iteration=iteration,
                    operation="data_parallel_sync_wait",
                    wait_for="data_parallel_all_reduce",
                )
            for worker in workers:
                self._record(
                    worker,
                    start=start,
                    category="Communication",
                    operation="data_parallel_all_reduce",
                    resources=["GPU", "NETWORK"],
                    iteration=iteration,
                    source=worker.name,
                    destination=(
                        f"{self.config.job_id}-stage-"
                        f"{pipeline_stage}-replicas"
                    ),
                    data_gb=gradient_shard_gb,
                    details={
                        "link": (
                            f"{self.config.job_id}/data-parallel/"
                            f"stage-{pipeline_stage}"
                        ),
                        "collective_group": [
                            worker.rank for worker in workers
                        ],
                        "base_duration": duration,
                        "contention_seconds": contention_seconds,
                        "bandwidth_contention_seconds": (
                            bandwidth_contention_seconds
                        ),
                        "checkpoint_upload_contention_seconds": (
                            contention_seconds
                        ),
                        "failure_wait_seconds": 0.0,
                        "effective_slowdown": (
                            actual_duration / duration
                            if duration > 0
                            else 1.0
                        ),
                    },
                )
            return

    def _wait_for_workers_healthy(
        self,
        workers: tuple[JobWorker, ...],
    ):
        while True:
            failed_workers = [worker for worker in workers if worker.failed]
            if not failed_workers:
                return
            recovery_events = [
                worker.recovered_event
                for worker in failed_workers
                if worker.recovered_event is not None
            ]
            if recovery_events:
                yield self.backend.all_of(recovery_events)
            else:
                yield self.backend.timeout(0)

    def _optimizer(self, worker: JobWorker, *, iteration: int):
        duration = self.config.optimizer_seconds / self.config.pipeline_stages
        yield self.backend.process(
            self._gpu_work(
                worker,
                duration=duration,
                iteration=iteration,
                operation=(
                    "data_parallel_optimizer"
                    if self.config.strategy == "data_parallel"
                    else "pipeline_optimizer"
                ),
            )
        )

    def run(self):
        if self.verbose:
            print(
                f"starting {self.config.job_id}: "
                f"strategy={self.config.strategy}, "
                f"DP={self.config.data_parallel_replicas}, "
                f"PP={self.config.pipeline_stages}, "
                f"ranks={self.config.rank_count}"
            )

        for iteration in range(1, self.iterations + 1):
            for worker in self.workers.values():
                worker.current_iteration = iteration
            microbatch_processes = [
                self.backend.process(
                    self._microbatch(
                        data_parallel_rank=dp_rank,
                        iteration=iteration,
                        microbatch=microbatch,
                    )
                )
                for dp_rank in range(self.config.data_parallel_replicas)
                for microbatch in range(self.config.microbatches)
            ]
            yield self.backend.all_of(microbatch_processes)
            yield _all_processes(
                self.backend,
                (
                    self._data_parallel_all_reduce(
                        pipeline_stage=stage,
                        iteration=iteration,
                    )
                    for stage in range(self.config.pipeline_stages)
                ),
            )
            yield _all_processes(
                self.backend,
                (
                    self._optimizer(worker, iteration=iteration)
                    for worker in self.workers.values()
                ),
            )

            if iteration % self.config.checkpoint_every == 0:
                if (
                    self.config.checkpoint_mode == "asynchronous"
                    and self.config.async_checkpoint_backpressure
                    and self.checkpoint_processes
                ):
                    # The measured DeepSpeed harness permits one background
                    # persist per rank.  The next save drains it first.
                    wait_start = self.backend.now
                    yield self.backend.all_of(self.checkpoint_processes)
                    wait_end = self.backend.now
                    if wait_end - wait_start > 1e-12:
                        for worker in self.workers.values():
                            self._record(
                                worker,
                                start=wait_start,
                                end=wait_end,
                                category="Synchronization",
                                operation="checkpoint_async_backpressure",
                                resources=[],
                                iteration=iteration,
                                details={
                                    "status": "idle_wait",
                                    "reason": "backpressure",
                                    "base_duration": wait_end - wait_start,
                                    "effective_slowdown": 1.0,
                                },
                            )
                    self.checkpoint_processes.clear()
                if self.config.checkpoint_strategy == "object_store":
                    # A shared store needs one copy of each pipeline shard.
                    writers = [
                        self.workers[(0, stage)]
                        for stage in range(self.config.pipeline_stages)
                    ]
                else:
                    # Node-local strategies need a recoverable copy per rank.
                    writers = list(self.workers.values())
                checkpoint_processes = []
                foreground_stage_events = []
                for writer in writers:
                    if (
                        self.config.checkpoint_mode == "asynchronous"
                        and self.config.async_checkpoint_backpressure
                        and hasattr(
                            self.checkpoint_strategy,
                            "stage_completion_event",
                        )
                    ):
                        foreground_stage_events.append(
                            self.checkpoint_strategy.stage_completion_event(
                                writer,
                                iteration,
                            )
                        )
                    checkpoint_process = self.backend.process(
                        self.checkpoint_strategy.checkpoint(
                            writer,
                            self.config,
                            iteration=iteration,
                        )
                    )
                    checkpoint_processes.append(checkpoint_process)
                    self.checkpoint_processes.append(checkpoint_process)
                if foreground_stage_events:
                    # DeepSpeed's async save returns after RAM staging, not at
                    # submission time. Persistence continues in the process.
                    yield self.backend.all_of(foreground_stage_events)
                if (
                    checkpoint_processes
                    and self.config.checkpoint_mode == "synchronous"
                ):
                    yield self.backend.all_of(checkpoint_processes)

        if self.checkpoint_processes:
            yield self.backend.all_of(self.checkpoint_processes)
        for worker in self.workers.values():
            worker.active = False


SimulationRuntime = (
    DistributedJobRuntime
    | ChunkedDataParallelRuntime
    | AggregatedDataParallelRuntime
)


@dataclass(slots=True)
class RuntimePlan:
    runtimes: list[SimulationRuntime]
    detailed_runtimes: list[DistributedJobRuntime]
    placements: list[dict[str, Any]]


class ExecutionPlanner:
    """Selects the runtime implementation for each configured job."""

    def __init__(
        self,
        *,
        backend: SimulationBackend,
        run_id: int,
        config: SimulatorConfig,
        logger: EventLogger,
        verbose: bool,
    ) -> None:
        self.backend = backend
        self.run_id = run_id
        self.config = config
        self.logger = logger
        self.verbose = verbose
        self.object_store = backend.resource(
            config.cluster.object_store_concurrency
        )
        self.physical_nodes = {
            f"node-{index}": PhysicalNodeState(
                cpu=backend.priority_resource(config.cluster.cpu_cores_per_node)
            )
            for index in range(config.cluster.node_count)
        }

    def build(self) -> RuntimePlan:
        runtimes: list[SimulationRuntime] = []
        detailed_runtimes: list[DistributedJobRuntime] = []
        placements: list[dict[str, Any]] = []
        node_offset = 0

        for job in self.config.jobs:
            runtime = self._runtime_for(job, node_offset=node_offset)
            runtimes.append(runtime)
            placements.extend(runtime.placement())
            if isinstance(runtime, DistributedJobRuntime):
                detailed_runtimes.append(runtime)
            node_offset += job.rank_count

        return RuntimePlan(
            runtimes=runtimes,
            detailed_runtimes=detailed_runtimes,
            placements=placements,
        )

    def _runtime_for(
        self,
        job: JobConfig,
        *,
        node_offset: int,
    ) -> SimulationRuntime:
        kwargs = {
            "run_id": self.run_id,
            "config": job,
            "simulator_config": self.config,
            "logger": self.logger,
            "node_offset": node_offset,
            "object_store": self.object_store,
            "verbose": self.verbose,
            "backend": self.backend,
        }
        if ChunkedDataParallelRuntime.supports(job):
            return ChunkedDataParallelRuntime(**kwargs)
        if job.rank_count > 10_000:
            return AggregatedDataParallelRuntime(**kwargs)
        return DistributedJobRuntime(
            **kwargs,
            physical_nodes=self.physical_nodes,
        )


def run_configured_jobs(
    env: simpy.Environment,
    *,
    run_id: int,
    config: SimulatorConfig,
    logger: EventLogger,
    verbose: bool,
) -> tuple[simpy.events.AllOf, list[dict[str, Any]]]:
    backend = SimPyBackend(env)
    plan = ExecutionPlanner(
        backend=backend,
        run_id=run_id,
        config=config,
        logger=logger,
        verbose=verbose,
    ).build()

    completion = backend.all_of(
        [backend.process(runtime.run()) for runtime in plan.runtimes]
    )
    failure_controller = FailureController(
        run_id=run_id,
        settings=config.failures,
        logger=logger,
        rng=random.Random(config.simulation.seed + run_id),
        recovery_handlers={
            runtime.config.job_id: runtime.checkpoint_strategy
            for runtime in plan.detailed_runtimes
        },
        backend=backend,
    )
    backend.process(
        failure_controller.monitor(
            (
                (runtime.config, worker)
                for runtime in plan.detailed_runtimes
                for worker in runtime.workers.values()
            )
        )
    )
    return completion, plan.placements
