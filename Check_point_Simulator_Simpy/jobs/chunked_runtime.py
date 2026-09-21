from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from nodes.node import EventLogger
from simulation.backend import SimulationBackend

from .config import JobConfig, SimulatorConfig


@dataclass(slots=True)
class ChunkedWorker:
    job_id: str
    rank: int
    physical_node: str
    scheduler_chunk_index: int
    scheduler_chunk_start: int
    scheduler_chunk_end: int
    scheduler_worker_cores: int
    gpu: Any
    cpu: Any
    network_link: Any
    ring_next_rank: int

    @property
    def name(self) -> str:
        return f"{self.job_id}-rank-{self.rank}"

    @property
    def scheduler_batch(self) -> int:
        return self.scheduler_chunk_index // self.scheduler_worker_cores


class ChunkedDataParallelRuntime:
    """Individual-rank data-parallel runtime processed in bounded chunks."""

    CHUNK_SIZE = 10_000
    MAX_RANKS = 1_000_000
    START_STAGGER_SECONDS = 1e-6
    SUPPORTED_CHECKPOINT_STRATEGIES = {"object_store", "ssd_tiered"}

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
    ) -> None:
        if not self.supports(config):
            raise ValueError(
                "Chunked individual execution supports failure-free "
                "data_parallel jobs up to 1M ranks with object_store or "
                "ssd_tiered checkpoints"
            )
        self.backend = backend
        self.run_id = run_id
        self.config = config
        self.cluster = simulator_config.cluster
        self.iterations = simulator_config.simulation.iterations
        self.logger = logger
        self.node_offset = node_offset
        self.object_store = object_store
        self.verbose = verbose
        self.rank_count = config.rank_count
        self.checkpoint_processes: list[Any] = []
        self.worker_cores = self._available_worker_cores()
        self.chunks = list(self._chunk_ranges())
        self._chunk_max_stagger = (
            (self.CHUNK_SIZE - 1) * self.START_STAGGER_SECONDS
        )

    @classmethod
    def supports(cls, config: JobConfig) -> bool:
        return (
            config.strategy == "data_parallel"
            and cls.CHUNK_SIZE < config.rank_count <= cls.MAX_RANKS
            and config.pipeline_stages == 1
            and config.failure_rate == 0.0
            and config.checkpoint_strategy in cls.SUPPORTED_CHECKPOINT_STRATEGIES
        )

    @classmethod
    def _available_worker_cores(cls) -> int:
        configured = os.environ.get("SIMULATOR_WORKER_CORES")
        if configured is not None:
            try:
                return max(1, int(configured))
            except ValueError:
                return 1
        process_cpu_count = getattr(os, "process_cpu_count", None)
        if process_cpu_count is not None:
            return max(1, int(process_cpu_count() or 1))
        return max(1, int(os.cpu_count() or 1))

    def _chunk_ranges(self) -> Iterable[tuple[int, int, int]]:
        for chunk_index, start in enumerate(
            range(0, self.rank_count, self.CHUNK_SIZE)
        ):
            yield (
                chunk_index,
                start,
                min(start + self.CHUNK_SIZE - 1, self.rank_count - 1),
            )

    def placement(self) -> list[dict[str, Any]]:
        return [
            {
                "job_id": self.config.job_id,
                "rank_start": chunk_start,
                "rank_end": chunk_end,
                "represented_rank_count": chunk_end - chunk_start + 1,
                "data_parallel_rank": f"{chunk_start}-{chunk_end}",
                "pipeline_stage": 0,
                "physical_node": (
                    f"nodes-{self.node_offset + chunk_start}-"
                    f"{self.node_offset + chunk_end}"
                ),
                "scheduler_chunk_index": chunk_index,
                "scheduler_chunk_start": chunk_start,
                "scheduler_chunk_end": chunk_end,
                "scheduler_worker_cores": self.worker_cores,
                "scheduler_batch": chunk_index // self.worker_cores,
                "chunked_individual": True,
                "individual_machine": True,
            }
            for chunk_index, chunk_start, chunk_end in self.chunks
        ]

    def _workers(self) -> Iterable[ChunkedWorker]:
        for chunk_index, chunk_start, chunk_end in self.chunks:
            for rank in range(chunk_start, chunk_end + 1):
                yield ChunkedWorker(
                    job_id=self.config.job_id,
                    rank=rank,
                    physical_node=f"node-{self.node_offset + rank}",
                    scheduler_chunk_index=chunk_index,
                    scheduler_chunk_start=chunk_start,
                    scheduler_chunk_end=chunk_end,
                    scheduler_worker_cores=self.worker_cores,
                    gpu=self.backend.resource(self.cluster.gpus_per_node),
                    cpu=self.backend.priority_resource(
                        self.cluster.cpu_cores_per_node
                    ),
                    network_link=self.backend.resource(),
                    ring_next_rank=(rank + 1) % self.rank_count,
                )

    def _record(
        self,
        worker: ChunkedWorker,
        *,
        start: float,
        end: float,
        category: str,
        operation: str,
        resources: list[str],
        iteration: int,
        source: str | None = None,
        destination: str | None = None,
        data_gb: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        scheduler_details = {
            "scheduler_chunk_index": worker.scheduler_chunk_index,
            "scheduler_chunk_start": worker.scheduler_chunk_start,
            "scheduler_chunk_end": worker.scheduler_chunk_end,
            "scheduler_worker_cores": worker.scheduler_worker_cores,
            "scheduler_batch": worker.scheduler_batch,
            "start_stagger_seconds": self.START_STAGGER_SECONDS,
            "individual_machine": True,
            "ring_next_rank": worker.ring_next_rank,
            **(details or {}),
        }
        self.logger.record(
            start=start,
            end=end,
            run_id=self.run_id,
            job_id=worker.job_id,
            rank=worker.rank,
            node=worker.name,
            physical_node=worker.physical_node,
            pipeline_stage=0,
            data_parallel_rank=worker.rank,
            category=category,
            operation=operation,
            resources=resources,
            iteration=iteration,
            source=source,
            destination=destination,
            data_gb=data_gb,
            details=scheduler_details,
        )

    def _rank_offset(self, worker: ChunkedWorker) -> float:
        return (
            worker.rank - worker.scheduler_chunk_start
        ) * self.START_STAGGER_SECONDS

    def _phase_start(self, phase_start: float, worker: ChunkedWorker, duration: float) -> float:
        wave_duration = duration + self._chunk_max_stagger
        return (
            phase_start
            + worker.scheduler_batch * wave_duration
            + self._rank_offset(worker)
        )

    def _phase(
        self,
        *,
        iteration: int,
        duration: float,
        category: str,
        operation: str,
        resources: list[str],
        data_gb: float | None = None,
        source: str | None = None,
        destination: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        phase_start = self.backend.now
        phase_end = phase_start
        for worker in self._workers():
            worker_start = self._phase_start(phase_start, worker, duration)
            worker_end = worker_start + duration
            phase_end = max(phase_end, worker_end)
            self._record(
                worker,
                start=worker_start,
                end=worker_end,
                category=category,
                operation=operation,
                resources=resources,
                iteration=iteration,
                source=source or worker.name,
                destination=destination,
                data_gb=data_gb,
                details=details,
            )
        yield self.backend.timeout(phase_end - phase_start)

    def _all_reduce(self, *, iteration: int):
        replicas = self.config.data_parallel_replicas
        duration = (
            2
            * (replicas - 1)
            / replicas
            * self.config.gradient_gb
            / self.cluster.network_bandwidth_gbps
        )
        yield from self._phase(
            iteration=iteration,
            duration=duration,
            category="Communication",
            operation="data_parallel_all_reduce",
            resources=["GPU", "NETWORK"],
            data_gb=self.config.gradient_gb,
            destination=f"{self.config.job_id}-all-ranks",
            details={
                "link": f"{self.config.job_id}/chunked-all-reduce",
                "collective_group": f"ranks-0-{self.rank_count - 1}",
                "base_duration": duration,
                "contention_seconds": 0.0,
                "bandwidth_contention_seconds": 0.0,
                "checkpoint_upload_contention_seconds": 0.0,
                "effective_slowdown": 1.0,
            },
        )

    def _object_store_checkpoint(self, *, iteration: int):
        worker = ChunkedWorker(
            job_id=self.config.job_id,
            rank=0,
            physical_node=f"node-{self.node_offset}",
            scheduler_chunk_index=0,
            scheduler_chunk_start=0,
            scheduler_chunk_end=min(self.CHUNK_SIZE - 1, self.rank_count - 1),
            scheduler_worker_cores=self.worker_cores,
            gpu=self.backend.resource(self.cluster.gpus_per_node),
            cpu=self.backend.priority_resource(self.cluster.cpu_cores_per_node),
            network_link=self.backend.resource(),
            ring_next_rank=1 % self.rank_count,
        )
        checkpoint_group = f"{self.config.job_id}-iteration-{iteration}"
        stage_duration = (
            self.config.checkpoint_gb / self.cluster.gpu_cpu_bandwidth_gbps
        )
        start = self.backend.now
        yield self.backend.timeout(stage_duration)
        self._record(
            worker,
            start=start,
            end=self.backend.now,
            category="Checkpoint",
            operation="checkpoint_stage_gpu_to_dram",
            resources=["CPU", "GPU"],
            iteration=iteration,
            data_gb=self.config.checkpoint_gb,
            details={
                "checkpoint_strategy": "object_store",
                "checkpoint_group": checkpoint_group,
                "shard": 0,
                "shard_count": 1,
                "base_duration": stage_duration,
                "effective_slowdown": 1.0,
            },
        )

        with self.object_store.request() as store_request:
            yield store_request
            upload_duration = (
                self.config.checkpoint_gb
                / self.cluster.object_store_bandwidth_gbps
            )
            start = self.backend.now
            yield self.backend.timeout(upload_duration)
            self._record(
                worker,
                start=start,
                end=self.backend.now,
                category="Checkpoint",
                operation="checkpoint_stage_dram_to_object_store",
                resources=["CPU", "NETWORK"],
                iteration=iteration,
                source=worker.name,
                destination="object-store",
                data_gb=self.config.checkpoint_gb,
                details={
                    "link": f"{self.config.job_id}/object-store",
                    "checkpoint_strategy": "object_store",
                    "checkpoint_group": checkpoint_group,
                    "shard": 0,
                    "shard_count": 1,
                    "base_duration": upload_duration,
                    "effective_slowdown": 1.0,
                    "overlap_gpu_slowdown": (
                        self.config.checkpoint_upload_gpu_slowdown
                    ),
                },
            )

    def _ssd_tiered_checkpoint(self, *, iteration: int):
        checkpoint_group = f"{self.config.job_id}-iteration-{iteration}"
        yield from self._phase(
            iteration=iteration,
            duration=(
                self.config.checkpoint_gb
                / self.cluster.gpu_cpu_bandwidth_gbps
            ),
            category="Checkpoint",
            operation="checkpoint_stage_gpu_to_dram",
            resources=["CPU", "GPU"],
            data_gb=self.config.checkpoint_gb,
            details={
                "checkpoint_strategy": "ssd_tiered",
                "checkpoint_group": checkpoint_group,
                "base_duration": (
                    self.config.checkpoint_gb
                    / self.cluster.gpu_cpu_bandwidth_gbps
                ),
                "effective_slowdown": 1.0,
            },
        )
        yield from self._phase(
            iteration=iteration,
            duration=(
                self.config.checkpoint_gb
                / self.cluster.local_ssd_bandwidth_gbps
            ),
            category="Checkpoint",
            operation="checkpoint_dram_to_local_ssd_chunk",
            resources=["CPU"],
            data_gb=self.config.checkpoint_gb,
            details={
                "checkpoint_strategy": "ssd_tiered",
                "chunk_index": 1,
                "chunk_count": 1,
                "effective_slowdown": 1.0,
            },
        )
        peer_data_gb = self.config.checkpoint_gb * 2
        yield from self._phase(
            iteration=iteration,
            duration=peer_data_gb / self.cluster.network_bandwidth_gbps,
            category="Checkpoint",
            operation="checkpoint_dram_to_peer_dram_chunk",
            resources=["CPU", "NETWORK"],
            data_gb=peer_data_gb,
            destination="paired-rank/peer-dram",
            details={
                "checkpoint_strategy": "ssd_tiered",
                "peer_pair_checkpoint_gb": peer_data_gb,
                "chunk_index": 1,
                "chunk_count": 1,
                "bandwidth_contention_seconds": 0.0,
                "effective_slowdown": 1.0,
            },
        )
        yield from self._phase(
            iteration=iteration,
            duration=(
                self.config.checkpoint_gb
                / self.cluster.local_ssd_bandwidth_gbps
            ),
            category="Checkpoint",
            operation="checkpoint_peer_dram_to_peer_ssd_chunk",
            resources=["CPU"],
            data_gb=self.config.checkpoint_gb,
            source="paired-rank/peer-dram",
            destination="paired-rank/peer-ssd",
            details={
                "checkpoint_strategy": "ssd_tiered",
                "chunk_index": 1,
                "chunk_count": 1,
                "effective_slowdown": 1.0,
            },
        )

    def _checkpoint(self, *, iteration: int):
        if self.config.checkpoint_strategy == "object_store":
            yield from self._object_store_checkpoint(iteration=iteration)
            return
        yield from self._ssd_tiered_checkpoint(iteration=iteration)

    def run(self):
        if self.verbose:
            print(
                f"starting {self.config.job_id}: "
                f"strategy=data_parallel, ranks={self.rank_count:,}, "
                f"execution=chunked individual ranks ({self.CHUNK_SIZE:,}/chunk)"
            )

        for iteration in range(1, self.iterations + 1):
            yield from self._phase(
                iteration=iteration,
                duration=self.config.forward_seconds,
                category="Compute",
                operation="data_parallel_forward",
                resources=["GPU"],
                details={
                    "status": "completed",
                    "attempt": 1,
                    "base_duration": self.config.forward_seconds,
                    "completed_fraction": 1.0,
                    "contention_seconds": 0.0,
                    "checkpoint_upload_contention_seconds": 0.0,
                    "effective_slowdown": 1.0,
                },
            )
            yield from self._phase(
                iteration=iteration,
                duration=self.config.backward_seconds,
                category="Compute",
                operation="data_parallel_backward",
                resources=["GPU"],
                details={
                    "status": "completed",
                    "attempt": 1,
                    "base_duration": self.config.backward_seconds,
                    "completed_fraction": 1.0,
                    "contention_seconds": 0.0,
                    "checkpoint_upload_contention_seconds": 0.0,
                    "effective_slowdown": 1.0,
                },
            )
            yield from self._all_reduce(iteration=iteration)
            yield from self._phase(
                iteration=iteration,
                duration=self.config.optimizer_seconds,
                category="Compute",
                operation="data_parallel_optimizer",
                resources=["GPU"],
                details={
                    "status": "completed",
                    "attempt": 1,
                    "base_duration": self.config.optimizer_seconds,
                    "completed_fraction": 1.0,
                    "contention_seconds": 0.0,
                    "checkpoint_upload_contention_seconds": 0.0,
                    "effective_slowdown": 1.0,
                },
            )
            if iteration % self.config.checkpoint_every == 0:
                checkpoint_process = self.backend.process(
                    self._checkpoint(iteration=iteration)
                )
                if self.config.checkpoint_mode == "synchronous":
                    yield checkpoint_process
                else:
                    self.checkpoint_processes.append(checkpoint_process)
        if self.checkpoint_processes:
            yield self.backend.all_of(self.checkpoint_processes)
