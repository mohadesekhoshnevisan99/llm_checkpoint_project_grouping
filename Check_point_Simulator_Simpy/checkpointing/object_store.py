from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nodes.node import EventLogger
from simulation.backend import SimulationBackend
from simulation.progress import advance_work

from .base import CheckpointWorker

if TYPE_CHECKING:
    from jobs.config import ClusterConfig, JobConfig


class ObjectStoreCheckpointStrategy:
    """
    Stage a checkpoint shard GPU -> DRAM, then upload DRAM -> object store.

    GPU staging blocks the writer's GPU. The upload releases the GPU so
    training can continue, but marks the writer as upload-contended. The job
    runtime uses that state to slow overlapping GPU work dynamically.
    """

    def __init__(
        self,
        *,
        run_id: int,
        cluster: ClusterConfig,
        logger: EventLogger,
        object_store: Any,
        backend: SimulationBackend,
        contention_quantum_seconds: float = 0.05,
    ) -> None:
        self.backend = backend
        self.run_id = run_id
        self.cluster = cluster
        self.logger = logger
        self.object_store = object_store
        self.contention_quantum_seconds = contention_quantum_seconds
        self.latest_checkpoint_iterations: dict[tuple[str, int], int] = {}

    def gpu_slowdown(
        self,
        worker: CheckpointWorker,
        job: JobConfig,
    ) -> float:
        if worker.checkpoint_uploads <= 0:
            return 1.0
        return job.checkpoint_upload_gpu_slowdown

    def network_slowdown(self, worker: CheckpointWorker) -> float:
        del worker
        return 1.0

    def handle_failure(
        self,
        worker: CheckpointWorker,
        failure_type: str,
    ) -> dict[str, float]:
        """Remote copies survive; only an in-flight DRAM stage can be lost."""

        return {
            "dram_flushed_gb": (
                worker.active_checkpoint_gb
                if failure_type in {"node", "spot"}
                else 0.0
            ),
            "ssd_flushed_gb": 0.0,
        }

    def checkpoint(
        self,
        worker: CheckpointWorker,
        job: JobConfig,
        *,
        iteration: int,
    ):
        shard_gb = job.checkpoint_gb / job.pipeline_stages
        checkpoint_group = f"{job.job_id}-iteration-{iteration}"
        gpu_to_dram = shard_gb / self.cluster.gpu_cpu_bandwidth_gbps
        worker.active_checkpoint_gb = shard_gb
        try:
            while True:
                yield from self._wait_until_healthy(worker)
                gpu_request = worker.gpu.request()
                cpu_request = worker.cpu.request()
                yield self.backend.all_of([gpu_request, cpu_request])
                if worker.failed:
                    worker.gpu.release(gpu_request)
                    worker.cpu.release(cpu_request)
                    continue
                attempt_generation = worker.failure_generation
                start = self.backend.now
                try:
                    stage_completed = yield from self._available_timeout(
                        worker,
                        generation=attempt_generation,
                        duration=gpu_to_dram,
                    )
                finally:
                    worker.gpu.release(gpu_request)

                if (
                    not stage_completed
                    or worker.failure_generation != attempt_generation
                ):
                    worker.cpu.release(cpu_request)
                    continue

                actual_stage_duration = self.backend.now - start
                self._record(
                    worker,
                    start=start,
                    category="Checkpoint",
                    operation="checkpoint_stage_gpu_to_dram",
                    resources=["CPU", "GPU"],
                    iteration=iteration,
                    data_gb=shard_gb,
                    details={
                        "checkpoint_strategy": "object_store",
                        "checkpoint_mode": job.checkpoint_mode,
                        "checkpoint_group": checkpoint_group,
                        "shard": worker.pipeline_stage,
                        "shard_count": job.pipeline_stages,
                        "base_duration": gpu_to_dram,
                        "failure_wait_seconds": 0.0,
                        "effective_slowdown": (
                            actual_stage_duration / gpu_to_dram
                        ),
                    },
                )
                worker.cpu.release(cpu_request)

                upload_succeeded = False
                with self.object_store.request() as store_request:
                    yield store_request
                    upload_cpu_request = worker.cpu.request()
                    yield upload_cpu_request
                    try:
                        if (
                            worker.failed
                            or worker.failure_generation
                            != attempt_generation
                        ):
                            continue
                        upload_generation = worker.failure_generation
                        start = self.backend.now
                        upload_duration = (
                            shard_gb
                            / self.cluster.object_store_bandwidth_gbps
                        )
                        worker.checkpoint_uploads += 1
                        try:
                            upload_completed = (
                                yield from self._available_timeout(
                                    worker,
                                    generation=upload_generation,
                                    duration=upload_duration,
                                )
                            )
                        finally:
                            worker.checkpoint_uploads -= 1

                        if (
                            not upload_completed
                            or worker.failure_generation != upload_generation
                        ):
                            continue

                        actual_upload_duration = self.backend.now - start
                        self._record(
                            worker,
                            start=start,
                            category="Checkpoint",
                            operation=(
                                "checkpoint_stage_dram_to_object_store"
                            ),
                            resources=["CPU", "NETWORK"],
                            iteration=iteration,
                            source=worker.name,
                            destination="object-store",
                            data_gb=shard_gb,
                            details={
                                "link": (
                                    f"{job.job_id}/"
                                    f"{worker.physical_node}/object-store"
                                ),
                                "checkpoint_strategy": "object_store",
                                "checkpoint_mode": job.checkpoint_mode,
                                "checkpoint_group": checkpoint_group,
                                "shard": worker.pipeline_stage,
                                "shard_count": job.pipeline_stages,
                                "base_duration": upload_duration,
                                "failure_wait_seconds": 0.0,
                                "effective_slowdown": (
                                    actual_upload_duration
                                    / upload_duration
                                ),
                                "overlap_gpu_slowdown": (
                                    job.checkpoint_upload_gpu_slowdown
                                ),
                            },
                        )
                        checkpoint_key = (
                            job.job_id,
                            worker.pipeline_stage,
                        )
                        self.latest_checkpoint_iterations[checkpoint_key] = max(
                            iteration,
                            self.latest_checkpoint_iterations.get(
                                checkpoint_key,
                                0,
                            ),
                        )
                        upload_succeeded = True
                    finally:
                        worker.cpu.release(upload_cpu_request)

                if upload_succeeded:
                    break
        finally:
            worker.active_checkpoint_gb = 0.0

    def recover(
        self,
        worker: CheckpointWorker,
        job: JobConfig,
    ):
        """Restore this rank's latest completed stage shard."""

        generation = worker.failure_generation
        shard_gb = job.checkpoint_gb / job.pipeline_stages
        checkpoint_iteration = self.latest_checkpoint_iterations.get(
            (job.job_id, worker.pipeline_stage),
            0,
        )

        with self.object_store.request() as store_request:
            yield store_request
            cpu_request = worker.cpu.request()
            yield cpu_request
            if worker.failure_generation != generation:
                worker.cpu.release(cpu_request)
                return False

            start = self.backend.now
            download_duration = (
                shard_gb / self.cluster.object_store_bandwidth_gbps
            )
            completed = yield from self._generation_timeout(
                worker,
                generation=generation,
                duration=download_duration,
            )
            if not completed:
                worker.cpu.release(cpu_request)
                return False
            self._record(
                worker,
                start=start,
                category="Recovery",
                operation="object_store_to_dram",
                resources=["CPU", "NETWORK"],
                iteration=checkpoint_iteration,
                source="object-store",
                destination=worker.name,
                data_gb=shard_gb,
                details={
                    "link": (
                        f"{job.job_id}/{worker.physical_node}/object-store"
                    ),
                    "checkpoint_strategy": "object_store",
                    "checkpoint_mode": job.checkpoint_mode,
                    "checkpoint_iteration": checkpoint_iteration,
                    "base_duration": download_duration,
                    "effective_slowdown": 1.0,
                },
            )

        gpu_request = worker.gpu.request()
        yield gpu_request
        if worker.failure_generation != generation:
            worker.gpu.release(gpu_request)
            worker.cpu.release(cpu_request)
            return False

        start = self.backend.now
        restore_duration = shard_gb / self.cluster.gpu_cpu_bandwidth_gbps
        completed = yield from self._generation_timeout(
            worker,
            generation=generation,
            duration=restore_duration,
        )
        if completed:
            self._record(
                worker,
                start=start,
                category="Recovery",
                operation="dram_to_gpu_restore",
                resources=["CPU", "GPU"],
                iteration=checkpoint_iteration,
                source=worker.name,
                destination=worker.name,
                data_gb=shard_gb,
                details={
                    "checkpoint_strategy": "object_store",
                    "checkpoint_mode": job.checkpoint_mode,
                    "checkpoint_iteration": checkpoint_iteration,
                    "base_duration": restore_duration,
                    "effective_slowdown": 1.0,
                },
            )
        worker.gpu.release(gpu_request)
        worker.cpu.release(cpu_request)
        return completed

    def _generation_timeout(
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

    def _available_timeout(
        self,
        worker: CheckpointWorker,
        *,
        generation: int,
        duration: float,
    ):
        result = yield from advance_work(
            self.backend,
            base_duration=duration,
            aborted=lambda: (
                worker.failed or worker.failure_generation != generation
            ),
            quantum=self.contention_quantum_seconds,
        )
        return result.completed

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
        details: dict | None = None,
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
