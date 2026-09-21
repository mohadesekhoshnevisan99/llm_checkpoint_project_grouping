from __future__ import annotations

import random
from typing import Any, Iterable, Protocol

from nodes.node import EventLogger
from simulation.backend import SimulationBackend

from .config import FailureSettings, JobConfig


FAILURE_SEVERITY = {
    "process": 1,
    "node": 2,
    "spot": 3,
}


class FailureWorker(Protocol):
    job_id: str
    rank: int
    data_parallel_rank: int
    pipeline_stage: int
    physical_node: str
    failed: bool
    failure_type: str | None
    failure_started: float | None
    failed_until: float
    failure_signal_count: int
    failure_batch_size: int
    failure_generation: int
    recovered_event: Any | None
    active_checkpoint_gb: float
    active: bool
    current_iteration: int
    failure_types_seen: list[str]
    failure_dram_flushed_gb: float
    failure_ssd_flushed_gb: float

    @property
    def name(self) -> str: ...


class RecoveryHandler(Protocol):
    def handle_failure(
        self,
        worker: FailureWorker,
        failure_type: str,
    ) -> dict[str, float]: ...

    def recover(self, worker: FailureWorker, job: JobConfig): ...


class FailureController:
    """Sample every node independently at each integer simulation second."""

    def __init__(
        self,
        *,
        run_id: int,
        settings: FailureSettings,
        logger: EventLogger,
        rng: random.Random,
        recovery_handlers: dict[str, RecoveryHandler],
        backend: SimulationBackend,
    ) -> None:
        self.backend = backend
        self.run_id = run_id
        self.settings = settings
        self.logger = logger
        self.rng = rng
        self.recovery_handlers = recovery_handlers
        self.restart_seconds = {
            "process": settings.process_restart_seconds,
            "node": settings.node_restart_seconds,
            "spot": settings.spot_restart_seconds,
        }

    def monitor(
        self,
        targets: Iterable[tuple[JobConfig, FailureWorker]],
    ):
        target_list = list(targets)
        while True:
            yield self.backend.timeout(1.0)
            triggered: list[tuple[JobConfig, FailureWorker, str]] = []
            for job, worker in target_list:
                if not worker.active:
                    continue
                if job.failure_rate <= 0.0:
                    continue
                if self.rng.random() >= job.failure_rate:
                    continue
                triggered.append((job, worker, self._sample_failure_type()))

            batch_size = len(triggered)
            for job, worker, failure_type in triggered:
                self._apply_failure(
                    job,
                    worker,
                    failure_type=failure_type,
                    batch_size=batch_size,
                )

    def _sample_failure_type(self) -> str:
        return self.rng.choices(
            ("process", "node", "spot"),
            weights=(
                self.settings.process_weight,
                self.settings.node_weight,
                self.settings.spot_weight,
            ),
            k=1,
        )[0]

    def _apply_failure(
        self,
        job: JobConfig,
        worker: FailureWorker,
        *,
        failure_type: str,
        batch_size: int,
    ) -> None:
        restart_until = self.backend.now + self.restart_seconds[failure_type]
        worker.failure_generation += 1
        flushed = self.recovery_handlers[job.job_id].handle_failure(
            worker,
            failure_type,
        )

        if not worker.failed:
            worker.failed = True
            worker.failure_type = failure_type
            worker.failure_started = self.backend.now
            worker.failed_until = restart_until
            worker.failure_signal_count = 1
            worker.failure_batch_size = batch_size
            worker.failure_types_seen = [failure_type]
            worker.failure_dram_flushed_gb = flushed["dram_flushed_gb"]
            worker.failure_ssd_flushed_gb = flushed["ssd_flushed_gb"]
            worker.recovered_event = self.backend.event()
            self.backend.process(self._recover(job, worker))
            return

        if worker.failure_started is None:
            worker.failure_type = failure_type
            worker.failure_started = self.backend.now
            worker.failed_until = restart_until
            worker.failure_signal_count = 1
            worker.failure_batch_size = batch_size
            worker.failure_types_seen = [failure_type]
            worker.failure_dram_flushed_gb = flushed["dram_flushed_gb"]
            worker.failure_ssd_flushed_gb = flushed["ssd_flushed_gb"]
            return

        worker.failure_signal_count += 1
        worker.failure_types_seen.append(failure_type)
        worker.failure_batch_size = max(worker.failure_batch_size, batch_size)
        current_type = worker.failure_type or "process"
        if FAILURE_SEVERITY[failure_type] > FAILURE_SEVERITY[current_type]:
            worker.failure_type = failure_type
        worker.failure_dram_flushed_gb = max(
            worker.failure_dram_flushed_gb,
            flushed["dram_flushed_gb"],
        )
        worker.failure_ssd_flushed_gb = max(
            worker.failure_ssd_flushed_gb,
            flushed["ssd_flushed_gb"],
        )
        worker.failed_until = max(worker.failed_until, restart_until)

    def _recover(self, job: JobConfig, worker: FailureWorker):
        recovery_handler = self.recovery_handlers[job.job_id]
        while True:
            if worker.failure_started is not None:
                while self.backend.now < worker.failed_until:
                    yield self.backend.timeout(
                        worker.failed_until - self.backend.now
                    )
                self._record_restart(job, worker)

            recovery_generation = worker.failure_generation
            recovered = yield self.backend.process(
                recovery_handler.recover(worker, job)
            )
            if (
                recovered
                and worker.failure_generation == recovery_generation
                and worker.failure_started is None
            ):
                break

        recovered_event = worker.recovered_event
        worker.failed = False
        worker.failure_type = None
        worker.failure_started = None
        worker.failed_until = self.backend.now
        worker.failure_signal_count = 0
        worker.failure_batch_size = 0
        worker.failure_types_seen = []
        worker.failure_dram_flushed_gb = 0.0
        worker.failure_ssd_flushed_gb = 0.0
        worker.recovered_event = None
        if recovered_event is not None and not recovered_event.triggered:
            recovered_event.succeed(True)

    def _record_restart(
        self,
        job: JobConfig,
        worker: FailureWorker,
    ) -> None:
        start = float(worker.failure_started or self.backend.now)
        failure_type = worker.failure_type or "process"
        end = self.backend.now
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
            category="Failure",
            operation=f"{failure_type}_failure_restart",
            resources=["CPU", "GPU"],
            iteration=worker.current_iteration,
            failure_type=failure_type,
            details={
                "highest_severity": failure_type,
                "observed_failure_types": list(worker.failure_types_seen),
                "coalesced_failure_signals": worker.failure_signal_count,
                "simultaneous_failed_nodes": worker.failure_batch_size,
                "dram_flushed_gb": worker.failure_dram_flushed_gb,
                "ssd_flushed_gb": worker.failure_ssd_flushed_gb,
                "base_duration": end - start,
                "effective_slowdown": 1.0,
            },
        )
        worker.failure_type = None
        worker.failure_started = None
        worker.failure_signal_count = 0
        worker.failure_batch_size = 0
        worker.failure_types_seen = []
        worker.failure_dram_flushed_gb = 0.0
        worker.failure_ssd_flushed_gb = 0.0
