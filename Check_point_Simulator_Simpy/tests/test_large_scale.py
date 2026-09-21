from __future__ import annotations

from dataclasses import dataclass

import simpy

from jobs.config import (
    ClusterConfig,
    FailureSettings,
    JobConfig,
    SimulationSettings,
    SimulatorConfig,
)
from jobs.runtime import run_configured_jobs
from nodes.node import EventLogger


@dataclass(slots=True)
class CountingLogger:
    event_count: int = 0
    by_operation: dict[str, int] | None = None
    first_rank: int | None = None
    last_rank: int | None = None
    max_chunk_size: int = 0
    batches: set[int] | None = None

    def __post_init__(self) -> None:
        self.by_operation = {}
        self.batches = set()

    def record(self, **kwargs) -> None:
        details = kwargs.get("details") or {}
        operation = str(kwargs["operation"])
        rank = int(kwargs["rank"])

        self.event_count += 1
        assert self.by_operation is not None
        assert self.batches is not None
        self.by_operation[operation] = self.by_operation.get(operation, 0) + 1
        self.first_rank = (
            rank if self.first_rank is None else min(self.first_rank, rank)
        )
        self.last_rank = (
            rank if self.last_rank is None else max(self.last_rank, rank)
        )
        self.max_chunk_size = max(
            self.max_chunk_size,
            int(details["scheduler_chunk_end"])
            - int(details["scheduler_chunk_start"])
            + 1,
        )
        self.batches.add(int(details["scheduler_batch"]))
        assert details["individual_machine"]


def make_large_data_parallel_config(
    *,
    rank_count: int,
    iterations: int,
    checkpoint_every: int,
) -> SimulatorConfig:
    return SimulatorConfig(
        simulation=SimulationSettings(seed=7, iterations=iterations),
        cluster=ClusterConfig(
            node_count=rank_count,
            cpu_cores_per_node=1,
            cpu_memory_gb=3.75,
            gpus_per_node=1,
            gpu_memory_gb=80.0,
            gpu_cpu_bandwidth_gbps=31.5,
            network_bandwidth_gbps=25.0,
            communication_launch_seconds=0.01,
            local_ssd_bandwidth_gbps=4.0,
            object_store_bandwidth_gbps=8.0,
            object_store_concurrency=4,
        ),
        failures=FailureSettings(
            probability_per_second=0.0,
            process_weight=1.0,
            node_weight=0.0,
            spot_weight=0.0,
            process_restart_seconds=3.0,
            node_restart_seconds=8.0,
            spot_restart_seconds=15.0,
        ),
        jobs=(
            JobConfig(
                job_id=f"data-object-store-{rank_count}",
                strategy="data_parallel",
                data_parallel_replicas=rank_count,
                pipeline_stages=1,
                microbatches=1,
                model_weights_gb=26.0,
                gradient_gb=26.0,
                activation_gb=2.0,
                checkpoint_gb=30.0,
                forward_seconds=5.0,
                backward_seconds=9.0,
                optimizer_seconds=1.0,
                checkpoint_every=checkpoint_every,
                checkpoint_strategy="object_store",
                checkpoint_mode="asynchronous",
                checkpoint_chunk_gb=4.0,
                checkpoint_upload_gpu_slowdown=1.15,
                failure_rate=0.0,
            ),
        ),
    )


def test_100k_data_parallel_nodes_run_as_chunked_individual_workers(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SIMULATOR_WORKER_CORES", "2")
    rank_count = 100_000
    config = make_large_data_parallel_config(
        rank_count=rank_count,
        iterations=1,
        checkpoint_every=1,
    )

    env = simpy.Environment()
    logger = EventLogger()
    completion, placements = run_configured_jobs(
        env,
        run_id=0,
        config=config,
        logger=logger,
        verbose=False,
    )
    env.run(until=completion)

    assert len(placements) == 10
    assert sum(
        placement["represented_rank_count"] for placement in placements
    ) == rank_count
    assert all(
        placement["individual_machine"]
        and placement["chunked_individual"]
        and placement["scheduler_worker_cores"] == 2
        for placement in placements
    )
    assert {placement["scheduler_chunk_index"] for placement in placements} == set(
        range(10)
    )
    assert {placement["scheduler_batch"] for placement in placements} == set(
        range(5)
    )

    assert {
        event.operation for event in logger.events
    } >= {
        "data_parallel_forward",
        "data_parallel_backward",
        "data_parallel_all_reduce",
        "data_parallel_optimizer",
        "checkpoint_stage_gpu_to_dram",
        "checkpoint_stage_dram_to_object_store",
    }
    assert len(logger.events) == 400_002
    first_forward = [
        event
        for event in logger.events
        if event.iteration == 1
        and event.operation == "data_parallel_forward"
    ]
    assert len(first_forward) == rank_count
    assert {event.rank for event in first_forward} == set(range(rank_count))
    assert sorted(
        {event.details["scheduler_batch"] for event in first_forward}
    ) == [0, 1, 2, 3, 4]
    assert all(event.details["individual_machine"] for event in first_forward)

    chunk_0_end = max(
        event.end
        for event in first_forward
        if event.details["scheduler_chunk_index"] == 0
    )
    chunk_2_start = min(
        event.start
        for event in first_forward
        if event.details["scheduler_chunk_index"] == 2
    )
    assert chunk_2_start >= chunk_0_end


def test_1m_data_parallel_nodes_execute_as_individual_chunked_workers(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SIMULATOR_WORKER_CORES", "2")
    rank_count = 1_000_000
    config = make_large_data_parallel_config(
        rank_count=rank_count,
        iterations=1,
        checkpoint_every=2,
    )

    env = simpy.Environment()
    logger = CountingLogger()
    completion, placements = run_configured_jobs(
        env,
        run_id=0,
        config=config,
        logger=logger,
        verbose=False,
    )
    env.run(until=completion)

    assert len(placements) == 100
    assert sum(
        placement["represented_rank_count"] for placement in placements
    ) == rank_count
    assert logger.event_count == 4_000_000
    assert logger.by_operation == {
        "data_parallel_forward": rank_count,
        "data_parallel_backward": rank_count,
        "data_parallel_all_reduce": rank_count,
        "data_parallel_optimizer": rank_count,
    }
    assert logger.first_rank == 0
    assert logger.last_rank == rank_count - 1
    assert logger.max_chunk_size == 10_000
    assert logger.batches == set(range(50))
