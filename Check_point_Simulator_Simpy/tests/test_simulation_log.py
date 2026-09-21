from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import simpy

from checkpointing.tiered import TieredCheckpointStrategy
from jobs import load_config, run_configured_jobs
from jobs.runtime import JobWorker
from nodes.node import EventLogger
from simulation import SimPyBackend
from visualise import (
    build_operation_styles,
    make_resource_timeline,
    write_dashboard,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "jobs.toml"
FAILURE_SEVERITY = {"process": 1, "node": 2, "spot": 3}
DATA_JOBS = {
    "data-parallel-13b",
    "data-parallel-13b-no-fail",
    "data-cpu-tiered-30b",
    "data-cpu-tiered-30b-no-fail",
    "data-ssd-tiered-30b",
    "data-ssd-tiered-30b-no-fail",
    "data-ssd-peer-1m",
}
AGGREGATE_JOBS = {"data-ssd-peer-1m"}
DETAILED_DATA_JOBS = DATA_JOBS - AGGREGATE_JOBS
PIPELINE_JOBS = {
    "pipeline-parallel-70b",
    "pipeline-parallel-70b-no-fail",
    "pipeline-paired-tiered-30b",
}
HYBRID_JOBS = {
    "hybrid-30b",
    "hybrid-30b-no-fail",
}
FAILURE_JOBS = {
    "data-parallel-13b",
    "pipeline-parallel-70b",
    "hybrid-30b",
    "data-cpu-tiered-30b",
    "data-ssd-tiered-30b",
    "data-ssd-peer-1m",
    "pipeline-paired-tiered-30b",
}
ALL_JOBS = DATA_JOBS | PIPELINE_JOBS | HYBRID_JOBS
OBJECT_STORE_JOBS = {
    "data-parallel-13b",
    "pipeline-parallel-70b",
    "hybrid-30b",
    "data-parallel-13b-no-fail",
    "pipeline-parallel-70b-no-fail",
    "hybrid-30b-no-fail",
}
CPU_TIERED_JOBS = {
    "data-cpu-tiered-30b",
    "data-cpu-tiered-30b-no-fail",
}
SSD_TIERED_JOBS = {
    "data-ssd-tiered-30b",
    "data-ssd-tiered-30b-no-fail",
}
PAIRED_TIERED_JOB = "pipeline-paired-tiered-30b"


@pytest.fixture(scope="module")
def simulation_log(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    config = load_config(CONFIG_PATH)
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

    represented_nodes = sum(
        int(placement.get("represented_rank_count", 1))
        for placement in placements
    )
    assert represented_nodes == config.required_nodes
    output = tmp_path_factory.mktemp("simulation") / "events.jsonl"
    logger.write_jsonl(output)
    return [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]


def events_by_job(events: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        grouped[event["job_id"]].append(event)
    return grouped


def test_jsonl_event_integrity(simulation_log: list[dict]) -> None:
    assert simulation_log
    assert len({event["event_id"] for event in simulation_log}) == len(
        simulation_log
    )
    assert simulation_log == sorted(
        simulation_log,
        key=lambda event: (
            event["start"],
            event["end"],
            event["event_id"],
        ),
    )

    for event in simulation_log:
        assert event["end"] >= event["start"]
        assert event["duration"] == pytest.approx(
            event["end"] - event["start"],
            abs=2e-9,
        )
        assert event["job_id"]
        assert event["physical_node"]
        assert event["rank"] is not None
        assert event["pipeline_stage"] is not None
        assert event["data_parallel_rank"] is not None


def test_each_job_uses_its_configured_parallelism(
    simulation_log: list[dict],
) -> None:
    grouped = events_by_job(simulation_log)
    assert set(grouped) == ALL_JOBS

    for job_id in DATA_JOBS:
        data_operations = {
            event["operation"] for event in grouped[job_id]
        }
        assert "data_parallel_all_reduce" in data_operations
        assert "pipeline_activation_send" not in data_operations

    for job_id in PIPELINE_JOBS:
        pipeline_operations = {
            event["operation"] for event in grouped[job_id]
        }
        assert "pipeline_activation_send" in pipeline_operations
        assert "pipeline_gradient_send" in pipeline_operations
        assert "data_parallel_all_reduce" not in pipeline_operations

    for job_id in HYBRID_JOBS:
        hybrid_operations = {
            event["operation"] for event in grouped[job_id]
        }
        assert "pipeline_activation_send" in hybrid_operations
        assert "data_parallel_all_reduce" in hybrid_operations


def test_checkpoint_shards_complete_and_uploads_slow_training(
    simulation_log: list[dict],
) -> None:
    grouped = events_by_job(simulation_log)
    expected_uploads = {
        "data-parallel-13b": 10,
        "data-parallel-13b-no-fail": 10,
        "pipeline-parallel-70b": 40,
        "pipeline-parallel-70b-no-fail": 40,
        "hybrid-30b": 20,
        "hybrid-30b-no-fail": 20,
    }
    expected_writers = {
        "data-parallel-13b": {0},
        "data-parallel-13b-no-fail": {0},
        "pipeline-parallel-70b": {0, 1, 2, 3},
        "pipeline-parallel-70b-no-fail": {0, 1, 2, 3},
        "hybrid-30b": {0, 1},
        "hybrid-30b-no-fail": {0, 1},
    }

    for job_id in OBJECT_STORE_JOBS:
        job_events = grouped[job_id]
        uploads = [
            event
            for event in job_events
            if event["operation"]
            == "checkpoint_stage_dram_to_object_store"
        ]
        assert len(uploads) == expected_uploads[job_id]
        assert {event["rank"] for event in uploads} == expected_writers[job_id]

        slowed = [
            event
            for event in job_events
            if (event.get("details") or {}).get(
                "checkpoint_upload_contention_seconds",
                0.0,
            )
            > 0.0
            and (event.get("details") or {}).get("status")
            != "interrupted"
        ]
        if job_id in expected_uploads:
            assert slowed
            assert all(
                event["details"]["effective_slowdown"] > 1.0
                for event in slowed
            )

        uploads_by_iteration: dict[int, list[dict]] = defaultdict(list)
        for upload in uploads:
            uploads_by_iteration[upload["iteration"]].append(upload)
        assert set(uploads_by_iteration) == set(range(2, 21, 2))
        for iteration_uploads in uploads_by_iteration.values():
            assert len(
                {
                    (
                        event["rank"],
                        event["details"]["shard"],
                    )
                    for event in iteration_uploads
                }
            ) == len(iteration_uploads)


def test_tiered_checkpoints_are_chunked_and_rank_paired(
    simulation_log: list[dict],
) -> None:
    grouped = events_by_job(simulation_log)
    expected_checkpoints = {
        (rank, iteration)
        for rank in range(4)
        for iteration in range(2, 21, 2)
    }

    for job_id in SSD_TIERED_JOBS | {PAIRED_TIERED_JOB}:
        job_events = grouped[job_id]
        local_chunks = [
            event
            for event in job_events
            if event["operation"] == "checkpoint_dram_to_local_ssd_chunk"
        ]
        assert local_chunks
        assert all(event["data_gb"] <= 4.0 for event in local_chunks)
        assert all(event["resources"] == ["CPU"] for event in local_chunks)
        assert all(
            event["details"]["chunk_count"] > 1 for event in local_chunks
        )
        completed_local = {
            (
                event["details"]["checkpoint_owner_rank"],
                event["iteration"],
            )
            for event in local_chunks
            if event["details"]["chunk_index"]
            == event["details"]["chunk_count"]
        }
        assert completed_local == expected_checkpoints

    for job_id in CPU_TIERED_JOBS:
        job_events = grouped[job_id]
        assert not any(
            event["operation"] == "checkpoint_dram_to_local_ssd_chunk"
            for event in job_events
        )
        completed_dram = {
            (event["rank"], event["iteration"])
            for event in job_events
            if event["operation"] == "checkpoint_stage_gpu_to_dram"
        }
        assert completed_dram == expected_checkpoints
        completed_peer_dram = {
            (
                event["details"]["checkpoint_owner_rank"],
                event["iteration"],
            )
            for event in job_events
            if event["operation"] == "checkpoint_dram_to_peer_dram_chunk"
            and event["details"]["chunk_index"]
            == event["details"]["chunk_count"]
        }
        assert completed_peer_dram == expected_checkpoints
        assert not any(
            event["operation"]
            == "checkpoint_peer_dram_to_peer_ssd_chunk"
            for event in job_events
        )

    peer_operations = {
        "checkpoint_dram_to_peer_dram_chunk",
        "checkpoint_peer_dram_to_peer_ssd_chunk",
    }
    for job_id in SSD_TIERED_JOBS | {PAIRED_TIERED_JOB}:
        peer_chunks = [
            event
            for event in grouped[job_id]
            if event["operation"] in peer_operations
        ]
        assert peer_chunks
        for event in peer_chunks:
            owner = event["details"]["checkpoint_owner_rank"]
            assert event["details"]["checkpoint_partner_rank"] == owner ^ 1

        for operation in peer_operations:
            completed = {
                (
                    event["details"]["checkpoint_owner_rank"],
                    event["iteration"],
                )
                for event in peer_chunks
                if event["operation"] == operation
                and event["details"]["chunk_index"]
                == event["details"]["chunk_count"]
            }
            assert completed == expected_checkpoints


def test_checkpoint_cpu_and_peer_bandwidth_slow_gpu_work(
    simulation_log: list[dict],
) -> None:
    grouped = events_by_job(simulation_log)
    baseline_jobs = {
        "data-cpu-tiered-30b-no-fail",
        "data-ssd-tiered-30b-no-fail",
    }

    for job_id in baseline_jobs:
        job_events = grouped[job_id]
        compute_events = [
            event
            for event in job_events
            if event["category"] == "Compute"
        ]
        slowed_compute = [
            event
            for event in compute_events
            if event["details"]["checkpoint_upload_contention_seconds"] > 0.0
        ]
        assert slowed_compute
        assert all(
            event["details"]["effective_slowdown"] > 1.0
            for event in slowed_compute
        )

        peer_chunks = [
            event
            for event in job_events
            if event["operation"] == "checkpoint_dram_to_peer_dram_chunk"
        ]
        assert peer_chunks
        for chunk in peer_chunks:
            assert chunk["details"]["peer_pair_checkpoint_gb"] == 136.0

        assert load_config(CONFIG_PATH).cluster.cpu_cores_per_node == 1
        for pair in ({0, 1}, {2, 3}):
            pair_chunks = sorted(
                (
                    event
                    for event in peer_chunks
                    if event["details"]["checkpoint_owner_rank"] in pair
                ),
                key=lambda event: (event["start"], event["end"]),
            )
            assert {
                event["details"]["checkpoint_owner_rank"]
                for event in pair_chunks
            } == pair
            for previous, current in zip(pair_chunks, pair_chunks[1:]):
                assert current["start"] >= previous["end"] - 1e-9
            assert any(
                previous["details"]["checkpoint_owner_rank"]
                != current["details"]["checkpoint_owner_rank"]
                for previous, current in zip(
                    pair_chunks,
                    pair_chunks[1:],
                )
            )

        def rank_has_slowed_overlap(rank: int, chunk: dict) -> bool:
            return any(
                event["rank"] == rank
                and event["start"] < chunk["end"]
                and chunk["start"] < event["end"]
                and event["details"][
                    "checkpoint_upload_contention_seconds"
                ]
                > 0.0
                for event in compute_events
            )

        assert any(
            rank_has_slowed_overlap(
                chunk["details"]["checkpoint_owner_rank"],
                chunk,
            )
            for chunk in peer_chunks
        )

        bandwidth_slowed_chunks = [
            event
            for event in peer_chunks
            if event["details"]["bandwidth_contention_seconds"] > 0.0
        ]
        bandwidth_slowed_collectives = [
            event
            for event in job_events
            if event["operation"] == "data_parallel_all_reduce"
            and event["details"]["bandwidth_contention_seconds"] > 0.0
        ]
        if job_id == "data-ssd-tiered-30b-no-fail":
            assert bandwidth_slowed_chunks
            assert bandwidth_slowed_collectives
            assert all(
                event["details"]["effective_slowdown"] > 1.0
                for event in (
                    bandwidth_slowed_chunks
                    + bandwidth_slowed_collectives
                )
            )

            peer_ssd_chunks = [
                event
                for event in job_events
                if event["operation"]
                == "checkpoint_peer_dram_to_peer_ssd_chunk"
            ]
            assert peer_ssd_chunks
            assert all(
                event["rank"]
                == event["details"]["checkpoint_partner_rank"]
                and event["resources"] == ["CPU"]
                for event in peer_ssd_chunks
            )
            assert any(
                rank_has_slowed_overlap(event["rank"], event)
                for event in peer_ssd_chunks
            )

            peer_sends_by_checkpoint: dict[
                tuple[int, int],
                list[dict],
            ] = defaultdict(list)
            peer_ssd_by_checkpoint: dict[
                tuple[int, int],
                list[dict],
            ] = defaultdict(list)
            for event in peer_chunks:
                peer_sends_by_checkpoint[
                    (
                        event["details"]["checkpoint_owner_rank"],
                        event["iteration"],
                    )
                ].append(event)
            for event in peer_ssd_chunks:
                peer_ssd_by_checkpoint[
                    (
                        event["details"]["checkpoint_owner_rank"],
                        event["iteration"],
                    )
                ].append(event)
            assert peer_sends_by_checkpoint.keys() == (
                peer_ssd_by_checkpoint.keys()
            )
            for key, ssd_chunks in peer_ssd_by_checkpoint.items():
                assert min(event["start"] for event in ssd_chunks) >= max(
                    event["end"]
                    for event in peer_sends_by_checkpoint[key]
                )


def test_failure_types_flush_only_the_expected_tiers(
    simulation_log: list[dict],
) -> None:
    failures = [
        event
        for event in simulation_log
        if event["category"] == "Failure"
    ]
    for failure in failures:
        details = failure["details"]
        if failure["failure_type"] == "process":
            assert details["dram_flushed_gb"] == 0.0
            assert details["ssd_flushed_gb"] == 0.0
        elif failure["failure_type"] == "node":
            assert details["ssd_flushed_gb"] == 0.0

    tiered_spot_failures = [
        failure
        for failure in failures
        if failure["job_id"]
        in {
            "data-cpu-tiered-30b",
            "data-ssd-tiered-30b",
            PAIRED_TIERED_JOB,
        }
        and failure["failure_type"] == "spot"
    ]
    assert tiered_spot_failures
    assert any(
        failure["details"]["ssd_flushed_gb"] > 0.0
        for failure in tiered_spot_failures
    )
    assert all(
        failure["details"]["ssd_flushed_gb"] == 0.0
        for failure in tiered_spot_failures
        if failure["job_id"] == "data-cpu-tiered-30b"
    )


def test_tiered_recovery_uses_latest_available_copy() -> None:
    env = simpy.Environment()
    logger = EventLogger()
    workers = {
        rank: JobWorker(
            job_id="pair-test",
            rank=rank,
            data_parallel_rank=0,
            pipeline_stage=rank,
            physical_node=f"node-{rank}",
            gpu=simpy.Resource(env, capacity=1),
            cpu=simpy.PriorityResource(env, capacity=1),
        )
        for rank in range(2)
    }
    config = load_config(CONFIG_PATH)
    strategy = TieredCheckpointStrategy(
        run_id=0,
        cluster=config.cluster,
        logger=logger,
        object_store=simpy.Resource(env, capacity=1),
        workers=workers,
        paired=True,
        persist_to_ssd=True,
        backend=SimPyBackend(env),
    )
    workers[0].current_iteration = 6
    strategy._put_copy(
        owner_rank=0,
        location_rank=0,
        tier="ssd",
        iteration=2,
        size_gb=10.0,
    )
    strategy._put_copy(
        owner_rank=0,
        location_rank=0,
        tier="dram",
        iteration=4,
        size_gb=10.0,
    )
    strategy._put_copy(
        owner_rank=0,
        location_rank=1,
        tier="ssd",
        iteration=6,
        size_gb=10.0,
    )

    latest = strategy._latest_available_copy(workers[0])
    assert latest is not None
    assert (latest.iteration, latest.location_rank, latest.tier) == (
        6,
        1,
        "ssd",
    )

    workers[1].failed = True
    latest = strategy._latest_available_copy(workers[0])
    assert latest is not None
    assert (latest.iteration, latest.location_rank, latest.tier) == (
        4,
        0,
        "dram",
    )
    workers[1].failed = False

    node_flush = strategy.handle_failure(workers[0], "node")
    assert node_flush == {"dram_flushed_gb": 10.0, "ssd_flushed_gb": 0.0}
    latest = strategy._latest_available_copy(workers[0])
    assert latest is not None and latest.iteration == 6

    strategy.handle_failure(workers[1], "spot")
    latest = strategy._latest_available_copy(workers[0])
    assert latest is not None
    assert (latest.iteration, latest.tier) == (2, "ssd")


def test_independent_failures_and_highest_severity(
    simulation_log: list[dict],
) -> None:
    failures = [
        event
        for event in simulation_log
        if event["category"] == "Failure"
    ]
    assert failures
    assert all(float(event["start"]).is_integer() for event in failures)
    assert {event["job_id"] for event in failures} == FAILURE_JOBS
    assert not any(
        event["category"] == "Failure"
        and event["job_id"].endswith("-no-fail")
        for event in simulation_log
    )

    failures_per_tick = Counter(event["start"] for event in failures)
    assert any(count > 1 for count in failures_per_tick.values())

    coalesced = [
        event
        for event in failures
        if event["details"]["coalesced_failure_signals"] > 1
    ]
    assert coalesced
    for event in coalesced:
        observed = event["details"]["observed_failure_types"]
        expected = max(observed, key=FAILURE_SEVERITY.__getitem__)
        assert event["details"]["highest_severity"] == expected

    assert any(
        event["details"]["dram_flushed_gb"] > 0.0
        for event in failures
    )


def test_interrupted_work_recovers_and_restarts(
    simulation_log: list[dict],
) -> None:
    interrupted_events = [
        event
        for event in simulation_log
        if event["category"] == "Compute"
        and event["details"].get("status") == "interrupted"
    ]
    assert interrupted_events

    for interrupted in interrupted_events:
        details = interrupted["details"]
        assert details["completed_fraction"] < 1.0

        failures = [
            event
            for event in simulation_log
            if event["category"] == "Failure"
            and event["job_id"] == interrupted["job_id"]
            and event["rank"] == interrupted["rank"]
            and event["start"] <= interrupted["end"] <= event["end"]
        ]
        assert failures

        completed_retries = [
            event
            for event in simulation_log
            if event["job_id"] == interrupted["job_id"]
            and event["rank"] == interrupted["rank"]
            and event["iteration"] == interrupted["iteration"]
            and event["operation"] == interrupted["operation"]
            and event["details"].get("microbatch")
            == details.get("microbatch")
            and event["details"].get("status") == "completed"
            and event["start"] >= interrupted["end"]
        ]
        assert completed_retries
        retry = min(completed_retries, key=lambda event: event["start"])
        assert retry["details"]["attempt"] > details["attempt"]
        assert retry["details"]["completed_fraction"] == 1.0

        restores = [
            event
            for event in simulation_log
            if event["operation"] == "dram_to_gpu_restore"
            and event["job_id"] == interrupted["job_id"]
            and event["rank"] == interrupted["rank"]
            and interrupted["end"] <= event["start"]
            and event["end"] <= retry["start"]
        ]
        assert restores
        restore = max(restores, key=lambda event: event["end"])
        preceding_failures = [
            event
            for event in simulation_log
            if event["category"] == "Failure"
            and event["job_id"] == interrupted["job_id"]
            and event["rank"] == interrupted["rank"]
            and event["end"] >= interrupted["end"]
            and event["end"] <= restore["start"]
        ]
        assert preceding_failures
        restart_end = max(event["end"] for event in preceding_failures)

        recovery_loads = [
            event
            for event in simulation_log
            if event["category"] == "Recovery"
            and event["operation"] != "dram_to_gpu_restore"
            and event["job_id"] == interrupted["job_id"]
            and event["rank"] == interrupted["rank"]
            and restart_end <= event["start"]
            and event["end"] <= restore["start"]
        ]
        source_tier = restore["details"].get("checkpoint_source_tier")
        if interrupted["job_id"] in OBJECT_STORE_JOBS:
            assert recovery_loads
            assert recovery_loads[-1]["operation"] == "object_store_to_dram"
        elif (
            source_tier == "dram"
            and restore["details"].get("checkpoint_source_rank")
            == interrupted["rank"]
        ):
            assert not recovery_loads
        else:
            assert recovery_loads

        if recovery_loads:
            assert restart_end <= min(
                event["start"] for event in recovery_loads
            )
            assert max(
                event["end"] for event in recovery_loads
            ) <= restore["start"]
        else:
            assert restart_end <= restore["start"]
        assert restore["end"] <= retry["start"]
        assert restore["iteration"] <= interrupted["iteration"]


def test_pipeline_dependency_chain(simulation_log: list[dict]) -> None:
    pipeline_events = [
        event
        for event in simulation_log
        if event["job_id"] in PIPELINE_JOBS | HYBRID_JOBS
    ]
    forward = {
        (
            event["job_id"],
            event["iteration"],
            event["data_parallel_rank"],
            event["pipeline_stage"],
            event["details"]["microbatch"],
        ): event
        for event in pipeline_events
        if event["operation"] == "pipeline_forward"
        and event["details"].get("status") == "completed"
    }
    backward = {
        (
            event["job_id"],
            event["iteration"],
            event["data_parallel_rank"],
            event["pipeline_stage"],
            event["details"]["microbatch"],
        ): event
        for event in pipeline_events
        if event["operation"] == "pipeline_backward"
        and event["details"].get("status") == "completed"
    }

    activation_transfers = [
        event
        for event in pipeline_events
        if event["operation"] == "pipeline_activation_send"
    ]
    assert activation_transfers
    for transfer in activation_transfers:
        key = (
            transfer["job_id"],
            transfer["iteration"],
            transfer["data_parallel_rank"],
            transfer["pipeline_stage"],
            transfer["details"]["microbatch"],
        )
        destination_key = (*key[:3], key[3] + 1, key[4])
        launches = [
            event
            for event in pipeline_events
            if event["operation"] == "pipeline_transfer_launch"
            and event["iteration"] == transfer["iteration"]
            and event["details"]["microbatch"]
            == transfer["details"]["microbatch"]
            and event["source"] == transfer["source"]
            and event["destination"] == transfer["destination"]
            and event["details"]["transfer_operation"]
            == transfer["operation"]
        ]
        receives = [
            event
            for event in pipeline_events
            if event["operation"] == "pipeline_activation_receive"
            and event["iteration"] == transfer["iteration"]
            and event["details"]["microbatch"]
            == transfer["details"]["microbatch"]
            and event["source"] == transfer["source"]
            and event["destination"] == transfer["destination"]
        ]
        assert len(launches) == 2
        assert len(receives) == 1
        assert all(event["resources"] == ["CPU"] for event in launches)
        assert set(transfer["resources"]) == {"GPU", "NETWORK"}
        assert receives[0]["resources"] == ["GPU"]
        assert forward[key]["end"] <= min(
            event["start"] for event in launches
        )
        assert max(event["end"] for event in launches) <= transfer["start"]
        assert receives[0]["start"] == transfer["start"]
        assert receives[0]["end"] == transfer["end"]
        assert transfer["end"] <= forward[destination_key]["start"]

    gradient_transfers = [
        event
        for event in pipeline_events
        if event["operation"] == "pipeline_gradient_send"
    ]
    assert gradient_transfers
    for transfer in gradient_transfers:
        key = (
            transfer["job_id"],
            transfer["iteration"],
            transfer["data_parallel_rank"],
            transfer["pipeline_stage"],
            transfer["details"]["microbatch"],
        )
        destination_key = (*key[:3], key[3] - 1, key[4])
        launches = [
            event
            for event in pipeline_events
            if event["operation"] == "pipeline_transfer_launch"
            and event["iteration"] == transfer["iteration"]
            and event["details"]["microbatch"]
            == transfer["details"]["microbatch"]
            and event["source"] == transfer["source"]
            and event["destination"] == transfer["destination"]
            and event["details"]["transfer_operation"]
            == transfer["operation"]
        ]
        receives = [
            event
            for event in pipeline_events
            if event["operation"] == "pipeline_gradient_receive"
            and event["iteration"] == transfer["iteration"]
            and event["details"]["microbatch"]
            == transfer["details"]["microbatch"]
            and event["source"] == transfer["source"]
            and event["destination"] == transfer["destination"]
        ]
        assert len(launches) == 2
        assert len(receives) == 1
        assert all(event["resources"] == ["CPU"] for event in launches)
        assert set(transfer["resources"]) == {"GPU", "NETWORK"}
        assert receives[0]["resources"] == ["GPU"]
        assert backward[key]["end"] <= min(
            event["start"] for event in launches
        )
        assert max(event["end"] for event in launches) <= transfer["start"]
        assert receives[0]["start"] == transfer["start"]
        assert receives[0]["end"] == transfer["end"]
        assert transfer["end"] <= backward[destination_key]["start"]


def test_collective_optimizer_checkpoint_chain(
    simulation_log: list[dict],
) -> None:
    grouped = events_by_job(simulation_log)

    for job_id in DETAILED_DATA_JOBS | HYBRID_JOBS:
        job_events = grouped[job_id]
        collectives = [
            event
            for event in job_events
            if event["operation"] == "data_parallel_all_reduce"
        ]
        assert collectives

        collective_groups: dict[tuple[int, int], list[dict]] = defaultdict(
            list
        )
        for event in collectives:
            collective_groups[
                (event["iteration"], event["pipeline_stage"])
            ].append(event)

        for (iteration, stage), group in collective_groups.items():
            assert len({event["start"] for event in group}) == 1
            assert len({event["end"] for event in group}) == 1
            collective_start = group[0]["start"]
            collective_end = group[0]["end"]
            participant_ranks = set(
                group[0]["details"]["collective_group"]
            )
            assert {event["rank"] for event in group} == participant_ranks

            backward_operations = {
                "data_parallel_backward",
                "pipeline_backward",
            }
            participant_backward = [
                event
                for event in job_events
                if event["iteration"] == iteration
                and event["pipeline_stage"] == stage
                and event["rank"] in participant_ranks
                and event["operation"] in backward_operations
                and event["details"].get("status") == "completed"
            ]
            assert participant_backward
            assert max(
                event["end"] for event in participant_backward
            ) <= collective_start

            optimizers = [
                event
                for event in job_events
                if event["iteration"] == iteration
                and event["pipeline_stage"] == stage
                and event["rank"] in participant_ranks
                and event["operation"]
                in {
                    "data_parallel_optimizer",
                    "pipeline_optimizer",
                }
                and event["details"].get("status") == "completed"
            ]
            assert len(optimizers) == len(participant_ranks)
            assert min(event["start"] for event in optimizers) >= collective_end

            participant_failures = [
                event
                for event in job_events
                if event["category"] == "Failure"
                and event["rank"] in participant_ranks
            ]
            for failure in participant_failures:
                overlaps = (
                    failure["start"] < collective_end
                    and collective_start < failure["end"]
                )
                assert not overlaps

    optimizer_operations = {
        "data_parallel_optimizer",
        "pipeline_optimizer",
    }
    for job_events in grouped.values():
        checkpoints = [
            event
            for event in job_events
            if event["operation"] == "checkpoint_stage_gpu_to_dram"
        ]
        for checkpoint in checkpoints:
            optimizer_end = max(
                event["end"]
                for event in job_events
                if event["rank"] == checkpoint["rank"]
                and event["iteration"] == checkpoint["iteration"]
                and event["operation"] in optimizer_operations
            )
            assert optimizer_end <= checkpoint["start"]

        uploads = [
            event
            for event in job_events
            if event["operation"]
            == "checkpoint_stage_dram_to_object_store"
        ]
        for upload in uploads:
            matching_stages = [
                event
                for event in checkpoints
                if event["rank"] == upload["rank"]
                and event["details"]["checkpoint_group"]
                == upload["details"]["checkpoint_group"]
            ]
            assert matching_stages
            assert max(event["end"] for event in matching_stages) <= upload[
                "start"
            ]


def test_all_iterations_complete_once_per_rank(
    simulation_log: list[dict],
) -> None:
    optimizer_operations = {
        "data_parallel_optimizer",
        "pipeline_optimizer",
    }
    grouped = events_by_job(simulation_log)
    for job_id, job_events in grouped.items():
        if job_id in AGGREGATE_JOBS:
            optimizers = [
                event
                for event in job_events
                if event["operation"] == "data_parallel_optimizer"
            ]
            assert len(optimizers) == 20 * 100
            by_iteration: dict[int, list[dict]] = defaultdict(list)
            for event in optimizers:
                by_iteration[event["iteration"]].append(event)
            assert set(by_iteration) == set(range(1, 21))
            for iteration_events in by_iteration.values():
                assert len(iteration_events) == 100
                assert sum(
                    event["details"]["represented_ranks"]
                    for event in iteration_events
                ) == 1_000_000
            continue
        ranks = {event["rank"] for event in job_events}
        for rank in ranks:
            optimizers = [
                event
                for event in job_events
                if event["rank"] == rank
                and event["operation"] in optimizer_operations
                and event["details"].get("status") == "completed"
            ]
            assert len(optimizers) == 20
            assert {event["iteration"] for event in optimizers} == set(
                range(1, 21)
            )


def test_completed_dependency_operations_avoid_failed_participants(
    simulation_log: list[dict],
) -> None:
    failures_by_job: dict[str, list[dict]] = defaultdict(list)
    for event in simulation_log:
        if event["category"] == "Failure":
            failures_by_job[event["job_id"]].append(event)

    checked_operations = {
        "pipeline_transfer_launch",
        "pipeline_activation_send",
        "pipeline_activation_receive",
        "pipeline_gradient_send",
        "pipeline_gradient_receive",
        "data_parallel_all_reduce",
        "checkpoint_stage_gpu_to_dram",
        "checkpoint_stage_dram_to_object_store",
        "checkpoint_dram_to_local_ssd_chunk",
        "checkpoint_dram_to_peer_dram_chunk",
        "checkpoint_peer_dram_to_peer_ssd_chunk",
    }
    for event in simulation_log:
        if event["job_id"] in AGGREGATE_JOBS:
            continue
        if event["operation"] not in checked_operations:
            continue

        participant_ranks = {event["rank"]}
        if event["operation"] == "data_parallel_all_reduce":
            participant_ranks.update(
                event["details"]["collective_group"]
            )
        elif event["operation"] in {
            "pipeline_transfer_launch",
            "pipeline_activation_send",
            "pipeline_activation_receive",
            "pipeline_gradient_send",
            "pipeline_gradient_receive",
        }:
            destination = str(event["destination"])
            participant_ranks.add(int(destination.rsplit("-", 1)[1]))
        elif event["operation"] in {
            "checkpoint_dram_to_peer_dram_chunk",
            "checkpoint_peer_dram_to_peer_ssd_chunk",
        }:
            participant_ranks.add(
                event["details"]["checkpoint_owner_rank"]
            )
            participant_ranks.add(
                event["details"]["checkpoint_partner_rank"]
            )

        for failure in failures_by_job[event["job_id"]]:
            if failure["rank"] not in participant_ranks:
                continue
            overlaps = (
                failure["start"] < event["end"]
                and event["start"] < failure["end"]
            )
            assert not overlaps


def test_million_rank_job_uses_cohort_and_expands_only_deviations(
    simulation_log: list[dict],
) -> None:
    job_events = [
        event
        for event in simulation_log
        if event["job_id"] == "data-ssd-peer-1m"
    ]
    assert len(job_events) < 15_000

    cohort_events = [
        event
        for event in job_events
        if event["details"].get("aggregate")
    ]
    assert cohort_events
    cohort_ranges = {
        (
            event["details"]["aggregate_rank_start"],
            event["details"]["aggregate_rank_end"],
        )
        for event in cohort_events
    }
    assert len(cohort_ranges) == 100
    assert sum(
        end - start + 1 for start, end in cohort_ranges
    ) == 1_000_000
    assert max(
        event["details"]["represented_ranks"] for event in cohort_events
    ) <= 10_000
    assert {
        event["details"]["aggregate_partitions"]
        for event in cohort_events
    } == {100}
    assert {
        event["details"]["scheduler_batch"]
        for event in cohort_events
    } == set(
        range(
            math.ceil(
                100
                / next(
                    event["details"]["scheduler_worker_cores"]
                    for event in cohort_events
                )
            )
        )
    )

    failures = [
        event for event in job_events if event["category"] == "Failure"
    ]
    assert 1 <= len(failures) <= 128
    failed_ranks = {event["rank"] for event in failures}
    assert all(event["node"] == f"rank-{event['rank']}" for event in failures)

    figure = make_resource_timeline(
        job_events,
        build_operation_styles(job_events),
    )
    lane_values = {
        str(lane)
        for trace in figure.data
        for lane in (getattr(trace, "y", None) or [])
    }
    assert any("remaining workers" in lane for lane in lane_values)
    for rank in failed_ranks:
        assert any(f"rank {rank:,}" in lane for lane in lane_values)


def test_visualization_contains_one_tab_per_job(
    simulation_log: list[dict],
    tmp_path: Path,
) -> None:
    output = tmp_path / "dashboard.html"
    sampled_events = [
        event
        for job_events in events_by_job(simulation_log).values()
        for event in job_events[:80]
    ]
    write_dashboard(
        sampled_events,
        output_path=output,
        source_log=Path("events.jsonl"),
    )
    document = output.read_text(encoding="utf-8")
    assert document.count('<button class="job-tab') == len(ALL_JOBS) + 1
    assert document.count('class="job-view') == len(ALL_JOBS) + 1
    assert ">All jobs</button>" in document
    for job_id in ALL_JOBS:
        assert f">{job_id}</button>" in document

    combined = make_resource_timeline(
        sampled_events,
        build_operation_styles(sampled_events),
    )
    combined_lanes = {
        str(lane)
        for trace in combined.data
        for lane in (getattr(trace, "y", None) or [])
    }
    assert any(
        lane.startswith("data-parallel-13b / rank 0")
        for lane in combined_lanes
    )
    assert any(
        lane.startswith("pipeline-parallel-70b / rank 0")
        for lane in combined_lanes
    )


def test_failures_use_compact_markers_not_resource_bars(
    simulation_log: list[dict],
) -> None:
    pipeline_events = [
        event
        for event in simulation_log
        if event["job_id"] == "pipeline-parallel-70b"
    ]
    failures = [
        event
        for event in pipeline_events
        if event["category"] == "Failure"
    ]
    assert failures

    figure = make_resource_timeline(
        pipeline_events,
        build_operation_styles(pipeline_events),
    )
    bar_operations = {
        str(trace.legendgroup)
        for trace in figure.data
        if trace.type == "bar"
    }
    assert not {
        "process_failure_restart",
        "node_failure_restart",
        "spot_failure_restart",
    } & bar_operations

    marker_traces = [
        trace
        for trace in figure.data
        if trace.type == "scatter" and trace.name == "Failure start"
    ]
    assert len(marker_traces) == 1
    assert len(marker_traces[0].x) == len(failures)


def test_resource_intervals_respect_capacity(
    simulation_log: list[dict],
) -> None:
    for resource in {"CPU", "GPU"}:
        intervals: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for event in simulation_log:
            if resource not in event["resources"]:
                continue
            if event["category"] in {"Failure", "Synchronization"}:
                continue
            intervals[(event["job_id"], event["rank"])].append(event)

        for resource_events in intervals.values():
            ordered = sorted(
                resource_events,
                key=lambda event: (event["start"], event["end"]),
            )
            for previous, current in zip(ordered, ordered[1:]):
                assert current["start"] >= previous["end"] - 1e-9

    waits = [
        event
        for event in simulation_log
        if event["category"] == "Synchronization"
    ]
    gpu_work = [
        event
        for event in simulation_log
        if "GPU" in event["resources"]
        and event["category"] not in {"Failure", "Synchronization"}
    ]
    for wait in waits:
        for work in gpu_work:
            if (
                wait["job_id"],
                wait["rank"],
            ) != (
                work["job_id"],
                work["rank"],
            ):
                continue
            overlap = min(wait["end"], work["end"]) - max(
                wait["start"],
                work["start"],
            )
            assert overlap <= 1e-9


def test_object_store_concurrency_limit(
    simulation_log: list[dict],
) -> None:
    uploads = [
        event
        for event in simulation_log
        if event["operation"]
        == "checkpoint_stage_dram_to_object_store"
    ]
    boundaries = [
        boundary
        for upload in uploads
        for boundary in (
            (upload["start"], 1),
            (upload["end"], -1),
        )
    ]
    active = 0
    peak = 0
    for _, delta in sorted(boundaries, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    assert peak == 4
