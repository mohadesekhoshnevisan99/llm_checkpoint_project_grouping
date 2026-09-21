from __future__ import annotations

import gzip
import json
import random
from collections import Counter
from types import SimpleNamespace

import pytest
import simpy

from checkpointing.crossjob import DonorRegistry
from run_crossjob import FlushSpanAccumulator
from run_scenario import (
    BatchedScenarioEventLogger,
    _format_wall_time,
    _scenario_work_progress,
    build_arrival_manifest,
    expand_jobs,
    job_hazard_process,
    normalize_scheduled_failures,
    run_arm,
    run_until_complete_or_cap,
)
from simulation import SimPyBackend
from visualise import load_events


def test_run_stops_when_jobs_complete_even_with_live_monitor() -> None:
    env = simpy.Environment()
    backend = SimPyBackend(env)

    def job():
        yield backend.timeout(5.0)

    def monitor():
        while True:
            yield backend.timeout(1.0)

    process = backend.process(job())
    backend.process(monitor())

    completed = run_until_complete_or_cap(env, backend, [process], 1000.0)

    assert completed
    assert backend.now == pytest.approx(5.0)


def test_run_stops_at_cap_when_job_does_not_complete() -> None:
    env = simpy.Environment()
    backend = SimPyBackend(env)

    def job():
        while True:
            yield backend.timeout(2.0)

    process = backend.process(job())

    completed = run_until_complete_or_cap(env, backend, [process], 7.0)

    assert not completed
    assert backend.now == pytest.approx(7.0)


def test_progress_is_weighted_by_cohort_work() -> None:
    runtimes = [
        SimpleNamespace(
            iterations=10,
            it=5,
            workers={index: object() for index in range(4)},
            end_time=None,
        ),
        SimpleNamespace(
            iterations=20,
            it=20,
            workers={0: object()},
            end_time=12.0,
        ),
    ]

    fraction, total_jobs, unfinished = _scenario_work_progress(runtimes)

    assert fraction == pytest.approx(40 / 60)
    assert total_jobs == 2
    assert unfinished == 1
    assert _format_wall_time(65) == "01:05"
    assert _format_wall_time(3661) == "1:01:01"


def flush_event(job: str, group: str, start: float, end: float):
    return SimpleNamespace(
        job_id=job,
        start=start,
        end=end,
        details={
            "checkpoint_strategy": "crossjob_peer",
            "checkpoint_group": group,
            "path": "peer",
        },
    )


def test_flush_span_accumulator_updates_without_rescanning_history() -> None:
    events = [
        flush_event("job-a", "g1", 0.0, 2.0),
        flush_event("job-a", "g1", 1.0, 5.0),
        flush_event("job-a", "g2", 10.0, 12.0),
        SimpleNamespace(job_id="ignored", start=0.0, end=100.0, details={}),
    ]
    accumulator = FlushSpanAccumulator()

    assert accumulator.update(events) == {"job-a": pytest.approx(3.5)}
    assert accumulator.scan_index == 4

    events.extend(
        [
            flush_event("job-a", "g2", 11.0, 15.0),
            flush_event("job-b", "g1", 20.0, 22.0),
        ]
    )

    assert accumulator.update(events) == {
        "job-a": pytest.approx(5.0),
        "job-b": pytest.approx(2.0),
    }
    assert accumulator.scan_index == 6


def test_batched_logger_spools_sorted_gzip_without_retaining_history(
    tmp_path,
) -> None:
    trace = tmp_path / "events.jsonl.gz"
    logger = BatchedScenarioEventLogger(trace_path=trace, batch_size=2)
    for event_id, start in enumerate((4.0, 1.0, 3.0, 2.0), start=1):
        logger.record(
            start=start,
            end=start + 0.5,
            rank=event_id,
            node=f"node-{event_id}",
            job_id="job-a",
            category="Training",
            operation="training_iterations",
        )

    assert logger.total_event_count == 4
    assert logger.events == []
    logger.finalize_trace()

    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert [row["start"] for row in rows] == [1.0, 2.0, 3.0, 4.0]


def test_detailed_visualizer_loads_gzip_trace(tmp_path) -> None:
    trace = tmp_path / "events.jsonl.gz"
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"start": 1.0, "operation": "job_pending"}))
        handle.write("\n")

    assert load_events(trace) == [{"start": 1.0, "operation": "job_pending"}]


def _tiny_policy_scenario(arm: str, flags: dict) -> dict:
    return {
        "simulation": {"seeds": [7], "max_sim_s": 100.0, "engine": "event"},
        "cluster": {
            "node_count": 8,
            "gpu_cpu_bandwidth_gbps": 40.0,
            "network_bandwidth_gbps": 50.0,
            "local_ssd_bandwidth_gbps": 6.0,
            "object_store_bandwidth_gbps": 8.0,
        },
        "classes": {
            "small": {
                "count": 1,
                "ranks": 4,
                "cohorts": 4,
                "checkpoint_gb_per_rank": 1.0,
                "iteration_seconds": 1.0,
                "iterations": 4,
                "checkpoint_every": 1,
                "kpeers": 6,
                "rpo_s": 10.0,
            }
        },
        "failures": {
            "per_node_per_second": 1.0,
            "weights": {"process": 0.4, "node": 0.2, "spot": 0.4},
            "restart_seconds": {"process": 0.1, "node": 0.1, "spot": 0.1},
        },
        "preemption": {"class_prefix": "small", "every_s": 0.5, "seed": 3},
        "store": {
            "in_gbps": 50.0,
            "out_gbps": 50.0,
            "disk_gbps": 50.0,
            "stream_gbps": 2.0,
        },
        "controller": {"slot_period_s": 10.0},
        "arms": {arm: flags},
    }


def test_legacy_arrival_manifest_matches_existing_expansion() -> None:
    scenario = _tiny_policy_scenario("configured", {"kpeers": True})
    scenario["classes"]["small"]["count"] = 2

    manifest = build_arrival_manifest(scenario, seed=7)

    assert [arrival.job_id for arrival in manifest] == ["small0", "small1"]
    assert [arrival.arrival_s for arrival in manifest] == [0.0, 0.0]
    assert [arrival.source for arrival in manifest] == ["initial", "initial"]


def test_scheduled_arrivals_allocate_stable_ids_and_batches() -> None:
    scenario = _tiny_policy_scenario("configured", {"kpeers": True})
    scenario["classes"]["small"]["count"] = 4
    scenario["job_arrivals"] = {
        "mode": "scheduled",
        "initial": {"small": 1},
        "events": [
            {"at_s": 5.0, "add": {"small": 2}},
            {"at_s": 9.0, "add": {"small": 1}},
        ],
    }

    manifest = build_arrival_manifest(scenario, seed=7)

    assert [arrival.job_id for arrival in manifest] == [
        "small0", "small1", "small2", "small3"
    ]
    assert [arrival.arrival_s for arrival in manifest] == [0.0, 5.0, 5.0, 9.0]
    assert [arrival.batch for arrival in manifest] == [0, 1, 1, 2]


def test_weighted_random_arrivals_are_seeded_and_respect_limits() -> None:
    scenario = _tiny_policy_scenario("configured", {"kpeers": True})
    small = scenario["classes"]["small"]
    small["count"] = 3
    scenario["classes"]["large"] = {**small, "count": 3, "ranks": 2}
    scenario["job_arrivals"] = {
        "mode": "weighted_random",
        "initial": {"small": 1},
        "random": {
            "start_s": 5.0,
            "end_s": 20.0,
            "total": 4,
            "seed": 41,
            "pool": {
                "small": {"weight": 1, "limit": 2},
                "large": {"weight": 5, "limit": 3},
            },
        },
    }

    first = build_arrival_manifest(scenario, seed=7)
    again = build_arrival_manifest(scenario, seed=7)
    other_seed = build_arrival_manifest(scenario, seed=11)

    assert first == again
    assert first != other_seed
    assert len(first) == 5
    assert first[0].job_id == "small0"
    assert first[0].arrival_s == 0.0
    assert all(5.0 <= arrival.arrival_s <= 20.0 for arrival in first[1:])
    counts = Counter(arrival.class_name for arrival in first[1:])
    assert counts["small"] <= 2
    assert counts["large"] <= 3


@pytest.mark.parametrize(
    ("job_arrivals", "message"),
    [
        (
            {"mode": "scheduled", "initial": {"small": 2}, "events": []},
            "exceeding",
        ),
        (
            {
                "mode": "weighted_random",
                "random": {
                    "start_s": 5,
                    "end_s": 10,
                    "total": 2,
                    "pool": {"small": {"weight": 1, "limit": 1}},
                },
            },
            "exceeds pool capacity",
        ),
        (
            {
                "mode": "weighted_random",
                "random": {
                    "start_s": 5,
                    "end_s": 10,
                    "pool": {"small": {"weight": 0, "limit": 1}},
                },
            },
            "weight must be positive",
        ),
    ],
)
def test_arrival_manifest_rejects_invalid_limits(job_arrivals, message) -> None:
    scenario = _tiny_policy_scenario("configured", {"kpeers": True})
    scenario["job_arrivals"] = job_arrivals

    with pytest.raises(ValueError, match=message):
        build_arrival_manifest(scenario, seed=7)


def test_scheduled_run_emits_lifecycle_and_starts_training_after_arrival(
    tmp_path,
) -> None:
    trace = tmp_path / "scheduled-arrivals.jsonl.gz"
    scenario = _tiny_policy_scenario("ideal", {"ideal": True, "kpeers": False})
    scenario["classes"]["small"].update(count=2, iterations=2)
    scenario["failures"]["per_node_per_second"] = 0.0
    scenario.pop("preemption")
    scenario["job_arrivals"] = {
        "mode": "scheduled",
        "initial": {"small": 1},
        "events": [{"at_s": 3.0, "add": {"small": 1}}],
    }

    row = run_arm(scenario, "ideal", 7, trace)

    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    arrivals = [event for event in events if event["operation"] == "job_arrival"]
    idle_donors = [
        event for event in events
        if event["operation"] == "job_idle_checkpoint"
    ]
    late_training = [
        event for event in events
        if event["job_id"] == "small1" and event["category"] == "Training"
    ]

    assert row["arrival_mode"] == "scheduled"
    assert row["jobs_initial"] == 1
    assert row["jobs_arrived"] == 2
    assert row["pending_at_cap"] == []
    assert row["active_at_cap"] == []
    assert [(event["job_id"], event["start"]) for event in arrivals] == [
        ("small0", 0.0), ("small1", 3.0)
    ]
    assert len(idle_donors) == 2
    assert {entry["nodes"] for entry in row["arrival_manifest"]} == {4}
    assert late_training
    assert min(event["start"] for event in late_training) >= 3.0
    assert all(slot.available for slot in DonorRegistry.for_run(0).slots.values())


def test_preemption_hazard_clock_does_not_start_before_job_arrival() -> None:
    env = simpy.Environment()
    backend = SimPyBackend(env)
    requests = []
    runtime = SimpleNamespace(
        arrival_s=5.0,
        end_time=None,
        healthy=True,
        last_failed_until=0.0,
        request_preemption=lambda: requests.append(backend.now),
    )
    backend.process(job_hazard_process(
        backend,
        runtime,
        mean_s=0.02,
        rng=random.Random(7),
        grace_s=0.0,
    ))

    env.run(until=4.99)
    assert requests == []

    env.run(until=5.2)
    assert requests
    assert min(requests) >= 5.0


def test_completed_job_nodes_serve_later_job_checkpoints(tmp_path) -> None:
    trace = tmp_path / "idle-donors.jsonl.gz"
    scenario = _tiny_policy_scenario("configured", {"kpeers": True})
    scenario["classes"]["small"].update(
        count=3, iterations=2, checkpoint_every=1, kpeers=1
    )
    scenario["failures"]["per_node_per_second"] = 0.0
    scenario.pop("preemption")
    scenario["job_arrivals"] = {
        "mode": "scheduled",
        "initial": {"small": 1},
        "events": [
            {"at_s": 3.0, "add": {"small": 1}},
            {"at_s": 10.0, "add": {"small": 1}},
        ],
    }

    run_arm(scenario, "configured", 7, trace)

    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    idle_at = next(
        event["start"] for event in events
        if event["job_id"] == "small0"
        and event["operation"] == "job_idle_checkpoint"
    )
    late_peer_flushes = [
        event for event in events
        if event["job_id"] == "small1"
        and event["operation"] == "checkpoint_dram_to_crossjob_peers_chunk"
    ]

    assert idle_at < 3.0
    assert late_peer_flushes
    assert min(event["start"] for event in late_peer_flushes) > idle_at


def test_ideal_arm_disables_checkpointing_failures_and_preemption(tmp_path) -> None:
    trace = tmp_path / "ideal.jsonl.gz"
    scenario = _tiny_policy_scenario("ideal", {"ideal": True, "kpeers": False})
    scenario["scheduled_failures"] = [{
        "id": "ignored-by-ideal",
        "job_id": "small0",
        "cohort": 1,
        "after_iteration": 2,
        "failure_type": "node",
    }]

    row = run_arm(scenario, "ideal", 7, trace)

    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    assert row["policy_kind"] == "ideal"
    assert row["rank_flushes"] == 0
    assert row["failures"] == {}
    assert len(row["scheduled_failures_planned"]) == 1
    assert row["scheduled_failures_fired"] == []
    assert row["censored_jobs"] == []
    assert not [event for event in events if event["category"] == "Checkpoint"]


def test_gemini_uses_its_paper_replication_degree_not_scenario_k(tmp_path) -> None:
    trace = tmp_path / "gemini.jsonl.gz"
    scenario = _tiny_policy_scenario("gemini", {"baseline": "gemini"})
    scenario["failures"]["per_node_per_second"] = 0.0
    scenario.pop("preemption")

    row = run_arm(scenario, "gemini", 7, trace)

    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    replicas = [
        event
        for event in events
        if event["operation"] == "checkpoint_dram_to_peer_dram_replica"
    ]
    assert row["policy_kind"] == "baseline"
    assert replicas
    assert {event["details"]["kpeers"] for event in replicas} == {2}


def test_policy_store_cadence_gates_donor_drains(tmp_path) -> None:
    trace = tmp_path / "policy-drain.jsonl.gz"
    scenario = _tiny_policy_scenario("configured", {"kpeers": True})
    scenario["classes"]["small"].update(
        count=2,
        cohorts=1,
        iterations=5,
        checkpoint_every=1,
        kpeers=1,
    )
    scenario["failures"]["per_node_per_second"] = 0.0
    scenario.pop("preemption")
    policy = {
        "name": "test-policy",
        "flags": {"kpeers": True, "slots": True},
        "jobs": {
            job_id: {
                "class": "small",
                "kpeers": 1,
                "checkpoint_every": 1,
                "capture_every": 1,
                "store_every": 3,
                "slot_s": slot_s,
            }
            for job_id, slot_s in (("small0", 0.0), ("small1", 2.0))
        },
    }

    row = run_arm(scenario, "configured", 7, trace, policy=policy)

    # Four L2 waves are attempted per job (iterations 1..4), but f3/f2=3 means
    # only the third wave from each job may drain to L3.
    assert row["l3_drains"] == 2
    assert row["drain_bytes_gb"] == pytest.approx(8.0)


def test_scheduled_failure_fires_once_after_rollback_replays_boundary(
    tmp_path,
) -> None:
    trace = tmp_path / "scheduled.jsonl.gz"
    scenario = _tiny_policy_scenario("jit", {"baseline": "jit"})
    scenario.pop("preemption")
    scenario["failures"].update(
        per_node_per_second=100.0,
        inject_random=False,
        weights={"process": 0.0, "node": 1.0, "spot": 0.0},
    )
    scenario["scheduled_failures"] = [{
        "id": "fixed-node-loss",
        "job_id": "small0",
        "cohort": 1,
        "after_iteration": 2,
        "failure_type": "node",
    }]

    row = run_arm(scenario, "jit", 7, trace)

    assert row["random_failures_injected"] is False
    assert row["failures"] == {"node": 1}
    assert row["restores"] == 1
    assert row["recovery_source_tiers"] == {"jit_peer_dram": 1}
    assert row["censored_jobs"] == []
    assert [event["event_id"] for event in row["scheduled_failures_fired"]] == [
        "fixed-node-loss"
    ]


@pytest.mark.parametrize(
    ("event", "message"),
    [
        ({"job_id": "missing", "cohort": 0, "after_iteration": 1,
          "failure_type": "node"}, "unknown job"),
        ({"job_id": "small0", "cohort": 4, "after_iteration": 1,
          "failure_type": "node"}, "outside"),
        ({"job_id": "small0", "cohort": 0, "after_iteration": 0,
          "failure_type": "node"}, "outside"),
        ({"job_id": "small0", "cohort": 0, "after_iteration": 1,
          "failure_type": "rack"}, "invalid failure_type"),
    ],
)
def test_scheduled_failure_validation_rejects_invalid_events(
    event,
    message,
) -> None:
    scenario = _tiny_policy_scenario("jit", {"baseline": "jit"})
    scenario["scheduled_failures"] = [{"id": "bad", **event}]

    with pytest.raises(ValueError, match=message):
        normalize_scheduled_failures(scenario, expand_jobs(scenario))
