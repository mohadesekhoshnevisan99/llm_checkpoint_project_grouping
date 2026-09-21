from __future__ import annotations

from pathlib import Path

import yaml

from gp_policy import _enact_placement_for_k
from run_scenario import expand_jobs, normalize_scheduled_failures


SCENARIO = (
    Path(__file__).parents[1]
    / "scenarios"
    / "sysname_targeted_pilot_500.yaml"
)


def test_targeted_pilot_is_exact_and_deterministic() -> None:
    scenario = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))

    assert sum(
        int(spec["count"]) * int(spec["ranks"])
        for spec in scenario["classes"].values()
    ) == 500
    assert scenario["cluster"]["node_count"] == 500
    assert all(
        int(spec["cohorts"]) == int(spec["ranks"])
        for spec in scenario["classes"].values()
    )
    assert scenario["failures"]["inject_random"] is False
    assert "preemption" not in scenario
    assert set(scenario["arms"]) == {"jit", "gemini", "megascale", "ideal"}


def test_targeted_pilot_failure_is_valid_and_targets_dominant_tail() -> None:
    scenario = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))

    grouped = normalize_scheduled_failures(scenario, expand_jobs(scenario))

    assert len(grouped["dominant0"]) == 1
    event = grouped["dominant0"][0]
    assert event.event_id == "dominant-tail-node-loss"
    assert event.cohort == 479
    assert event.after_iteration == 36
    assert event.failure_type == "node"
    assert all(not events for job_id, events in grouped.items()
               if job_id != "dominant0")


def test_zero_crossjob_capacity_switches_dominant_to_local_store() -> None:
    assert _enact_placement_for_k("crossjob", 0) == "local"
    assert _enact_placement_for_k("crossjob", 1) == "crossjob"
    assert _enact_placement_for_k("local", 0) == "local"
