from __future__ import annotations

from pathlib import Path

import pytest
import yaml


SCENARIO_DIR = Path(__file__).parents[1] / "scenarios" / "adversarial_5k"
SCENARIOS = (
    "failure_storm.yaml",
    "donor_scarcity.yaml",
    "bandwidth_storm.yaml",
)
EXPECTED_ARMS = {
    "jit",
    "checkfreq",
    "gemini",
    "megascale",
    "checknrun",
    "naive1000_all",
    "naive10_100",
    "ideal",
}


@pytest.mark.parametrize("filename", SCENARIOS)
def test_adversarial_scenario_is_exactly_5000_nodes(filename: str) -> None:
    scenario = yaml.safe_load((SCENARIO_DIR / filename).read_text())

    logical_nodes = sum(
        int(spec["count"]) * int(spec["ranks"])
        for spec in scenario["classes"].values()
    )

    assert logical_nodes == 5_000
    assert scenario["cluster"]["node_count"] == 5_000
    assert set(scenario["arms"]) == EXPECTED_ARMS
    assert scenario["simulation"]["seeds"] == [7]


@pytest.mark.parametrize("filename", SCENARIOS)
def test_adversarial_jobs_are_ten_minute_stress_tests(filename: str) -> None:
    scenario = yaml.safe_load((SCENARIO_DIR / filename).read_text())

    for spec in scenario["classes"].values():
        assert spec["iterations"] * spec["iteration_seconds"] == 600
        assert 1 <= spec["cohorts"] <= spec["ranks"]
