from __future__ import annotations

import json
from pathlib import Path

import pytest

from visualise_policy_matrix import (
    generate_policy_matrix,
    load_result_rows,
    summarise_policies,
)


def write_result(path: Path, scenario: str, rows: list[dict]) -> None:
    path.write_text(
        json.dumps({"scenario": scenario, "rows": rows, "aggregate": {}}),
        encoding="utf-8",
    )


def result_row(
    arm: str,
    seed: int,
    completion: float,
    *,
    makespan: float | None = None,
    **metrics: object,
) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "train_end_s": completion,
        "makespan_s": completion if makespan is None else makespan,
        "censored_jobs": [],
        "failures": {},
        "restores": 0,
        "recovery_source_tiers": {},
        "durable_frac": None,
        "rank_flushes": 0,
        **metrics,
    }


def test_generate_matrix_merges_files_computes_metrics_and_links_dashboards(
    tmp_path: Path,
) -> None:
    named = tmp_path / "named_results.json"
    ours = tmp_path / "ours_results.json"
    write_result(
        named,
        "scenarios\\fleet.yaml",
        [
            result_row(
                "jit",
                7,
                125,
                makespan=130,
                censored_jobs=["frontier3"],
                failures={"node": 1, "process": 2},
                restores=3,
                recovery_source_tiers={"store": 2, "initial_state": 1},
                durable_frac=0.75,
                rank_flushes=10,
                aborted_flushes=2,
                fallback_flushes=3,
                partial_grants=4,
                lost_iters_max=9,
            ),
            result_row("ideal", 7, 100, makespan=110, policy_kind="ideal"),
        ],
    )
    write_result(
        ours,
        "C:/work/scenarios/fleet.yaml",
        [
            result_row(
                "ours_full",
                7,
                110,
                failures={"preemption": 4},
                restores=4,
                recovery_source_tiers={"crossjob_peer": 4},
                durable_frac=0.9,
                rank_flushes=20,
            )
        ],
    )
    dashboards = tmp_path / "dashboards"
    dashboards.mkdir()
    (dashboards / "jit_aggregate.html").write_text("jit", encoding="utf-8")
    (dashboards / "fleet_ours_full_s7_aggregate.html").write_text(
        "ours", encoding="utf-8"
    )
    output = tmp_path / "policy_matrix.html"

    summaries = generate_policy_matrix(
        [named, ours], output, dashboard_dir=dashboards
    )

    by_arm = {summary.arm: summary for summary in summaries}
    assert by_arm["jit"].overhead_pct == pytest.approx(25)
    assert by_arm["jit"].goodput_pct == pytest.approx(80)
    assert by_arm["jit"].failure_count == 3
    assert by_arm["jit"].durable_pct == pytest.approx(75)
    assert by_arm["jit"].initial_state_recoveries == pytest.approx(1)
    assert by_arm["jit"].aborted_flushes == pytest.approx(2)
    assert by_arm["jit"].fallback_flushes == pytest.approx(3)
    assert by_arm["jit"].partial_grants == pytest.approx(4)
    assert by_arm["jit"].lost_iters_max == pytest.approx(9)
    assert by_arm["ours_full"].overhead_pct == pytest.approx(10)
    document = output.read_text(encoding="utf-8")
    assert "Checkpoint policy matrix · fleet.yaml" in document
    assert "+25.0%" in document
    assert "80.0%" in document
    assert "initial_state 1" in document
    assert "Initial-state recoveries" in document
    assert "3 local fallback; 4 partial" in document
    assert 'href="dashboards/jit_aggregate.html"' in document
    assert 'href="dashboards/fleet_ours_full_s7_aggregate.html"' in document
    # Plotly is embedded, not loaded from a CDN.
    assert "plotly.js" in document
    assert 'src="https://cdn.plot.ly/' not in document


def test_summaries_pair_each_policy_run_with_same_seed_ideal() -> None:
    rows = [
        result_row("ideal", 7, 100, policy_kind="ideal"),
        result_row("ideal", 11, 200, policy_kind="ideal"),
        result_row(
            "jit",
            7,
            125,
            failures={"node": 2},
            restores=2,
            rank_flushes=10,
        ),
        result_row(
            "jit",
            11,
            220,
            failures={"node": 4},
            restores=4,
            rank_flushes=30,
        ),
    ]

    jit = next(summary for summary in summarise_policies(rows) if summary.arm == "jit")

    assert jit.completion_s == pytest.approx(172.5)
    assert jit.overhead_pct == pytest.approx(17.5)
    assert jit.goodput_pct == pytest.approx(85.454545)
    assert jit.failures == {"node": 3}
    assert jit.restores == 3
    assert jit.flushes == 20


def test_missing_recovery_metadata_is_not_reported_as_zero() -> None:
    missing = result_row("legacy", 7, 120)
    missing.pop("recovery_source_tiers")
    explicit_zero = result_row("current", 7, 110)

    summaries = {row.arm: row for row in summarise_policies([missing, explicit_zero])}

    assert summaries["legacy"].initial_state_recoveries is None
    assert summaries["current"].initial_state_recoveries == 0


def test_load_result_rows_rejects_different_scenarios(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_result(first, "scenarios/one.yaml", [result_row("ideal", 7, 100)])
    write_result(second, "scenarios/two.yaml", [result_row("jit", 7, 120)])

    with pytest.raises(ValueError, match="different scenarios"):
        load_result_rows([first, second])
