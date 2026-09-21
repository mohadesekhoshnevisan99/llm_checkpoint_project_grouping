from __future__ import annotations

import gzip
import json
from pathlib import Path

from visualise_aggregate import (
    aggregate_activity,
    allocated_nodes_figure,
    category_group,
    checkpoint_policy_figure,
    configured_checkpoint_policy,
    generate_dashboard,
    iter_events,
    load_run_context,
    local_ssd_write_figure,
    route_figure,
    service_class,
    summarize_trace,
    topology_figure,
)


def test_lifecycle_events_keep_their_own_category() -> None:
    assert category_group({
        "category": "Lifecycle",
        "operation": "job_pending",
    }) == "Lifecycle"


def event(
    event_id: int,
    *,
    job_id: str,
    node: str,
    start: float,
    end: float,
    category: str,
    operation: str,
    represents: int,
    source: str | None = None,
    destination: str | None = None,
    data_gb: float | None = None,
    path: str | None = None,
    checkpoint_strategy: str | None = None,
) -> dict:
    details = {"represents": represents}
    if path is not None:
        details["path"] = path
    if checkpoint_strategy is not None:
        details["checkpoint_strategy"] = checkpoint_strategy
    return {
        "event_id": event_id,
        "start": start,
        "end": end,
        "duration": end - start,
        "job_id": job_id,
        "node": node,
        "physical_node": node,
        "rank": event_id,
        "category": category,
        "operation": operation,
        "source": source,
        "destination": destination,
        "data_gb": data_gb,
        "details": details,
    }


def write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in events),
        encoding="utf-8",
    )


def test_generate_dashboard_aggregates_services_and_cohorts(tmp_path: Path) -> None:
    events = [
        event(
            1,
            job_id="fxl0",
            node="fxl0-cohort-0",
            start=0.0,
            end=10.0,
            category="Training",
            operation="training_iterations",
            represents=16,
        ),
        event(
            2,
            job_id="frontier0",
            node="frontier0-cohort-0",
            start=0.0,
            end=8.0,
            category="Training",
            operation="training_iterations",
            represents=8,
        ),
        event(
            3,
            job_id="fxl0",
            node="fxl0-cohort-0",
            start=10.0,
            end=14.0,
            category="Checkpoint",
            operation="checkpoint_dram_to_crossjob_peers_chunk",
            represents=16,
            source="fxl0-cohort-0/dram",
            destination="frontier0-cohort-0/ssd",
            data_gb=512.0,
            path="peer",
        ),
    ]
    trace = tmp_path / "trace.jsonl"
    output = tmp_path / "aggregate.html"
    write_jsonl(trace, events)

    summary = generate_dashboard(trace, output, buckets=20, top_spans=10)

    assert summary.event_count == 3
    assert summary.physical_nodes == 24
    assert summary.routes[("fxl", "frontier")] == 512.0
    assert summary.checkpoint_paths["peer"] == 16
    document = output.read_text(encoding="utf-8")
    assert "Aggregate Trace Explorer" in document
    assert "Slow-span explorer" in document
    assert "fxl0" in document


def test_iter_events_reads_gzip_and_service_classes(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl.gz"
    row = {"start": 0.0, "end": 1.0, "job_id": "standard_g14"}
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    assert list(iter_events(trace)) == [row]
    assert service_class("standard_g14") == "standard_g"


def test_route_figure_ranks_routes_and_reports_share(tmp_path: Path) -> None:
    log = tmp_path / "routes.jsonl"
    write_jsonl(
        log,
        [
            event(
                1,
                job_id="fxl0",
                node="fxl0-cohort-0",
                start=0,
                end=2,
                category="Checkpoint",
                operation="flush",
                represents=1,
                source="fxl0",
                destination="object-store",
                data_gb=20,
            ),
            event(
                2,
                job_id="frontier0",
                node="frontier0-cohort-0",
                start=2,
                end=3,
                category="Checkpoint",
                operation="flush",
                represents=1,
                source="frontier0",
                destination="fxl0",
                data_gb=5,
            ),
        ],
    )
    output = tmp_path / "aggregate.html"
    summary = generate_dashboard(log, output)

    figure = route_figure(summary)
    assert list(figure.data[0].y)[-1] == "fxl → object-store"
    assert figure.data[0].customdata[-1][0] == 80.0
    assert "Largest cross-service routes" in output.read_text(encoding="utf-8")


def test_local_ssd_write_figure_counts_only_owner_local_dram_writes(
    tmp_path: Path,
) -> None:
    local_fallback = event(
        1,
        job_id="dominant0",
        node="dominant0-cohort-0",
        start=0,
        end=1,
        category="Checkpoint",
        operation="checkpoint_dram_to_local_ssd_chunk",
        represents=8,
        source="dominant0-cohort-0/dram",
        destination="dominant0-cohort-0/ssd",
        data_gb=10,
        path="local_fallback",
    )
    local_fallback["iteration"] = 4
    planned_local = event(
        2,
        job_id="dominant0",
        node="dominant0-cohort-0",
        start=1,
        end=2,
        category="Checkpoint",
        operation="checkpoint_dram_to_local_ssd_chunk",
        represents=8,
        source="dominant0-cohort-0/dram",
        destination="dominant0-cohort-0/ssd",
        data_gb=3,
    )
    planned_local["iteration"] = 6
    excluded = [
        event(
            3,
            job_id="dominant0",
            node="dominant0-cohort-0",
            start=2,
            end=3,
            category="Checkpoint",
            operation="checkpoint_dram_to_crossjob_peers_chunk",
            represents=8,
            source="dominant0-cohort-0/dram",
            destination="helper0-cohort-0/ssd",
            data_gb=100,
            path="peer",
        ),
        event(
            4,
            job_id="dominant0",
            node="dominant0-cohort-0",
            start=3,
            end=4,
            category="Checkpoint",
            operation="checkpoint_peer_dram_to_peer_ssd_chunk",
            represents=8,
            source="helper0-cohort-0/dram",
            destination="helper0-cohort-0/ssd",
            data_gb=200,
        ),
        event(
            5,
            job_id="dominant0",
            node="dominant0-cohort-0",
            start=4,
            end=5,
            category="Checkpoint",
            operation="checkpoint_stage_dram_to_object_store",
            represents=8,
            source="dominant0-cohort-0/dram",
            destination="object-store",
            data_gb=300,
            path="store",
        ),
        event(
            6,
            job_id="dominant0",
            node="dominant0-cohort-0",
            start=5,
            end=6,
            category="Recovery",
            operation="checkpoint_local_ssd_to_dram_recovery_chunk",
            represents=8,
            source="dominant0-cohort-0/ssd",
            destination="dominant0-cohort-0/dram",
            data_gb=400,
        ),
    ]
    trace = tmp_path / "local-ssd.jsonl"
    output = tmp_path / "local-ssd.html"
    write_jsonl(trace, [local_fallback, planned_local, *excluded])

    summary = generate_dashboard(trace, output)
    figure = local_ssd_write_figure(summary)

    assert summary.service_local_ssd_write_gb == {"dominant0": 13}
    assert summary.service_local_ssd_write_events == {"dominant0": 2}
    assert summary.service_local_ssd_write_mode_gb == {
        ("dominant0", "fallback"): 10,
        ("dominant0", "planned"): 3,
    }
    assert summary.service_local_ssd_latest_iteration == {"dominant0": 6}
    assert list(figure.data[0].y) == ["dominant0"]
    assert list(figure.data[0].x) == [3]
    assert list(figure.data[1].x) == [10]
    document = output.read_text(encoding="utf-8")
    assert "Per-job owner-local SSD write traffic" in document
    assert "cumulative physical write traffic" in document


def test_local_ssd_write_figure_has_clear_empty_state(tmp_path: Path) -> None:
    trace = tmp_path / "peer-only.jsonl"
    write_jsonl(trace, [checkpoint_event("crossjob_peer", "peer")])

    figure = local_ssd_write_figure(summarize_trace(trace))

    assert not figure.data
    assert "No completed owner-local" in figure.layout.annotations[0].text


def test_allocated_nodes_do_not_double_count_overlapping_spans(tmp_path: Path) -> None:
    trace = tmp_path / "overlap.jsonl"
    events = [
        event(
            1,
            job_id="fxl0",
            node="fxl0-cohort-0",
            start=0,
            end=10,
            category="Training",
            operation="training_iterations",
            represents=16,
        ),
        event(
            2,
            job_id="fxl0",
            node="fxl0-cohort-0",
            start=2,
            end=8,
            category="Checkpoint",
            operation="flush",
            represents=16,
        ),
    ]
    write_jsonl(trace, events)
    output = tmp_path / "overlap.html"
    summary = generate_dashboard(trace, output, buckets=20)
    activity = aggregate_activity(trace, summary, buckets=20)

    allocated = allocated_nodes_figure(activity, summary)
    assert max(allocated.data[0].y) == 16
    assert max(
        sum(values)
        for values in zip(
            activity.category_values["Training"],
            activity.category_values["Checkpoint"],
        )
    ) > 16


def test_allocated_nodes_follow_dynamic_arrival_lifecycle(tmp_path: Path) -> None:
    trace = tmp_path / "dynamic-nodes.jsonl"
    events = [
        event(
            1, job_id="early0", node="early0-lifecycle", start=0, end=0,
            category="Lifecycle", operation="job_arrival", represents=1,
        ),
        event(
            2, job_id="late0", node="late0-lifecycle", start=0, end=5,
            category="Lifecycle", operation="job_pending", represents=1,
        ),
        event(
            3, job_id="early0", node="early0-cohort-0", start=0, end=10,
            category="Training", operation="training_iterations", represents=4,
        ),
        event(
            4, job_id="late0", node="late0-lifecycle", start=5, end=5,
            category="Lifecycle", operation="job_arrival", represents=1,
        ),
        event(
            5, job_id="late0", node="late0-cohort-0", start=5, end=10,
            category="Training", operation="training_iterations", represents=3,
        ),
        event(
            6, job_id="early0", node="early0-lifecycle", start=10, end=10,
            category="Lifecycle", operation="job_idle_checkpoint", represents=1,
        ),
        event(
            7, job_id="late0", node="late0-lifecycle", start=10, end=10,
            category="Lifecycle", operation="job_idle_checkpoint", represents=1,
        ),
    ]
    write_jsonl(trace, events)

    summary = summarize_trace(trace)
    activity = aggregate_activity(trace, summary, buckets=20)
    figure = allocated_nodes_figure(activity, summary)
    by_class = {series.name: series for series in figure.data}

    assert summary.physical_nodes == 7
    assert summary.service_arrival == {"early0": 0.0, "late0": 5.0}
    assert max(by_class["early"].y) == 4
    assert max(by_class["late"].y) == 3
    assert all(
        value == 0
        for center, value in zip(activity.centers, by_class["late"].y)
        if center < 5
    )
    assert min(
        center
        for center, value in zip(activity.centers, by_class["late"].y)
        if value > 0
    ) >= 5
    assert by_class["late"].line.shape == "hv"
    assert max(by_class["Idle checkpoint donors"].y) == 7


def test_scenario_and_result_metadata_add_efficiency_metrics(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_jsonl(
        trace,
        [
            event(
                1,
                job_id="tiny0",
                node="tiny0-cohort-0",
                start=0,
                end=20,
                category="Training",
                operation="training_iterations",
                represents=4,
            )
        ],
    )
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        """
cluster: {node_count: 8, network_bandwidth_gbps: 8}
classes:
  tiny:
    count: 1
    ranks: 4
    iterations: 2
    iteration_seconds: 4
    checkpoint_every: 2
    checkpoint_gb_per_rank: 1
    kpeers: 2
    rpo_s: 8
arms:
  ours_full: {kpeers: true}
""",
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "arm": "ours_full",
                        "seed": 7,
                        "makespan_s": 20,
                        "failures": {"node": 1},
                        "restores": 1,
                        "recovery_source_tiers": {"dram": 1},
                        "durable_frac": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "with-context.html"
    generate_dashboard(
        trace,
        output,
        scenario_path=scenario,
        result_path=result,
        arm="ours_full",
        seed=7,
    )
    document = output.read_text(encoding="utf-8")

    assert "Weighted training goodput" in document
    assert "Recovery from dram" in document
    assert "Per-job efficiency and recovery" in document
    assert "Equal / not configured" in document
    assert "Durable checkpoints" in document


def test_same_class_cross_job_routes_are_kept_but_same_job_routes_are_not(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "same-class.jsonl"
    common = {
        "job_id": "frontier0",
        "node": "frontier0-cohort-0",
        "category": "Checkpoint",
        "operation": "flush",
        "represents": 1,
    }
    write_jsonl(
        trace,
        [
            event(
                1,
                start=0,
                end=1,
                source="frontier0/dram",
                destination="frontier4/ssd",
                data_gb=10,
                **common,
            ),
            event(
                2,
                start=1,
                end=2,
                source="frontier0/dram",
                destination="frontier0/ssd",
                data_gb=99,
                **common,
            ),
        ],
    )

    summary = summarize_trace(trace)

    assert summary.routes[("frontier", "frontier")] == 10
    assert summary.route_events[("frontier", "frontier")] == 1
    topology = topology_figure(summary)
    assert "frontier (other jobs)" in topology.data[0].node.label
    assert topology.data[0].link.source[0] != topology.data[0].link.target[0]


def policy_scenario(arms: str) -> str:
    return f"""
cluster:
  node_count: 8
  gpu_cpu_bandwidth_gbps: 40
  network_bandwidth_gbps: 8
  local_ssd_bandwidth_gbps: 4
classes:
  tiny:
    count: 1
    ranks: 4
    iterations: 20
    iteration_seconds: 10
    checkpoint_every: 2
    checkpoint_gb_per_rank: 1
    kpeers: 2
    rpo_s: 100
controller: {{slot_period_s: 60}}
arms:
{arms}
"""


def checkpoint_event(strategy: str, path: str) -> dict:
    return event(
        1,
        job_id="tiny0",
        node="tiny0-cohort-0",
        start=0,
        end=1,
        category="Checkpoint",
        operation="checkpoint_stage",
        represents=4,
        source="tiny0/dram",
        destination="object-store",
        data_gb=4,
        path=path,
        checkpoint_strategy=strategy,
    )


def test_named_baseline_does_not_claim_cross_job_pipeline(tmp_path: Path) -> None:
    trace = tmp_path / "gemini.jsonl"
    write_jsonl(trace, [checkpoint_event("gemini", "gemini_replica")])
    scenario = tmp_path / "gemini.yaml"
    scenario.write_text(
        policy_scenario("  gemini: {baseline: gemini}\n"),
        encoding="utf-8",
    )
    output = tmp_path / "gemini.html"

    generate_dashboard(
        trace,
        output,
        scenario_path=scenario,
        arm="gemini",
    )
    document = output.read_text(encoding="utf-8")

    assert "Configured policy" in document
    assert "GEMINI (SOSP&#x27;23) baseline" in document
    assert "Observed trace behavior" in document
    assert "in-job peer DRAM" in document
    assert "striped onto SSDs owned by other jobs" not in document


def test_naive_tiered_arm_reports_each_configured_cadence(tmp_path: Path) -> None:
    trace = tmp_path / "naive.jsonl"
    write_jsonl(trace, [checkpoint_event("crossjob_peer", "peer")])
    scenario = tmp_path / "naive.yaml"
    scenario.write_text(
        policy_scenario(
            """  naive10_100:
    kpeers: true
    slots: true
    cadence_l1_s: 10
    cadence_l2_s: 100
    cadence_l3_s: 100
"""
        ),
        encoding="utf-8",
    )
    context = load_run_context(scenario_path=scenario, arm="naive10_100")
    summary = summarize_trace(trace)

    configured = configured_checkpoint_policy(context)
    figure = checkpoint_policy_figure(summary, context)
    values = figure.data[0].cells.values

    assert configured.label == "Hand-set tiered cadence"
    assert "L1 capture at 10s" in configured.description
    assert "L2 persistence at 100s" in configured.description
    assert values[3][0].startswith("10s (")
    assert values[4][0].startswith("100s (")
    assert values[5][0].startswith("100s (")


def test_ideal_arm_reports_checkpointing_disabled(tmp_path: Path) -> None:
    trace = tmp_path / "ideal.jsonl"
    write_jsonl(
        trace,
        [
            event(
                1,
                job_id="tiny0",
                node="tiny0-cohort-0",
                start=0,
                end=10,
                category="Training",
                operation="training_iterations",
                represents=4,
            )
        ],
    )
    scenario = tmp_path / "ideal.yaml"
    scenario.write_text(
        policy_scenario("  ideal: {ideal: true, kpeers: false}\n"),
        encoding="utf-8",
    )
    output = tmp_path / "ideal.html"

    generate_dashboard(trace, output, scenario_path=scenario, arm="ideal")
    document = output.read_text(encoding="utf-8")

    assert "Ideal comparator (checkpointing disabled)" in document
    assert "No checkpoint strategy observed" in document
    assert "striped onto SSDs owned by other jobs" not in document


def test_gp_result_policy_uses_embedded_decisions_not_yaml_defaults(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "gp.jsonl"
    write_jsonl(trace, [checkpoint_event("crossjob_peer", "peer")])
    scenario = tmp_path / "gp.yaml"
    scenario.write_text(
        policy_scenario("  ours_full: {kpeers: true}\n"),
        encoding="utf-8",
    )
    result = tmp_path / "gp-result.json"
    result.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "arm": "ours_gp_rpo",
                        "seed": 7,
                        "policy_kind": "gp",
                        "policy_name": "ours_gp_rpo",
                        "policy_by_class": {
                            "tiny": {
                                "capture_every": 3,
                                "checkpoint_every": 4,
                                "store_every": 20,
                                "kpeers": 2,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    context = load_run_context(
        scenario_path=scenario,
        result_path=result,
        arm="ours_gp_rpo",
        seed=7,
    )
    summary = summarize_trace(trace)

    configured = configured_checkpoint_policy(context)
    figure = checkpoint_policy_figure(summary, context)
    values = figure.data[0].cells.values

    assert configured.label == "GP-solved policy: ours_gp_rpo"
    assert values[3][0].startswith("3 it (")
    assert values[4][0].startswith("4 it (")
    assert values[5][0].startswith("20 it (")
    assert values[6][0] == "policy-selected / k=2"


def test_external_result_without_policy_details_hides_scenario_defaults(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "external.jsonl"
    write_jsonl(trace, [checkpoint_event("crossjob_peer", "peer")])
    scenario = tmp_path / "external.yaml"
    scenario.write_text(
        policy_scenario("  ours_full: {kpeers: true}\n"),
        encoding="utf-8",
    )
    result = tmp_path / "external-result.json"
    result.write_text(
        json.dumps({"rows": [{"arm": "external-solve", "seed": 7}]}),
        encoding="utf-8",
    )
    context = load_run_context(
        scenario_path=scenario,
        result_path=result,
        arm="external-solve",
        seed=7,
    )
    summary = summarize_trace(trace)

    figure = checkpoint_policy_figure(summary, context)
    values = figure.data[0].cells.values

    assert values[3][0] == "external policy; not embedded"
    assert values[4][0] == "external policy; not embedded"
    assert values[5][0] == "external policy; not embedded"
