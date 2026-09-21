from __future__ import annotations

import argparse
import gzip
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio


# One fixed color per known operation. Fallback colors are generated from a
# deterministic hue sequence, so newly added operations also remain distinct.
OPERATION_STYLES: dict[str, tuple[str, str]] = {
    "data_parallel_forward": ("Data-parallel forward", "#1D4ED8"),
    "data_parallel_backward": ("Data-parallel backward", "#EA580C"),
    "data_parallel_optimizer": ("Data-parallel optimizer", "#16A34A"),
    "data_parallel_sync_wait": ("Waiting for all-reduce", "#94A3B8"),
    "pipeline_flush_wait": ("Pipeline flush wait", "#64748B"),
    "pipeline_forward": ("Pipeline forward", "#0072B2"),
    "pipeline_backward": ("Pipeline backward", "#D55E00"),
    "pipeline_optimizer": ("Pipeline optimizer", "#009E73"),
    "pipeline_sync_wait": ("Waiting for synchronization", "#94A3B8"),
    "pipeline_activation_send": ("Activation transfer", "#56B4E9"),
    "pipeline_activation_receive": ("Activation receive", "#0284C7"),
    "pipeline_gradient_send": ("Pipeline gradient transfer", "#CC79A7"),
    "pipeline_gradient_receive": ("Pipeline gradient receive", "#A21CAF"),
    "pipeline_transfer_launch": ("PyTorch P2P launch", "#F59E0B"),
    "data_parallel_all_reduce": ("Data-parallel all-reduce", "#7B2CBF"),
    "checkpoint_stage_gpu_to_dram": (
        "Stage checkpoint: GPU → DRAM",
        "#E69F00",
    ),
    "checkpoint_stage_dram_to_object_store": (
        "Stage checkpoint: DRAM → object store",
        "#8B4513",
    ),
    "checkpoint_dram_to_local_ssd_chunk": (
        "Checkpoint chunk: DRAM to local SSD",
        "#A16207",
    ),
    "checkpoint_dram_to_peer_dram_chunk": (
        "Checkpoint chunk: DRAM to paired DRAM",
        "#0F766E",
    ),
    "checkpoint_peer_dram_to_peer_ssd_chunk": (
        "Checkpoint chunk: paired DRAM to paired SSD",
        "#7C3AED",
    ),
    "checkpoint_dram_to_crossjob_peers_chunk": (
        "Checkpoint chunk: DRAM to cross-job donors",
        "#008300",
    ),
    "checkpoint_crossjob_peers_to_dram_recovery_chunk": (
        "Recovery chunk: cross-job donors to DRAM",
        "#4A3AA7",
    ),
    "checkpoint_ssd_to_dram_recovery_chunk": (
        "Recovery chunk: local SSD to DRAM",
        "#EAB308",
    ),
    "checkpoint_peer_dram_to_dram_recovery_chunk": (
        "Recovery chunk: paired DRAM to DRAM",
        "#14B8A6",
    ),
    "checkpoint_peer_ssd_to_dram_recovery_chunk": (
        "Recovery chunk: paired SSD to DRAM",
        "#8B5CF6",
    ),
    "initial_state_to_dram_chunk": (
        "Recovery chunk: initial state to DRAM",
        "#DC2626",
    ),
    "forward_pass": ("Forward pass", "#0072B2"),
    "backward_pass": ("Backward pass", "#D55E00"),
    "optimizer_step": ("Optimizer step", "#009E73"),
    "ring_all_reduce_reduce_scatter": ("Reduce-scatter", "#CC79A7"),
    "ring_all_reduce_all_gather": ("All-gather", "#56B4E9"),
    "checkpoint_gpu_to_dram": ("Checkpoint: GPU → DRAM", "#E69F00"),
    "checkpoint_dram_to_object_store": (
        "Checkpoint: DRAM → object store",
        "#8B4513",
    ),
    "object_store_to_dram": ("Recovery: object store → DRAM", "#F0E442"),
    "dram_to_gpu_restore": ("Recovery: DRAM → GPU", "#6A3D9A"),
    "process_failure_restart": ("Process restart", "#E31A1C"),
    "node_failure_restart": ("Node restart", "#111111"),
    "spot_failure_restart": ("Spot replacement", "#FF1493"),
    "job_pending": ("Waiting for configured arrival", "#ADB5BD"),
    "job_arrival": ("Job arrival", "#2B8A3E"),
    "job_departure": ("Job departure", "#495057"),
    "job_idle_checkpoint": ("Idle checkpoint donor", "#6741D9"),
}

OPERATION_PATTERNS: dict[str, str] = {
    "data_parallel_forward": "/",
    "data_parallel_backward": "\\",
    "data_parallel_optimizer": ".",
    "data_parallel_sync_wait": "",
    "pipeline_flush_wait": "",
    "pipeline_forward": "/",
    "pipeline_backward": "\\",
    "pipeline_optimizer": ".",
    "pipeline_sync_wait": "",
    "pipeline_activation_send": "x",
    "pipeline_activation_receive": "x",
    "pipeline_gradient_send": "+",
    "pipeline_gradient_receive": "+",
    "pipeline_transfer_launch": ".",
    "data_parallel_all_reduce": "|",
    "checkpoint_stage_gpu_to_dram": "-",
    "checkpoint_stage_dram_to_object_store": "|",
    "checkpoint_dram_to_local_ssd_chunk": "|",
    "checkpoint_dram_to_peer_dram_chunk": "+",
    "checkpoint_peer_dram_to_peer_ssd_chunk": ".",
    "checkpoint_ssd_to_dram_recovery_chunk": "/",
    "checkpoint_peer_dram_to_dram_recovery_chunk": "x",
    "checkpoint_peer_ssd_to_dram_recovery_chunk": "\\",
    "initial_state_to_dram_chunk": "-",
    "forward_pass": "/",
    "backward_pass": "\\",
    "optimizer_step": ".",
    "ring_all_reduce_reduce_scatter": "x",
    "ring_all_reduce_all_gather": "+",
    "checkpoint_gpu_to_dram": "-",
    "checkpoint_dram_to_object_store": "|",
    "object_store_to_dram": "/",
    "dram_to_gpu_restore": "\\",
    "process_failure_restart": "x",
    "node_failure_restart": "+",
    "spot_failure_restart": ".",
}

FAILURE_LINE_STYLES: dict[str, tuple[str, str]] = {
    "process": ("#E31A1C", "dash"),
    "node": ("#111111", "dashdot"),
    "spot": ("#FF1493", "dot"),
}

COHORT_NODE_PATTERN = re.compile(r"^ranks-(\d+)-(\d+)$")


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Log file not found: {path}. Run `make run` first."
        )

    events: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    if not events:
        raise ValueError(f"No events were found in {path}")
    return events


def build_operation_styles(
    events: list[dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    operations = sorted({str(event.get("operation", "unknown")) for event in events})
    styles = dict(OPERATION_STYLES)

    unknown = [operation for operation in operations if operation not in styles]
    for index, operation in enumerate(unknown):
        # Golden-angle hue spacing avoids repeated fallback colors.
        hue = int((index * 137.508 + 18) % 360)
        readable = operation.replace("_", " ").strip().title()
        styles[operation] = (readable, f"hsl({hue}, 68%, 43%)")

    colors = [styles[operation][1] for operation in operations]
    if len(colors) != len(set(colors)):
        raise RuntimeError("Every process type must have a unique color")
    return styles


def add_failure_markers(
    figure: go.Figure,
    events: list[dict[str, Any]],
    *,
    include_labels: bool,
) -> None:
    failures = sorted(
        (event for event in events if event.get("category") == "Failure"),
        key=lambda event: float(event.get("start", 0.0)),
    )

    for index, event in enumerate(failures):
        failure_type = str(event.get("failure_type") or "failure")
        color, dash = FAILURE_LINE_STYLES.get(
            failure_type,
            ("#DC2626", "dash"),
        )
        failure_time = float(event["start"])
        figure.add_shape(
            type="line",
            x0=failure_time,
            x1=failure_time,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=color, width=1.25, dash=dash),
            opacity=0.45,
            layer="above",
        )

        if include_labels:
            rank = event.get("rank")
            rank_label = (
                f"{int(rank):,}" if rank is not None else "unknown"
            )
            simultaneous = int(
                (event.get("details") or {}).get(
                    "simultaneous_failed_nodes",
                    1,
                )
            )
            simultaneous_label = (
                f" · {simultaneous} nodes same tick"
                if simultaneous > 1
                else ""
            )
            label = (
                f"{failure_type.upper()} · rank {rank_label} · "
                f"iter {event.get('iteration')}{simultaneous_label}"
            )
            figure.add_annotation(
                x=failure_time,
                y=0.99 - 0.08 * (index % 2),
                xref="x",
                yref="paper",
                text=label,
                textangle=-90,
                showarrow=False,
                font=dict(color=color, size=10),
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
            )


def add_failure_points(
    figure: go.Figure,
    events: list[dict[str, Any]],
    *,
    failed_ranks: set[tuple[str, int]],
    cohort_bounds: dict[str, tuple[int, int]],
    strip_job_prefix: str | None = None,
) -> None:
    failures = sorted(
        (event for event in events if event.get("category") == "Failure"),
        key=lambda event: float(event.get("start", 0.0)),
    )
    if not failures:
        return

    lanes: list[str] = []
    colors: list[str] = []
    for event in failures:
        owner_label, _ = resource_lane_identity(
            event,
            failed_ranks=failed_ranks,
            cohort_bounds=cohort_bounds,
        )
        if strip_job_prefix and owner_label.startswith(strip_job_prefix):
            owner_label = owner_label[len(strip_job_prefix) :]
        lanes.append(f"{owner_label} — GPU")
        failure_type = str(event.get("failure_type") or "failure")
        colors.append(
            FAILURE_LINE_STYLES.get(
                failure_type,
                ("#DC2626", "dash"),
            )[0]
        )

    figure.add_trace(
        go.Scatter(
            name="Failure start",
            legendgroup="failure-start",
            mode="markers",
            x=[float(event["start"]) for event in failures],
            y=lanes,
            marker=dict(
                symbol="x",
                size=11,
                color=colors,
                line=dict(width=2),
            ),
            customdata=[event_customdata(event) for event in failures],
            hovertemplate=HOVER_TEMPLATE,
        )
    )


def event_customdata(row: dict[str, Any]) -> list[Any]:
    details = row.get("details") or {}
    rank = row.get("rank")
    rank_label = f"{int(rank):,}" if rank is not None else None
    if row.get("_visual_cohort_clone"):
        representation = "failed rank expanded from worker cohort"
    elif COHORT_NODE_PATTERN.fullmatch(str(row.get("node") or "")):
        representation = "synchronized worker cohort"
    else:
        representation = "individual rank"

    return [
        row.get("category"),
        row.get("operation"),
        row.get("iteration"),
        row.get("start"),
        row.get("end"),
        row.get("duration"),
        row.get("failure_type"),
        row.get("source"),
        row.get("destination"),
        row.get("data_gb"),
        details.get("link"),
        details.get("base_duration"),
        details.get("effective_slowdown", 1.0),
        details.get("contention_seconds", 0.0),
        rank_label,
        row.get("node"),
        representation,
        details.get("failure_stage"),
        details.get("failure_progress"),
        row.get("job_id"),
        row.get("pipeline_stage"),
        row.get("data_parallel_rank"),
        row.get("physical_node"),
        details.get("highest_severity"),
        details.get("coalesced_failure_signals"),
        details.get("simultaneous_failed_nodes"),
        details.get("failure_wait_seconds", 0.0),
        details.get("observed_failure_types"),
        details.get("checkpoint_partner_rank"),
        details.get("checkpoint_source_tier"),
        details.get("checkpoint_source_rank"),
        details.get("chunk_index"),
        details.get("chunk_count"),
        details.get("bandwidth_contention_seconds", 0.0),
        details.get("peer_pair_checkpoint_gb"),
    ]


HOVER_TEMPLATE = (
    "<b>%{customdata[1]}</b><br>"
    "category=%{customdata[0]}<br>"
    "rank=%{customdata[14]}<br>"
    "node=%{customdata[15]}<br>"
    "representation=%{customdata[16]}<br>"
    "failure stage=%{customdata[17]}<br>"
    "failure progress=%{customdata[18]}<br>"
    "job=%{customdata[19]}<br>"
    "pipeline stage=%{customdata[20]}<br>"
    "data replica=%{customdata[21]}<br>"
    "physical node=%{customdata[22]}<br>"
    "highest failure=%{customdata[23]}<br>"
    "coalesced signals=%{customdata[24]}<br>"
    "nodes failed same tick=%{customdata[25]}<br>"
    "failure wait=%{customdata[26]}s<br>"
    "observed failures=%{customdata[27]}<br>"
    "checkpoint partner=%{customdata[28]}<br>"
    "checkpoint source tier=%{customdata[29]}<br>"
    "checkpoint source rank=%{customdata[30]}<br>"
    "chunk=%{customdata[31]}/%{customdata[32]}<br>"
    "bandwidth contention=%{customdata[33]}s<br>"
    "pair checkpoint data=%{customdata[34]} GB<br>"
    "iteration=%{customdata[2]}<br>"
    "start=%{customdata[3]:.3f}s<br>"
    "end=%{customdata[4]:.3f}s<br>"
    "actual duration=%{customdata[5]:.3f}s<br>"
    "base duration=%{customdata[11]}s<br>"
    "effective slowdown=%{customdata[12]:.3f}×<br>"
    "contention time=%{customdata[13]:.3f}s<br>"
    "failure=%{customdata[6]}<br>"
    "source=%{customdata[7]}<br>"
    "destination=%{customdata[8]}<br>"
    "data=%{customdata[9]} GB<br>"
    "link=%{customdata[10]}"
    "<extra></extra>"
)


def expand_failed_ranks_from_cohort(
    events: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    set[tuple[str, int]],
    dict[str, list[tuple[int, int]]],
]:
    """
    Give failed logical ranks full lanes when the simulation used aggregation.

    The synchronized worker cohort already contains the timing each worker
    experienced. Cloning those intervals for failed ranks is a visualization
    expansion and does not change the simulation timing.
    """

    cohort_bounds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    cohort_events: dict[
        tuple[str, tuple[int, int]],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for event in events:
        match = COHORT_NODE_PATTERN.fullmatch(str(event.get("node") or ""))
        if match is None:
            continue
        job_id = str(event.get("job_id") or "")
        bounds = (int(match.group(1)), int(match.group(2)))
        if bounds not in cohort_bounds[job_id]:
            cohort_bounds[job_id].append(bounds)
        cohort_events[(job_id, bounds)].append(event)

    failed_ranks = {
        (str(event.get("job_id") or ""), int(event["rank"]))
        for event in events
        if event.get("category") == "Failure"
        and event.get("rank") is not None
    }
    if not cohort_bounds:
        return events, failed_ranks, {}

    expanded_events = list(events)
    for job_id, bounds_list in cohort_bounds.items():
        for bounds in bounds_list:
            cohort_start, cohort_end = bounds
            expanded_ranks = sorted(
                rank
                for failed_job, rank in failed_ranks
                if failed_job == job_id
                and cohort_start <= rank <= cohort_end
                and not any(
                    event.get("category") != "Failure"
                    and event.get("job_id") == job_id
                    and event.get("rank") == rank
                    and event.get("node") == f"rank-{rank}"
                    for event in events
                )
            )
            for rank in expanded_ranks:
                for event in cohort_events[(job_id, bounds)]:
                    details = dict(event.get("details") or {})
                    details["represented_ranks"] = 1
                    if details.get("aggregate") and "steps_per_rank" in details:
                        details["transfers"] = int(details["steps_per_rank"])

                    clone = {
                        **event,
                        "rank": rank,
                        "node": f"rank-{rank}",
                        "details": details,
                        "_visual_cohort_clone": True,
                    }
                    if clone.get("source") == event.get("node"):
                        clone["source"] = f"rank-{rank}"
                    expanded_events.append(clone)

    return expanded_events, failed_ranks, dict(cohort_bounds)


def resource_lane_identity(
    event: dict[str, Any],
    *,
    failed_ranks: set[tuple[str, int]],
    cohort_bounds: dict[str, list[tuple[int, int]]],
) -> tuple[str, tuple[str, int, int]]:
    rank = int(event["rank"])
    node = str(event.get("node") or "")
    job_id = str(event.get("job_id") or "")
    job_prefix = f"{job_id} / " if job_id else ""
    cohort_match = COHORT_NODE_PATTERN.fullmatch(node)

    if cohort_match is not None:
        start, end = int(cohort_match.group(1)), int(cohort_match.group(2))
        failed_in_cohort = sum(
            failed_job == job_id and start <= failed_rank <= end
            for failed_job, failed_rank in failed_ranks
        )
        remaining = end - start + 1 - failed_in_cohort
        rank_word = "rank" if remaining == 1 else "ranks"
        return (
            f"{job_prefix}remaining workers "
            f"({remaining:,} {rank_word}, aggregated)",
            (job_id, 1, start),
        )

    failed_suffix = " (FAILED)" if (job_id, rank) in failed_ranks else ""
    stage = event.get("pipeline_stage")
    replica = event.get("data_parallel_rank")
    role = (
        f" · DP {replica} · stage {stage}"
        if stage is not None and replica is not None
        else ""
    )
    label = f"{job_prefix}rank {rank:,}{role}{failed_suffix}"
    if any(start <= rank <= end for start, end in cohort_bounds.get(job_id, [])):
        return label, (job_id, 2, rank)
    return label, (job_id, 0, rank)


def add_operation_bars(
    figure: go.Figure,
    rows: list[dict[str, Any]],
    *,
    lane_key: str,
    styles: dict[str, tuple[str, str]],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("operation", "unknown"))].append(row)

    ordered_operations = sorted(
        grouped,
        key=lambda operation: min(
            float(row.get("start", 0.0)) for row in grouped[operation]
        ),
    )

    for operation in ordered_operations:
        operation_rows = grouped[operation]
        display_name, color = styles[operation]
        figure.add_trace(
            go.Bar(
                name=display_name,
                legendgroup=operation,
                orientation="h",
                y=[row[lane_key] for row in operation_rows],
                x=[float(row["duration"]) for row in operation_rows],
                base=[float(row["start"]) for row in operation_rows],
                marker=dict(
                    color=color,
                    line=dict(width=0.8, color="white"),
                    pattern=dict(
                        shape=OPERATION_PATTERNS.get(operation, ""),
                        solidity=0.28,
                    ),
                ),
                customdata=[event_customdata(row) for row in operation_rows],
                hovertemplate=HOVER_TEMPLATE,
            )
        )


def timeline_layout(
    *,
    title: str,
    subtitle: str,
    lane_title: str,
    lane_order: list[str],
    height_per_lane: int,
) -> dict[str, Any]:
    longest_lane = max((len(lane) for lane in lane_order), default=0)
    # Category labels are rendered at roughly 7-8 px per character. Supplying
    # enough initial margin plus automargin prevents large rank IDs from being
    # clipped before Plotly can measure the browser's actual font metrics.
    left_margin = max(180, longest_lane * 8 + 45)

    return dict(
        title={
            "text": f"{title}<br><sup>{subtitle}</sup>",
            "x": 0.01,
            "xanchor": "left",
        },
        barmode="overlay",
        bargap=0.16,
        height=max(520, 70 * len(lane_order) + height_per_lane),
        xaxis=dict(
            title="Simulated time (seconds)",
            showgrid=True,
            zeroline=False,
            rangeslider=dict(visible=True, thickness=0.08),
        ),
        yaxis=dict(
            title=lane_title,
            categoryorder="array",
            categoryarray=lane_order,
            autorange="reversed",
            fixedrange=False,
            automargin=True,
            ticklabeloverflow="allow",
        ),
        legend=dict(
            title="Process",
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
        hovermode="closest",
        dragmode="zoom",
        margin=dict(l=left_margin, r=285, t=125, b=80, autoexpand=True),
    )


def highlight_remaining_cohort(
    figure: go.Figure,
    lane_order: list[str],
) -> None:
    """Draw a red outline around all remaining-worker lanes."""

    indices = [
        index
        for index, lane in enumerate(lane_order)
        if lane.startswith("remaining workers")
        or "/ remaining workers" in lane
    ]
    if not indices:
        return

    figure.add_shape(
        type="rect",
        xref="paper",
        x0=0,
        x1=1,
        yref="y",
        y0=min(indices) - 0.45,
        y1=max(indices) + 0.45,
        line=dict(color="#DC2626", width=3),
        fillcolor="rgba(220, 38, 38, 0.035)",
        layer="above",
    )


def make_resource_timeline(
    events: list[dict[str, Any]],
    styles: dict[str, tuple[str, str]],
) -> go.Figure:
    visual_events, failed_ranks, cohort_bounds = (
        expand_failed_ranks_from_cohort(events)
    )
    job_ids = {
        str(event["job_id"])
        for event in events
        if event.get("job_id") is not None
    }
    strip_job_prefix = (
        f"{next(iter(job_ids))} / " if len(job_ids) == 1 else None
    )
    rows: list[dict[str, Any]] = []
    lane_sort_keys: dict[str, tuple[str, int, int]] = {}

    for event in visual_events:
        if event.get("category") == "Failure":
            continue
        rank = event.get("rank")
        if rank is None:
            continue
        rank = int(rank)
        owner_label, sort_key = resource_lane_identity(
            event,
            failed_ranks=failed_ranks,
            cohort_bounds=cohort_bounds,
        )
        if strip_job_prefix and owner_label.startswith(strip_job_prefix):
            owner_label = owner_label[len(strip_job_prefix) :]
        lane_sort_keys[owner_label] = sort_key
        for resource in event.get("resources", []):
            if resource not in {"CPU", "GPU"}:
                continue
            rows.append(
                {
                    **event,
                    "rank": rank,
                    "resource": resource,
                    "lane": f"{owner_label} — {resource}",
                }
            )

    if not rows:
        return go.Figure().update_layout(
            title="No CPU or GPU occupancy events found"
        )

    # With autorange='reversed', this array is the visible top-to-bottom order.
    lane_order = [
        f"{owner_label} — {resource}"
        for owner_label in sorted(
            lane_sort_keys,
            key=lambda label: lane_sort_keys[label],
        )
        for resource in ("GPU", "CPU")
    ]

    figure = go.Figure()
    add_operation_bars(figure, rows, lane_key="lane", styles=styles)
    add_failure_markers(figure, events, include_labels=False)
    add_failure_points(
        figure,
        events,
        failed_ranks=failed_ranks,
        cohort_bounds=cohort_bounds,
        strip_job_prefix=strip_job_prefix,
    )
    configured_jobs = any(event.get("job_id") for event in events)
    subtitle = (
        "Lanes identify job, data replica, and pipeline stage"
        if configured_jobs
        else (
            "Failed aggregate ranks are expanded from the synchronized "
            "worker cohort · vertical lines mark failures"
        )
    )
    figure.update_layout(
        **timeline_layout(
            title="GPU and CPU occupancy timeline",
            subtitle=subtitle,
            lane_title="Rank resource",
            lane_order=lane_order,
            height_per_lane=90,
        )
    )
    highlight_remaining_cohort(figure, lane_order)
    return figure


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def link_lane(
    event: dict[str, Any],
    *,
    failed_ranks: set[tuple[str, int]],
    cohort_bounds: dict[str, list[tuple[int, int]]],
) -> str:
    details = event.get("details") or {}
    if details.get("aggregate") and str(event.get("operation", "")).startswith(
        "ring_all_reduce_"
    ):
        owner_label, _ = resource_lane_identity(
            event,
            failed_ranks=failed_ranks,
            cohort_bounds=cohort_bounds,
        )
        return f"{owner_label} — ring"

    explicit_link = details.get("link")
    if explicit_link:
        return str(explicit_link)

    source = str(event.get("source") or "unknown")
    destination = str(event.get("destination") or "unknown")
    if source == "object-store":
        return f"{destination}-object-store"
    if destination == "object-store":
        return f"{source}-object-store"
    return f"{source} → {destination}"


def make_link_timeline(
    events: list[dict[str, Any]],
    styles: dict[str, tuple[str, str]],
) -> go.Figure:
    visual_events, failed_ranks, cohort_bounds = (
        expand_failed_ranks_from_cohort(events)
    )
    job_ids = {
        str(event["job_id"])
        for event in events
        if event.get("job_id") is not None
    }
    strip_link_prefix = (
        f"{next(iter(job_ids))}/" if len(job_ids) == 1 else None
    )
    rows: list[dict[str, Any]] = []
    for event in visual_events:
        if "NETWORK" not in event.get("resources", []):
            continue
        if not event.get("source") or not event.get("destination"):
            continue
        lane = link_lane(
            event,
            failed_ranks=failed_ranks,
            cohort_bounds=cohort_bounds,
        )
        if strip_link_prefix and lane.startswith(strip_link_prefix):
            lane = lane[len(strip_link_prefix) :]
        rows.append({**event, "link_lane": lane})

    if not rows:
        return go.Figure().update_layout(title="No link occupancy events found")

    lanes = sorted({row["link_lane"] for row in rows}, key=natural_key)
    ring_lanes = [
        lane
        for lane in lanes
        if lane.startswith("ring-") or lane.endswith("— ring")
    ]
    object_lanes = [lane for lane in lanes if "object-store" in lane]
    other_lanes = [
        lane for lane in lanes if lane not in ring_lanes and lane not in object_lanes
    ]
    lane_order = ring_lanes + object_lanes + other_lanes

    figure = go.Figure()
    add_operation_bars(figure, rows, lane_key="link_lane", styles=styles)
    add_failure_markers(figure, events, include_labels=False)
    figure.update_layout(
        **timeline_layout(
            title="Link occupancy timeline",
            subtitle=(
                "Each lane is a job-scoped pipeline, data-parallel, or "
                "object-store path"
            ),
            lane_title="Network link",
            lane_order=lane_order,
            height_per_lane=80,
        )
    )
    highlight_remaining_cohort(figure, lane_order)
    return figure


def summary_cards(events: list[dict[str, Any]]) -> str:
    simulation_end = max(float(event["end"]) for event in events)
    failures = sum(1 for event in events if event["category"] == "Failure")
    checkpoint_iterations = sorted(
        {
            int(event["iteration"])
            for event in events
            if event["operation"] in {
                "checkpoint_dram_to_object_store",
                "checkpoint_stage_dram_to_object_store",
                "checkpoint_dram_to_local_ssd_chunk",
                "checkpoint_dram_to_peer_dram_chunk",
                "checkpoint_peer_dram_to_peer_ssd_chunk",
            }
            and event.get("iteration") is not None
        }
    )
    jobs = {
        str(event["job_id"])
        for event in events
        if event.get("job_id") is not None
    }
    communication_gb = sum(
        float(event.get("data_gb") or 0.0)
        for event in events
        if "NETWORK" in event.get("resources", [])
    )
    slowed = sum(
        1
        for event in events
        if float((event.get("details") or {}).get("effective_slowdown", 1.0))
        > 1.000001
    )

    checkpoint_text = (
        f"{checkpoint_iterations[0]}–{checkpoint_iterations[-1]}"
        if checkpoint_iterations
        else "none"
    )
    values = [
        ("Simulated time", f"{simulation_end:.2f} s"),
        ("Jobs", str(len(jobs)) if jobs else "legacy"),
        ("Failure events", str(failures)),
        ("Checkpoint iterations", checkpoint_text),
        ("Contention-slowed events", str(slowed)),
        ("Logged network traffic", f"{communication_gb:.1f} GB"),
    ]
    return "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in values
    )


def write_dashboard(
    events: list[dict[str, Any]],
    *,
    output_path: Path,
    source_log: Path,
) -> None:
    plot_config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
    }
    configured_job_ids = sorted(
        {
            str(event["job_id"])
            for event in events
            if event.get("job_id") is not None
        }
    )
    configured_jobs = bool(configured_job_ids)
    event_groups = (
        [("All jobs", events)]
        + [
            (
                job_id,
                [event for event in events if event.get("job_id") == job_id],
            )
            for job_id in configured_job_ids
        ]
        if configured_jobs
        else [("Simulation", events)]
    )

    tab_buttons: list[str] = []
    job_panels: list[str] = []
    plot_ids: list[str] = []
    include_plotlyjs = True
    for index, (job_id, job_events) in enumerate(event_groups):
        safe_job_id = re.sub(r"[^A-Za-z0-9_-]+", "-", job_id)
        tab_id = f"job-{index}-{safe_job_id}"
        resource_id = f"{tab_id}-resource"
        link_id = f"{tab_id}-links"
        plot_ids.extend([resource_id, link_id])

        styles = build_operation_styles(job_events)
        resource_html = pio.to_html(
            make_resource_timeline(job_events, styles),
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            config=plot_config,
            div_id=resource_id,
        )
        include_plotlyjs = False
        link_html = pio.to_html(
            make_link_timeline(job_events, styles),
            include_plotlyjs=False,
            full_html=False,
            config=plot_config,
            div_id=link_id,
        )

        active_class = " active" if index == 0 else ""
        resource_hint = (
            "Combined lanes use job / rank / resource identity. "
            "Aggregated jobs retain one cohort lane and expand deviations."
            if job_id == "All jobs"
            else (
                "Gray GPU bars are synchronization waits. Hover shows job, "
                "rank, data replica, pipeline stage, checkpoint overlap, "
                "and timing."
            )
        )
        tab_buttons.append(
            f'<button class="job-tab{active_class}" '
            f"onclick=\"showJob('{tab_id}', this)\">"
            f"{html.escape(job_id)}</button>"
        )
        job_panels.append(
            f"""
    <div id="{tab_id}" class="job-view{active_class}">
      <section class="panel">
        <div class="panel-heading">
          <h2>{html.escape(job_id)} · resource occupancy</h2>
          <div class="zoom-controls">
            <button onclick="zoomAxis('{resource_id}', 'x', 0.65)">X zoom in</button>
            <button onclick="zoomAxis('{resource_id}', 'x', 1.55)">X zoom out</button>
            <button onclick="resetAxis('{resource_id}', 'x')">Reset X</button>
            <button onclick="zoomAxis('{resource_id}', 'y', 0.65)">Y zoom in</button>
            <button onclick="zoomAxis('{resource_id}', 'y', 1.55)">Y zoom out</button>
            <button onclick="resetAxis('{resource_id}', 'y')">Reset Y</button>
          </div>
        </div>
        <p class="hint">{resource_hint}</p>
        {resource_html}
      </section>

      <section class="panel">
        <div class="panel-heading">
          <h2>{html.escape(job_id)} · link occupancy</h2>
          <div class="zoom-controls">
            <button onclick="zoomAxis('{link_id}', 'x', 0.65)">X zoom in</button>
            <button onclick="zoomAxis('{link_id}', 'x', 1.55)">X zoom out</button>
            <button onclick="resetAxis('{link_id}', 'x')">Reset X</button>
            <button onclick="zoomAxis('{link_id}', 'y', 0.65)">Y zoom in</button>
            <button onclick="zoomAxis('{link_id}', 'y', 1.55)">Y zoom out</button>
            <button onclick="resetAxis('{link_id}', 'y')">Reset Y</button>
          </div>
        </div>
        <p class="hint">Pipeline, data-parallel collective, and object-store links are shown separately.</p>
        {link_html}
      </section>
    </div>
"""
        )

    tabs_html = "".join(tab_buttons)
    panels_html = "".join(job_panels)
    plot_ids_json = json.dumps(plot_ids)
    assumption = (
        "<strong>Configured job strategies:</strong> use the tabs to switch "
        "between data-parallel, pipeline-parallel, and hybrid jobs. "
        "Object-store uploads overlap training and slow the uploading "
        "rank's GPU."
        if configured_jobs
        else (
            "<strong>Contention model:</strong> independent CPU and GPU work "
            "on the same rank slows both sides. CPU work is 1.25× slower "
            "while GPU work is active; GPU work is 1.15× slower while CPU "
            "work is active."
        )
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Distributed Training Simulation V4</title>
  <style>
    body {{
      margin: 0;
      background: #f5f7fb;
      color: #1f2937;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin-bottom: 4px; }}
    .version-badge {{
      display: inline-block;
      margin-left: 10px;
      padding: 4px 9px;
      border-radius: 999px;
      background: #0f172a;
      color: white;
      font-size: 0.72rem;
      vertical-align: middle;
    }}
    .source {{ color: #64748b; margin-top: 0; }}
    .assumption {{
      padding: 12px 15px;
      border-left: 5px solid #E69F00;
      background: #fff7e6;
      border-radius: 8px;
      margin: 16px 0;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin: 24px 0;
    }}
    .card, .panel {{
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }}
    .card {{ padding: 18px; }}
    .card .label {{ color: #64748b; font-size: 0.9rem; }}
    .card .value {{ font-size: 1.55rem; font-weight: 700; margin-top: 4px; }}
    .panel {{ padding: 12px; margin-top: 20px; overflow: hidden; }}
    .panel-heading {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 10px 0;
    }}
    .panel-heading h2 {{ margin: 0; font-size: 1.05rem; }}
    .zoom-controls {{ display: flex; flex-wrap: wrap; gap: 7px; }}
    .zoom-controls button {{
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      color: #0f172a;
      border-radius: 7px;
      padding: 7px 10px;
      cursor: pointer;
      font-weight: 600;
    }}
    .zoom-controls button:hover {{ background: #e2e8f0; }}
    .hint {{ color: #64748b; font-size: 0.86rem; margin: 6px 10px 0; }}
    .job-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 22px 0 8px;
    }}
    .job-tab {{
      border: 1px solid #94a3b8;
      background: white;
      color: #334155;
      border-radius: 9px;
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 700;
    }}
    .job-tab.active {{
      color: white;
      background: #2563eb;
      border-color: #2563eb;
    }}
    .job-view {{ display: none; }}
    .job-view.active {{ display: block; }}
  </style>
</head>
<body>
  <main>
    <h1>Distributed training simulation <span class="version-badge">VISUALIZATION V4</span></h1>
    <p class="source">Generated from {html.escape(str(source_log))}</p>
    <div class="assumption">{assumption}</div>
    <section class="cards">{summary_cards(events)}</section>
    <nav class="job-tabs" aria-label="Job selector">{tabs_html}</nav>
    {panels_html}
  </main>

  <script>
    const initialRanges = {{}};

    function plotElement(plotId) {{
      return document.getElementById(plotId);
    }}

    function captureRanges(plotId) {{
      const graph = plotElement(plotId);
      if (!graph || !graph._fullLayout) return;
      if (!initialRanges[plotId]) initialRanges[plotId] = {{}};
      for (const axisName of ['x', 'y']) {{
        if (!initialRanges[plotId][axisName]) {{
          initialRanges[plotId][axisName] = graph._fullLayout[axisName + 'axis'].range.slice();
        }}
      }}
    }}

    function zoomAxis(plotId, axisName, factor) {{
      const graph = plotElement(plotId);
      if (!graph || !graph._fullLayout) return;
      captureRanges(plotId);
      const range = graph._fullLayout[axisName + 'axis'].range.slice();
      const center = (range[0] + range[1]) / 2;
      const halfSpan = Math.abs(range[1] - range[0]) * factor / 2;
      const nextRange = range[0] <= range[1]
        ? [center - halfSpan, center + halfSpan]
        : [center + halfSpan, center - halfSpan];
      const update = {{}};
      update[axisName + 'axis.range'] = nextRange;
      Plotly.relayout(graph, update);
    }}

    function resetAxis(plotId, axisName) {{
      const graph = plotElement(plotId);
      if (!graph) return;
      captureRanges(plotId);
      const update = {{}};
      update[axisName + 'axis.range'] = initialRanges[plotId][axisName];
      Plotly.relayout(graph, update);
    }}

    function showJob(tabId, button) {{
      document.querySelectorAll('.job-view').forEach((view) => {{
        view.classList.toggle('active', view.id === tabId);
      }});
      document.querySelectorAll('.job-tab').forEach((tab) => {{
        tab.classList.toggle('active', tab === button);
      }});
      window.setTimeout(() => {{
        const activeView = document.getElementById(tabId);
        activeView.querySelectorAll('.plotly-graph-div').forEach((graph) => {{
          Plotly.Plots.resize(graph);
          captureRanges(graph.id);
        }});
      }}, 0);
    }}

    window.addEventListener('load', () => {{
      window.setTimeout(() => {{
        for (const plotId of {plot_ids_json}) captureRanges(plotId);
      }}, 0);
    }});
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive Plotly dashboard from the simulation log."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("results/simulation_log.jsonl"),
        help="Plain or gzip-compressed JSONL event log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/simulation_visualization.html"),
        help="Output standalone HTML file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.log)
    write_dashboard(events, output_path=args.output, source_log=args.log)
    print(f"Visualization written to {args.output}")


if __name__ == "__main__":
    main()
