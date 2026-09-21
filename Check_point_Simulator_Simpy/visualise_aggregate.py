"""Create a service-oriented aggregate dashboard from simulator JSONL traces.

Unlike visualise.py, this module never creates one lane per rank or node. It
streams the trace, groups jobs into service classes, and emits a bounded HTML
dashboard suitable for very wide scenario runs.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import html
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


CATEGORY_COLORS = {
    "Training": "#2c7be5",
    "Checkpoint": "#f59f00",
    "Communication": "#6f42c1",
    "Recovery": "#e03131",
    "Failure": "#c92a2a",
    "Controller": "#0ca678",
    "Lifecycle": "#495057",
    "Other": "#868e96",
}

OPERATION_LABELS = {
    "training_iterations": "Training + all-reduce envelope",
    "checkpoint_stage_gpu_to_dram": "GPU → DRAM checkpoint capture",
    "checkpoint_dram_to_crossjob_peers_chunk": "DRAM → cross-job peer SSD",
    "checkpoint_donor_ssd_to_l3_drain_chunk": "Peer SSD → object-store L3 drain",
    "checkpoint_dram_to_local_ssd_chunk": "DRAM → local-SSD fallback",
    "checkpoint_crossjob_peers_to_dram_recovery_chunk": "Peer SSD → DRAM recovery",
    "initial_state_to_dram_chunk": "Initial state → DRAM recovery",
    "dram_to_gpu_restore": "DRAM → GPU restore tail",
    "job_pending": "Waiting for configured arrival",
    "job_arrival": "Job arrival",
    "job_departure": "Job departure",
    "job_idle_checkpoint": "Training complete; idle checkpoint donor",
}

STRATEGY_LABELS = {
    "crossjob_peer": "Cross-job peer checkpointing",
    "gemini": "GEMINI",
    "checkfreq": "CheckFreq",
    "checknrun": "Check-N-Run",
    "megascale": "MegaScale",
    "jit": "Just-In-Time checkpointing",
}

BASELINE_PRESENTATIONS = {
    "gemini": {
        "label": "GEMINI (SOSP'23) baseline",
        "description": (
            "Periodic GPU-to-DRAM capture is replicated into peer DRAM inside "
            "the same job, with a much less frequent remote-persistent-store "
            "backstop. It does not use cross-job donor SSD striping."
        ),
        "pipeline": ("GPU", "owner DRAM", "in-job peer DRAM", "remote store"),
    },
    "checkfreq": {
        "label": "CheckFreq (FAST'21) baseline",
        "description": (
            "A profiled cadence bounds visible capture overhead, then persists "
            "asynchronously to local SSD. It does not use cross-job donors."
        ),
        "pipeline": ("GPU", "owner DRAM", "local SSD"),
    },
    "checknrun": {
        "label": "Check-N-Run (NSDI'22) baseline",
        "description": (
            "Differential and quantized checkpoints are written to the object "
            "store at a fixed cadence using 0.1x modeled write bytes."
        ),
        "pipeline": ("GPU", "owner DRAM", "object store (compressed)"),
    },
    "megascale": {
        "label": "MegaScale (NSDI'24) baseline",
        "description": (
            "Full-size checkpoints are staged in DRAM and flushed "
            "asynchronously to the object store at a fixed cadence."
        ),
        "pipeline": ("GPU", "owner DRAM", "object store"),
    },
    "jit": {
        "label": "Just-In-Time (EuroSys'24) baseline",
        "description": (
            "Periodic checkpointing is disabled. On a single-node failure, "
            "state is drained from a surviving data-parallel replica."
        ),
        "pipeline": ("surviving replica DRAM", "replacement node", "GPU"),
    },
}


def operation_label(operation: str) -> str:
    return OPERATION_LABELS.get(operation, operation.replace("_", " "))


def iter_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a plain or gzip-compressed JSONL trace."""

    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}"
                )
            yield event


def event_service(event: dict[str, Any]) -> str:
    job_id = event.get("job_id")
    if job_id not in (None, ""):
        return str(job_id)
    node = event.get("physical_node") or event.get("node")
    return endpoint_service(str(node)) if node else "simulation"


def service_class(service: str) -> str:
    if service in {"object-store", "simulation", "unknown"}:
        return service
    result = re.sub(r"\d+$", "", service).rstrip("-_")
    return result or service


def endpoint_service(endpoint: str) -> str:
    value = endpoint.strip()
    if not value:
        return "unknown"
    value = value.split("/", 1)[0]
    if value == "__store__" or "object-store" in value.lower():
        return "object-store"
    for separator in ("-cohort-", "-rank-"):
        if separator in value:
            return value.split(separator, 1)[0]
    if re.fullmatch(r"node-\d+", value):
        return "simulation"
    return value


def endpoint_services(value: Any, fallback: str) -> list[str]:
    if value in (None, ""):
        return [fallback]
    services = {
        endpoint_service(part)
        for part in str(value).split("+")
        if part.strip()
    }
    return sorted(services or {fallback})


def event_route_shares(
    event: dict[str, Any], fallback: str, data_gb: float
) -> list[tuple[tuple[str, str], float]]:
    """Return class-level byte routes while retaining cross-job self-class edges."""

    details = event.get("details") or {}
    source_services = endpoint_services(event.get("source"), fallback)
    destination_services = endpoint_services(event.get("destination"), fallback)

    # Peer cohort events record the exact number of represented shards assigned
    # to each donor. Use those weights instead of equally splitting donor jobs.
    shards_by_donor = details.get("shards_by_donor") or {}
    if shards_by_donor and len(source_services) == 1:
        donor_weights: Counter = Counter()
        for donor, weight in shards_by_donor.items():
            donor_weights[endpoint_service(str(donor))] += _float_or_zero(weight)
        total_weight = sum(donor_weights.values())
        if total_weight > 0:
            source = source_services[0]
            return [
                (
                    (service_class(source), service_class(destination)),
                    data_gb * weight / total_weight,
                )
                for destination, weight in donor_weights.items()
                if source != destination
            ]

    # Donor-drain events carry exact per-donor piece sizes.
    drain_pieces = details.get("drain_pieces") or {}
    if drain_pieces and len(destination_services) == 1:
        destination = destination_services[0]
        source_gb: Counter = Counter()
        for source, piece_gb in drain_pieces.items():
            source_gb[endpoint_service(str(source))] += _float_or_zero(piece_gb)
        return [
            (
                (service_class(source), service_class(destination)),
                piece_gb,
            )
            for source, piece_gb in source_gb.items()
            if source != destination
        ]

    raw_pairs = [
        (source, destination)
        for source in source_services
        for destination in destination_services
        if source != destination
    ]
    if not raw_pairs:
        return []
    share = data_gb / len(raw_pairs)
    return [
        ((service_class(source), service_class(destination)), share)
        for source, destination in raw_pairs
    ]


def logical_weight(event: dict[str, Any]) -> float:
    details = event.get("details") or {}
    try:
        value = float(details.get("represents", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return value if math.isfinite(value) and value > 0 else 1.0


def event_duration(event: dict[str, Any]) -> float:
    try:
        start = float(event.get("start", 0.0))
        end = float(event.get("end", start))
        value = float(event.get("duration", end - start))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, value) if math.isfinite(value) else 0.0


def category_group(event: dict[str, Any]) -> str:
    category = str(event.get("category") or "")
    operation = str(event.get("operation") or "").lower()
    combined = f"{category} {operation}".lower()
    if category.lower() == "lifecycle" or operation.startswith("job_"):
        return "Lifecycle"
    if "failure" in combined or event.get("failure_type"):
        return "Failure"
    if any(word in combined for word in ("recovery", "restore", "restart")):
        return "Recovery"
    if "checkpoint" in combined or "snapshot" in combined:
        return "Checkpoint"
    if any(
        word in combined
        for word in (
            "communication",
            "collective",
            "all_reduce",
            "all-reduce",
            "transfer",
            "network",
            "synchronization",
        )
    ):
        return "Communication"
    if any(word in combined for word in ("training", "computation", "compute")):
        return "Training"
    if "controller" in combined:
        return "Controller"
    return "Other"


@dataclass
class Reservoir:
    limit: int = 2048
    seed: int = 17
    seen: int = 0
    values: list[float] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.limit:
            self.values.append(value)
            return
        index = self._rng.randrange(self.seen)
        if index < self.limit:
            self.values[index] = value

    def percentile(self, percentile: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        position = (len(ordered) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass
class OperationStat:
    count: int = 0
    logical_count: float = 0.0
    span_seconds: float = 0.0
    node_seconds: float = 0.0
    base_node_seconds: float = 0.0
    data_gb: float = 0.0
    maximum: float = 0.0
    durations: Reservoir = field(default_factory=Reservoir)

    def add(
        self,
        duration: float,
        weight: float,
        data_gb: float,
        base_duration: float,
    ) -> None:
        self.count += 1
        self.logical_count += weight
        self.span_seconds += duration
        self.node_seconds += duration * weight
        self.base_node_seconds += base_duration * weight
        self.data_gb += data_gb
        self.maximum = max(self.maximum, duration)
        self.durations.add(duration)


@dataclass
class TraceSummary:
    event_count: int = 0
    logical_event_count: float = 0.0
    min_time: float = math.inf
    max_time: float = -math.inf
    total_data_gb: float = 0.0
    failure_count: float = 0.0
    services: set[str] = field(default_factory=set)
    service_classes: set[str] = field(default_factory=set)
    node_weights: dict[tuple[str, str], float] = field(default_factory=dict)
    service_events: Counter = field(default_factory=Counter)
    service_node_seconds: Counter = field(default_factory=Counter)
    service_data_gb: Counter = field(default_factory=Counter)
    service_checkpoint_gb: Counter = field(default_factory=Counter)
    service_local_ssd_write_gb: Counter = field(default_factory=Counter)
    service_local_ssd_write_events: Counter = field(default_factory=Counter)
    service_local_ssd_write_mode_gb: Counter = field(default_factory=Counter)
    service_local_ssd_latest_iteration: dict[str, int] = field(default_factory=dict)
    service_category_node_seconds: Counter = field(default_factory=Counter)
    service_recovery_tiers: Counter = field(default_factory=Counter)
    service_start: dict[str, float] = field(default_factory=dict)
    service_arrival: dict[str, float] = field(default_factory=dict)
    service_departure: dict[str, float] = field(default_factory=dict)
    service_training_end: dict[str, float] = field(default_factory=dict)
    class_jobs: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    class_events: Counter = field(default_factory=Counter)
    class_node_seconds: Counter = field(default_factory=Counter)
    class_data_gb: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)
    operations: dict[tuple[str, str], OperationStat] = field(
        default_factory=dict
    )
    routes: Counter = field(default_factory=Counter)
    route_events: Counter = field(default_factory=Counter)
    checkpoint_paths: Counter = field(default_factory=Counter)
    checkpoint_strategies: Counter = field(default_factory=Counter)
    recovery_tiers: Counter = field(default_factory=Counter)
    top_spans: list[tuple[float, int, dict[str, Any]]] = field(
        default_factory=list
    )

    @property
    def duration(self) -> float:
        if not math.isfinite(self.min_time) or not math.isfinite(self.max_time):
            return 0.0
        return max(0.0, self.max_time - self.min_time)

    @property
    def physical_nodes(self) -> float:
        return sum(self.node_weights.values())


@dataclass
class RunContext:
    """Optional scenario/result metadata that is not encoded in JSONL spans."""

    scenario_path: Path | None = None
    scenario: dict[str, Any] = field(default_factory=dict)
    result_path: Path | None = None
    result: dict[str, Any] = field(default_factory=dict)
    arm: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class CheckpointPolicyPresentation:
    """Human-readable configured policy, kept separate from trace observation."""

    kind: str
    label: str
    description: str
    pipeline: tuple[str, ...] = ()


def _arm_config(context: RunContext) -> dict[str, Any]:
    configured = (context.scenario.get("arms") or {}).get(context.arm or "", {})
    return dict(configured) if isinstance(configured, dict) else {}


def _result_policy_metadata(context: RunContext) -> dict[str, Any]:
    """Normalize optional policy metadata embedded in a selected result row."""

    nested = context.result.get("checkpoint_policy") or {}
    if not isinstance(nested, dict):
        nested = {}
    by_class = (
        context.result.get("policy_by_class")
        or nested.get("by_class")
        or nested.get("classes")
        or {}
    )
    flags = context.result.get("policy_flags") or nested.get("flags") or {}
    return {
        "kind": context.result.get("policy_kind") or nested.get("kind"),
        "name": (
            context.result.get("policy_name")
            or nested.get("name")
            or context.arm
        ),
        "by_class": by_class if isinstance(by_class, dict) else {},
        "flags": flags if isinstance(flags, dict) else {},
    }


def _observed_strategy_label(summary: TraceSummary) -> str:
    names = [name for name, _ in summary.checkpoint_strategies.most_common()]
    if not names:
        return "No checkpoint strategy observed"
    return " + ".join(
        STRATEGY_LABELS.get(name, name.replace("_", " ").title())
        for name in names
    )


def configured_checkpoint_policy(
    context: RunContext,
) -> CheckpointPolicyPresentation:
    """Resolve the selected YAML/result arm without consulting trace operations."""

    arm_cfg = _arm_config(context)
    baseline = str(arm_cfg.get("baseline") or "")
    if baseline:
        presentation = BASELINE_PRESENTATIONS.get(baseline)
        if presentation is None:
            return CheckpointPolicyPresentation(
                kind="baseline",
                label=f"Named baseline: {baseline}",
                description="The selected scenario arm dispatches to a named baseline.",
            )
        return CheckpointPolicyPresentation(
            kind=f"baseline:{baseline}",
            label=str(presentation["label"]),
            description=str(presentation["description"]),
            pipeline=tuple(presentation["pipeline"]),
        )

    if arm_cfg.get("ideal"):
        return CheckpointPolicyPresentation(
            kind="ideal",
            label="Ideal comparator (checkpointing disabled)",
            description=(
                "The selected arm disables periodic checkpoints, failures, and "
                "preemption so the run contains compute and modeled all-reduce only."
            ),
            pipeline=("training only",),
        )

    if arm_cfg.get("cadence_l2_s") is not None:
        l2 = float(arm_cfg["cadence_l2_s"])
        l1 = float(arm_cfg.get("cadence_l1_s", l2))
        l3 = float(arm_cfg.get("cadence_l3_s", l2))
        placement = "cross-job peer SSD" if arm_cfg.get("kpeers", True) else "local SSD"
        return CheckpointPolicyPresentation(
            kind="naive-tiered",
            label="Hand-set tiered cadence",
            description=(
                f"The selected arm fixes L1 capture at {l1:g}s, L2 persistence "
                f"at {l2:g}s, and L3 durability at {l3:g}s. These values are "
                "operator supplied, not GP-solved."
            ),
            pipeline=("GPU", "owner DRAM", placement, "object store"),
        )

    if arm_cfg.get("cadence_s") is not None and not baseline:
        cadence = float(arm_cfg["cadence_s"])
        placement = "cross-job peer SSD" if arm_cfg.get("kpeers", True) else "local SSD"
        return CheckpointPolicyPresentation(
            kind="naive-all-tiers",
            label=f"Naive fixed cadence ({cadence:g}s, all tiers)",
            description=(
                "One hand-set wall-clock interval drives DRAM capture, L2 "
                "persistence, and the L3 backstop for every workload class."
            ),
            pipeline=("GPU", "owner DRAM", placement, "object store"),
        )

    if arm_cfg.get("checkpoint_every") is not None:
        every = int(arm_cfg["checkpoint_every"])
        placement = "cross-job peer SSD" if arm_cfg.get("kpeers", True) else "local SSD"
        return CheckpointPolicyPresentation(
            kind="fixed-iterations",
            label=f"Fixed checkpoint cadence (every {every} iterations)",
            description=(
                "The arm overrides every workload class with the same "
                "iteration-count checkpoint interval."
            ),
            pipeline=("GPU", "owner DRAM", placement, "durability backstop"),
        )

    if arm_cfg:
        if arm_cfg.get("store_mode"):
            return CheckpointPolicyPresentation(
                kind="store",
                label="Direct object-store checkpointing",
                description="Every persistent checkpoint is routed to the shared store.",
                pipeline=("GPU", "owner DRAM", "object store"),
            )
        if not arm_cfg.get("kpeers", True):
            return CheckpointPolicyPresentation(
                kind="local",
                label="Local-only tiered checkpointing",
                description=(
                    "Cross-job placement is disabled, so persistent checkpoints "
                    "stay on owner-local SSD with the configured durability backstop."
                ),
                pipeline=("GPU", "owner DRAM", "local SSD", "object store"),
            )
        return CheckpointPolicyPresentation(
            kind="crossjob",
            label="Cross-job peer checkpointing",
            description=(
                "Checkpoints are captured in owner DRAM, striped onto SSDs owned "
                "by other jobs, and protected by the selected L3 backstop."
            ),
            pipeline=("GPU", "owner DRAM", "cross-job peer SSD", "object store"),
        )

    result_policy = _result_policy_metadata(context)
    if context.result and context.arm:
        kind = str(result_policy.get("kind") or "external")
        name = str(result_policy.get("name") or context.arm)
        has_decisions = bool(result_policy.get("by_class"))
        detail = (
            "Per-class enacted decisions are embedded in the result row."
            if has_decisions
            else (
                "The result identifies this arm, but its per-class cadence and "
                "peer decisions were not embedded; scenario defaults are not "
                "shown as if they were enacted."
            )
        )
        label = (
            f"GP-solved policy: {name}"
            if kind.lower() == "gp"
            else f"Result-selected external policy: {name}"
        )
        return CheckpointPolicyPresentation(
            kind=f"external:{kind}",
            label=label,
            description=detail,
        )

    return CheckpointPolicyPresentation(
        kind="unknown",
        label="Configured checkpoint policy unavailable",
        description=(
            "Pass --scenario and select an arm to show configured policy intent. "
            "Observed trace strategy remains available separately."
        ),
    )


def load_run_context(
    *,
    scenario_path: Path | None = None,
    result_path: Path | None = None,
    arm: str | None = None,
    seed: int | None = None,
) -> RunContext:
    scenario: dict[str, Any] = {}
    if scenario_path is not None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Scenario metadata requires PyYAML") from exc
        loaded = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Scenario root must be a mapping: {scenario_path}")
        scenario = loaded

    result: dict[str, Any] = {}
    if result_path is not None:
        loaded_result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_result, dict):
            raise ValueError(f"Result root must be an object: {result_path}")
        rows = loaded_result.get("rows") or []
        matching = [
            row
            for row in rows
            if (arm is None or str(row.get("arm")) == arm)
            and (seed is None or int(row.get("seed", -1)) == seed)
        ]
        if matching:
            result = matching[0]
        elif len(rows) == 1:
            result = rows[0]
        if arm is None and result.get("arm") is not None:
            arm = str(result["arm"])
        if seed is None and result.get("seed") is not None:
            seed = int(result["seed"])

    return RunContext(
        scenario_path=scenario_path,
        scenario=scenario,
        result_path=result_path,
        result=result,
        arm=arm,
        seed=seed,
    )


def _float_or_zero(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result > 0 else 0.0


def summarize_trace(path: Path, *, top_spans: int = 500) -> TraceSummary:
    summary = TraceSummary()
    for sequence, event in enumerate(iter_events(path)):
        try:
            start = float(event.get("start", 0.0))
            end = float(event.get("end", start))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        duration = event_duration(event)
        weight = logical_weight(event)
        data_gb = _float_or_zero(event.get("data_gb"))
        service = event_service(event)
        class_name = service_class(service)
        category = category_group(event)
        operation = str(event.get("operation") or "unknown")

        summary.event_count += 1
        summary.logical_event_count += weight
        summary.min_time = min(summary.min_time, start)
        summary.max_time = max(summary.max_time, end)
        summary.total_data_gb += data_gb
        summary.services.add(service)
        summary.service_classes.add(class_name)
        summary.class_jobs[class_name].add(service)
        summary.service_events[service] += 1
        summary.service_node_seconds[service] += duration * weight
        summary.service_data_gb[service] += data_gb
        summary.service_category_node_seconds[(service, category)] += duration * weight
        if category == "Checkpoint":
            summary.service_checkpoint_gb[service] += data_gb
        summary.class_events[class_name] += 1
        summary.class_node_seconds[class_name] += duration * weight
        summary.class_data_gb[class_name] += data_gb
        summary.categories[category] += weight

        node = str(
            event.get("physical_node")
            or event.get("node")
            or f"rank-{event.get('rank', 'unknown')}"
        )
        if category != "Lifecycle":
            node_key = (service, node)
            summary.node_weights[node_key] = max(
                summary.node_weights.get(node_key, 0.0), weight
            )

        operation_key = (category, operation)
        if operation_key not in summary.operations:
            seed = sum(ord(char) for char in f"{category}:{operation}")
            summary.operations[operation_key] = OperationStat(
                durations=Reservoir(seed=seed)
            )
        details = event.get("details") or {}
        base_duration = _float_or_zero(details.get("base_duration")) or duration
        summary.operations[operation_key].add(
            duration, weight, data_gb, base_duration
        )

        if operation == "job_arrival":
            summary.service_arrival[service] = min(
                summary.service_arrival.get(service, start), start
            )
        elif operation in {"job_departure", "job_idle_checkpoint"}:
            summary.service_departure[service] = max(
                summary.service_departure.get(service, end), end
            )
        if operation != "job_pending":
            summary.service_start[service] = min(
                summary.service_start.get(service, start), start
            )
        if category == "Training":
            summary.service_training_end[service] = max(
                summary.service_training_end.get(service, end), end
            )

        if category == "Failure":
            summary.failure_count += weight

        strategy_name = details.get("checkpoint_strategy")
        if strategy_name:
            summary.checkpoint_strategies[str(strategy_name)] += weight
        path_name = details.get("path")
        if path_name:
            summary.checkpoint_paths[str(path_name)] += weight
        if category == "Checkpoint" and operation == "checkpoint_dram_to_local_ssd_chunk":
            # data_gb is already the physical/cohort aggregate. Multiplying it by
            # `represents` would double-count wide cohort traces.
            mode = "fallback" if path_name == "local_fallback" else "planned"
            summary.service_local_ssd_write_gb[service] += data_gb
            summary.service_local_ssd_write_events[service] += 1
            summary.service_local_ssd_write_mode_gb[(service, mode)] += data_gb
            try:
                checkpoint_iteration = int(event.get("iteration"))
            except (TypeError, ValueError):
                checkpoint_iteration = None
            if checkpoint_iteration is not None:
                summary.service_local_ssd_latest_iteration[service] = max(
                    summary.service_local_ssd_latest_iteration.get(service, -1),
                    checkpoint_iteration,
                )
        recovery_tier = details.get("checkpoint_source_tier")
        if recovery_tier and operation == "dram_to_gpu_restore":
            summary.recovery_tiers[str(recovery_tier)] += weight
            summary.service_recovery_tiers[(service, str(recovery_tier))] += 1

        if data_gb > 0 and (event.get("source") or event.get("destination")):
            touched_routes: set[tuple[str, str]] = set()
            for pair, share in event_route_shares(event, service, data_gb):
                summary.routes[pair] += share
                touched_routes.add(pair)
            for pair in touched_routes:
                summary.route_events[pair] += 1

        span = {
            "start": start,
            "end": end,
            "duration": duration,
            "service": service,
            "class": class_name,
            "category": category,
            "operation": operation,
            "node": node,
            "iteration": event.get("iteration"),
            "rank": event.get("rank"),
            "data_gb": data_gb,
            "status": details.get("status") or event.get("failure_type") or "ok",
        }
        item = (duration, sequence, span)
        if len(summary.top_spans) < max(1, top_spans):
            heapq.heappush(summary.top_spans, item)
        elif duration > summary.top_spans[0][0]:
            heapq.heapreplace(summary.top_spans, item)

    return summary


@dataclass
class Activity:
    centers: list[float]
    category_values: dict[str, list[float]]
    class_values: dict[str, list[float]]
    failure_values: list[float]


def aggregate_activity(
    path: Path,
    summary: TraceSummary,
    *,
    buckets: int = 240,
) -> Activity:
    bucket_count = max(20, min(int(buckets), 1000))
    duration = max(summary.duration, 1e-9)
    width = duration / bucket_count
    centers = [summary.min_time + (index + 0.5) * width for index in range(bucket_count)]
    category_values = {
        category: [0.0] * bucket_count for category in CATEGORY_COLORS
    }
    class_values = {
        class_name: [0.0] * bucket_count
        for class_name in summary.service_classes
    }
    failure_values = [0.0] * bucket_count

    for event in iter_events(path):
        try:
            start = float(event.get("start", 0.0))
            end = float(event.get("end", start))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        weight = logical_weight(event)
        category = category_group(event)
        class_name = service_class(event_service(event))
        if end <= start:
            if category == "Failure":
                index = min(
                    bucket_count - 1,
                    max(0, int((start - summary.min_time) / width)),
                )
                failure_values[index] += weight
            continue

        first = min(
            bucket_count - 1,
            max(0, int((start - summary.min_time) / width)),
        )
        last = min(
            bucket_count - 1,
            max(0, int((end - summary.min_time - 1e-12) / width)),
        )
        for index in range(first, last + 1):
            left = summary.min_time + index * width
            right = left + width
            overlap = max(0.0, min(end, right) - max(start, left))
            occupancy = weight * overlap / width
            category_values[category][index] += occupancy
            class_values[class_name][index] += occupancy
        if category == "Failure":
            failure_values[first] += weight

    return Activity(
        centers=centers,
        category_values=category_values,
        class_values=class_values,
        failure_values=failure_values,
    )


def _base_layout(title: str, height: int = 470) -> dict[str, Any]:
    return {
        "title": {"text": title, "x": 0.02, "xanchor": "left"},
        "height": height,
        "margin": {"l": 70, "r": 35, "t": 65, "b": 55},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#f8fafc",
        "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#243b53"},
        "hoverlabel": {"font": {"family": "Consolas, monospace"}},
    }


def activity_figure(activity: Activity) -> go.Figure:
    figure = go.Figure()
    for category, color in CATEGORY_COLORS.items():
        values = activity.category_values.get(category, [])
        if not any(value > 0 for value in values):
            continue
        figure.add_trace(
            go.Scatter(
                x=activity.centers,
                y=values,
                name=category,
                mode="lines",
                stackgroup="activity",
                line={"width": 0.8, "color": color},
                hovertemplate=(
                    "%{x:.1f}s<br>%{y:,.1f} active node-equivalents"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        **_base_layout("Recorded span occupancy by category (overlap allowed)"),
        xaxis_title="Simulated time (seconds)",
        yaxis_title="Average weighted concurrent spans",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    return figure


def heatmap_figure(activity: Activity, summary: TraceSummary) -> go.Figure:
    classes = sorted(
        summary.service_classes,
        key=lambda value: summary.class_node_seconds[value],
        reverse=True,
    )
    z_values = [activity.class_values[name] for name in classes]
    figure = go.Figure(
        go.Heatmap(
            x=activity.centers,
            y=classes,
            z=z_values,
            colorscale=[
                [0.0, "#f8fafc"],
                [0.2, "#c5f6fa"],
                [0.5, "#38d9a9"],
                [1.0, "#087f5b"],
            ],
            colorbar={"title": "active<br>nodes"},
            hovertemplate=(
                "%{y}<br>%{x:.1f}s<br>%{z:,.1f} active node-equivalents<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_base_layout("Service-class span-occupancy heatmap", max(380, 65 + 42 * len(classes))),
        xaxis_title="Simulated time (seconds)",
        yaxis_title="Service class",
    )
    return figure


def allocated_nodes_figure(activity: Activity, summary: TraceSummary) -> go.Figure:
    """Show fixed job nodes transitioning from training to idle donor duty."""

    service_nodes: Counter = Counter()
    for (service, _node), weight in summary.node_weights.items():
        service_nodes[service] += weight
    timeline = list(activity.centers)
    if not timeline or timeline[-1] < summary.max_time:
        timeline.append(summary.max_time)
    values = {
        class_name: [0.0] * len(timeline)
        for class_name in summary.service_classes
    }
    for service, training_end in summary.service_training_end.items():
        class_name = service_class(service)
        start = summary.service_arrival.get(
            service, summary.service_start.get(service, summary.min_time)
        )
        end = summary.service_departure.get(service, training_end)
        for index, center in enumerate(timeline):
            if start <= center < end:
                values[class_name][index] += service_nodes[service]
    idle_values = [0.0] * len(timeline)
    for service, idle_start in summary.service_departure.items():
        for index, center in enumerate(timeline):
            if center >= idle_start:
                idle_values[index] += service_nodes[service]

    classes = sorted(
        values,
        key=lambda name: max(values[name], default=0.0),
        reverse=True,
    )
    figure = go.Figure()
    palette = ["#1971c2", "#0ca678", "#f59f00", "#6f42c1", "#e03131", "#748ffc"]
    for index, class_name in enumerate(classes):
        if not any(values[class_name]):
            continue
        figure.add_trace(
            go.Scatter(
                x=timeline,
                y=values[class_name],
                name=class_name,
                mode="lines",
                stackgroup="allocated",
                line={
                    "width": 0.8,
                    "color": palette[index % len(palette)],
                    "shape": "hv",
                },
                hovertemplate=(
                    "%{x:.1f}s<br>%{y:,.0f} fixed nodes training in this class"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    if any(idle_values):
        figure.add_trace(
            go.Scatter(
                x=timeline,
                y=idle_values,
                name="Idle checkpoint donors",
                mode="lines",
                stackgroup="allocated",
                line={"width": 0.8, "color": "#6741D9", "shape": "hv"},
                hovertemplate=(
                    "%{x:.1f}s<br>%{y:,.0f} completed-job nodes available "
                    "only for checkpointing<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        **_base_layout("Fixed job nodes: training and idle checkpoint donors"),
        xaxis_title="Simulated time (seconds)",
        yaxis_title="Modeled nodes by current role",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    return figure


def completion_figure(summary: TraceSummary) -> go.Figure:
    """Summarize when each class releases nodes as its jobs finish."""

    service_nodes: Counter = Counter()
    for (service, _node), weight in summary.node_weights.items():
        service_nodes[service] += weight
    rows = []
    for class_name in summary.service_classes:
        services = sorted(
            service
            for service in summary.class_jobs[class_name]
            if service in summary.service_training_end
        )
        if not services:
            continue
        ends = [
            summary.service_departure.get(
                service, summary.service_training_end[service]
            )
            for service in services
        ]
        rows.append(
            (
                class_name,
                len(services),
                sum(service_nodes[service] for service in services),
                min(ends),
                statistics.median(ends),
                max(ends),
            )
        )
    rows.sort(key=lambda row: row[-1])
    figure = go.Figure(
        go.Table(
            header={
                "values": [
                    "Class",
                    "Jobs",
                    "Nodes becoming idle donors",
                    "First finish (s)",
                    "Median finish (s)",
                    "Last finish (s)",
                ],
                "fill_color": "#e7f5ff",
                "align": "left",
                "font": {"color": "#102a43", "size": 12},
                "height": 30,
            },
            cells={
                "values": list(zip(*rows)) if rows else [[]] * 6,
                "format": [None, ",d", ",.0f", ",.1f", ",.1f", ",.1f"],
                "fill_color": "#ffffff",
                "align": "left",
                "height": 27,
            },
        )
    )
    figure.update_layout(
        **_base_layout(
            "When fixed training allocations become idle checkpoint donors",
            max(320, 160 + 28 * len(rows)),
        )
    )
    return figure


def operation_work_figure(summary: TraceSummary) -> go.Figure:
    """Report additive node-time by operation, separating nominal and excess work."""

    items = sorted(
        summary.operations.items(),
        key=lambda item: item[1].node_seconds,
        reverse=True,
    )[:20]
    items.reverse()
    labels = [
        f"{category} · {operation_label(operation)}"
        for (category, operation), _ in items
    ]
    base_hours = [
        min(stat.base_node_seconds, stat.node_seconds) / 3600.0
        for _, stat in items
    ]
    excess_hours = [
        max(0.0, stat.node_seconds - stat.base_node_seconds) / 3600.0
        for _, stat in items
    ]
    total = sum(stat.node_seconds for stat in summary.operations.values())
    custom = [
        [
            stat.span_seconds,
            stat.node_seconds,
            100.0 * stat.node_seconds / total if total else 0.0,
            stat.count,
            stat.logical_count,
            stat.data_gb,
            stat.durations.percentile(0.50),
            stat.durations.percentile(0.95),
        ]
        for _, stat in items
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=base_hours,
            y=labels,
            orientation="h",
            name="Nominal work",
            marker_color="#1971c2",
            customdata=custom,
            hovertemplate=(
                "%{y}<br>nominal %{x:,.2f} node-hours"
                "<br>recorded span-seconds %{customdata[0]:,.2f}"
                "<br>total node-seconds %{customdata[1]:,.2f}"
                "<br>%{customdata[2]:.2f}% of all node-time"
                "<br>records %{customdata[3]:,}"
                "<br>logical spans %{customdata[4]:,.0f}"
                "<br>data %{customdata[5]:,.2f} GB"
                "<br>p50 / p95 %{customdata[6]:.4g}s / %{customdata[7]:.4g}s"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            x=excess_hours,
            y=labels,
            orientation="h",
            name="Contention / slowdown excess",
            marker_color="#f59f00",
            customdata=custom,
            hovertemplate=(
                "%{y}<br>excess %{x:,.2f} node-hours"
                "<br>total node-seconds %{customdata[1]:,.2f}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_base_layout("Time spent on each operation", max(520, 180 + 35 * len(items))),
        xaxis_title="Aggregate node-hours (duration × represented nodes)",
        yaxis_title="",
        barmode="stack",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def topology_figure(summary: TraceSummary) -> go.Figure:
    if not summary.routes:
        figure = go.Figure()
        figure.add_annotation(
            text="No cross-service byte transfers were found in this trace.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
        figure.update_layout(**_base_layout("Service dependency map", 370))
        return figure

    route_items = sorted(summary.routes.items(), key=lambda item: item[1], reverse=True)
    base_names = sorted({name for pair in summary.routes for name in pair})
    same_class_targets = [
        f"__same_class__:{source}"
        for (source, destination), _ in route_items
        if source == destination
    ]
    node_keys = base_names + same_class_targets
    indices = {name: index for index, name in enumerate(node_keys)}
    labels = [
        (
            f"{key.split(':', 1)[1]} (other jobs)"
            if key.startswith("__same_class__:")
            else key
        )
        for key in node_keys
    ]
    colors = [
        (
            "#6f42c1"
            if key.startswith("__same_class__:")
            else "#f08c00"
            if key == "object-store"
            else "#1971c2"
        )
        for key in node_keys
    ]
    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "label": labels,
                "color": colors,
                "pad": 18,
                "thickness": 18,
                "line": {"color": "#d9e2ec", "width": 1},
            },
            link={
                "source": [indices[pair[0]] for pair, _ in route_items],
                "target": [
                    indices[
                        f"__same_class__:{pair[1]}"
                        if pair[0] == pair[1]
                        else pair[1]
                    ]
                    for pair, _ in route_items
                ],
                "value": [value for _, value in route_items],
                "customdata": [
                    [
                        summary.route_events[pair],
                        (
                            "same class, different jobs"
                            if pair[0] == pair[1]
                            else "cross-class route"
                        ),
                    ]
                    for pair, _ in route_items
                ],
                "hovertemplate": (
                    "%{source.label} → %{target.label}<br>"
                    "%{value:,.1f} GB<br>%{customdata[0]:,} transfer records"
                    "<br>%{customdata[1]}<extra></extra>"
                ),
                "color": [
                    (
                        "rgba(111, 66, 193, 0.48)"
                        if pair[0] == pair[1]
                        else "rgba(44, 123, 229, 0.28)"
                    )
                    for pair, _ in route_items
                ],
            },
        )
    )
    figure.update_layout(**_base_layout("Service dependency map by transferred bytes", 500))
    return figure


def route_figure(summary: TraceSummary, *, limit: int = 20) -> go.Figure:
    """Rank cross-service routes without the visual ambiguity of Sankey cycles."""

    items = sorted(summary.routes.items(), key=lambda item: item[1], reverse=True)[
        : max(1, limit)
    ]
    if not items:
        figure = go.Figure()
        figure.add_annotation(
            text="No cross-service byte transfers were found in this trace.",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        figure.update_layout(**_base_layout("Largest cross-service routes", 360))
        return figure

    total_gb = sum(summary.routes.values())
    items.reverse()
    labels = [f"{source} → {destination}" for (source, destination), _ in items]
    values = [value for _, value in items]
    customdata = [
        [
            100.0 * value / total_gb if total_gb else 0.0,
            summary.route_events[pair],
            value / summary.route_events[pair]
            if summary.route_events[pair]
            else 0.0,
            "same class, different jobs" if pair[0] == pair[1] else "cross class",
        ]
        for pair, value in items
    ]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={
                "color": [
                    "#6f42c1" if pair[0] == pair[1] else "#1971c2"
                    for pair, _ in items
                ]
            },
            customdata=customdata,
            text=[f"{value:,.1f} GB" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>%{x:,.2f} GB"
                "<br>%{customdata[0]:.2f}% of cross-service bytes"
                "<br>%{customdata[1]:,} transfer records"
                "<br>%{customdata[2]:,.3f} GB/record"
                "<br>%{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_base_layout(
            "Largest cross-service routes (ranked)",
            max(430, 145 + 29 * len(items)),
        ),
        xaxis_title="Transferred data (GB)",
        yaxis_title="Source → destination",
    )
    figure.update_xaxes(range=[0, max(values) * 1.18])
    return figure


def local_ssd_write_figure(
    summary: TraceSummary, *, limit: int = 30
) -> go.Figure:
    """Show cumulative owner-local DRAM-to-SSD traffic by logical job."""

    items = sorted(
        (
            (service, float(written_gb))
            for service, written_gb in summary.service_local_ssd_write_gb.items()
            if written_gb > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not items:
        figure = go.Figure()
        figure.add_annotation(
            text="No completed owner-local DRAM → SSD writes were found.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
        figure.update_layout(
            **_base_layout("Per-job owner-local SSD write traffic", 360)
        )
        return figure

    selected = items[: max(1, limit)]
    omitted = items[max(1, limit) :]
    rows = []
    for service, total_gb in selected:
        rows.append(
            {
                "service": service,
                "planned": float(
                    summary.service_local_ssd_write_mode_gb[(service, "planned")]
                ),
                "fallback": float(
                    summary.service_local_ssd_write_mode_gb[(service, "fallback")]
                ),
                "total": total_gb,
                "events": int(summary.service_local_ssd_write_events[service]),
                "latest": summary.service_local_ssd_latest_iteration.get(service),
            }
        )
    if omitted:
        omitted_names = {service for service, _value in omitted}
        rows.append(
            {
                "service": f"Other jobs ({len(omitted_names)})",
                "planned": sum(
                    summary.service_local_ssd_write_mode_gb[(service, "planned")]
                    for service in omitted_names
                ),
                "fallback": sum(
                    summary.service_local_ssd_write_mode_gb[(service, "fallback")]
                    for service in omitted_names
                ),
                "total": sum(value for _service, value in omitted),
                "events": sum(
                    summary.service_local_ssd_write_events[service]
                    for service in omitted_names
                ),
                "latest": None,
            }
        )

    rows.reverse()
    total_all = sum(summary.service_local_ssd_write_gb.values())
    customdata = [
        [
            row["events"],
            row["latest"] if row["latest"] is not None else "—",
            row["total"],
            100.0 * row["total"] / total_all if total_all else 0.0,
        ]
        for row in rows
    ]
    figure = go.Figure()
    for mode, label, color in (
        ("planned", "Planned owner-local persistence", "#1971c2"),
        ("fallback", "Fallback after peer placement failed", "#f08c00"),
    ):
        values = [row[mode] for row in rows]
        figure.add_trace(
            go.Bar(
                x=values,
                y=[row["service"] for row in rows],
                orientation="h",
                name=label,
                marker_color=color,
                customdata=customdata,
                text=[f"{value:,.1f} GB" if value > 0 else "" for value in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "%{y}<br>"
                    + label
                    + ": %{x:,.3f} GB"
                    "<br>%{customdata[2]:,.3f} GB total"
                    "<br>%{customdata[3]:.2f}% of all local writes"
                    "<br>%{customdata[0]:,} completed write records"
                    "<br>Latest recorded iteration: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **_base_layout(
            "Per-job owner-local SSD write traffic",
            max(430, 150 + 29 * len(rows)),
        ),
        barmode="stack",
        xaxis_title="Cumulative completed DRAM → local SSD writes (GB)",
        yaxis_title="Logical checkpoint owner job",
        legend={"orientation": "h", "y": 1.08},
    )
    maximum = max(row["total"] for row in rows)
    if maximum > 0:
        figure.update_xaxes(range=[0, maximum * 1.2])
    return figure


def latency_figure(summary: TraceSummary) -> go.Figure:
    items = sorted(
        summary.operations.items(),
        key=lambda item: item[1].node_seconds,
        reverse=True,
    )[:20]
    items.reverse()
    labels = [
        f"{category} · {operation_label(operation)}"
        for (category, operation), _ in items
    ]
    p50 = [stat.durations.percentile(0.50) for _, stat in items]
    p95 = [stat.durations.percentile(0.95) for _, stat in items]
    custom = [
        [
            stat.count,
            stat.logical_count,
            stat.durations.percentile(0.99),
            stat.maximum,
            stat.node_seconds,
            stat.data_gb,
        ]
        for _, stat in items
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=p95,
            y=labels,
            orientation="h",
            name="p95",
            marker_color="#9ec5fe",
            customdata=custom,
            hovertemplate=(
                "%{y}<br>p95 %{x:.4g}s<br>p99 %{customdata[2]:.4g}s"
                "<br>max %{customdata[3]:.4g}s<br>records %{customdata[0]:,}"
                "<br>logical spans %{customdata[1]:,.0f}"
                "<br>node-seconds %{customdata[4]:,.1f}"
                "<br>data %{customdata[5]:,.1f} GB<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            x=p50,
            y=labels,
            orientation="h",
            name="p50",
            marker_color="#1971c2",
            hovertemplate="%{y}<br>p50 %{x:.4g}s<extra></extra>",
        )
    )
    figure.update_layout(
        **_base_layout("Operation latency for the highest-work span types", 700),
        xaxis_title="Span duration (seconds; reservoir-estimated percentiles)",
        yaxis_title="",
        barmode="overlay",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def path_figure(summary: TraceSummary) -> go.Figure:
    items = [
        item
        for item in summary.checkpoint_paths.most_common()
        if item[0] != "l3_drain"
    ]
    if not items:
        figure = go.Figure()
        figure.add_annotation(
            text="No checkpoint path labels were found.",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        figure.update_layout(**_base_layout("Observed checkpoint placement paths", 360))
        return figure
    figure = go.Figure(
        go.Pie(
            labels=[name for name, _ in items],
            values=[value for _, value in items],
            hole=0.56,
            sort=False,
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,.0f} logical spans<extra></extra>",
        )
    )
    figure.update_layout(**_base_layout("Observed checkpoint placement paths", 430))
    return figure


def checkpoint_policy_figure(
    summary: TraceSummary, context: RunContext
) -> go.Figure:
    classes = context.scenario.get("classes") or {}
    configured = configured_checkpoint_policy(context)
    if not classes:
        figure = go.Figure()
        figure.add_annotation(
            text=(
                f"Configured: <b>{html.escape(configured.label)}</b><br>"
                f"Observed: <b>{html.escape(_observed_strategy_label(summary))}</b><br>"
                "Pass --scenario to include class-level cadence and shard size."
            ),
            x=0.5,
            y=0.5,
            showarrow=False,
            align="center",
        )
        figure.update_layout(**_base_layout("Checkpoint policy by workload class", 330))
        return figure

    cluster = context.scenario.get("cluster") or {}
    arm_cfg = _arm_config(context)
    result_policy = _result_policy_metadata(context)
    policy_by_class = result_policy.get("by_class") or {}

    def iteration_wall_seconds(spec: dict[str, Any]) -> float:
        compute = float(spec.get("iteration_seconds", 0.0))
        ranks = int(spec.get("ranks", 1))
        rates = spec.get("rates") or {}
        nic = float(rates.get("nic", cluster.get("network_bandwidth_gbps", 0.0)))
        gradient = float(spec.get("gradient_gb_per_rank", 4.0))
        all_reduce = (
            2.0 * (ranks - 1) / ranks * gradient / nic
            if ranks > 1 and nic > 0
            else 0.0
        )
        return compute + all_reduce

    def interval(iterations: Any, wall_seconds: float) -> str:
        if iterations in (None, ""):
            return "not embedded"
        values = iterations if isinstance(iterations, (list, tuple, set)) else [iterations]
        formatted = []
        for value in values:
            try:
                count = int(value)
            except (TypeError, ValueError):
                formatted.append(str(value))
                continue
            if count >= 10**8:
                formatted.append("disabled")
            else:
                formatted.append(f"{count} it (~{count * wall_seconds:g}s)")
        return ", ".join(dict.fromkeys(formatted))

    baseline_policy = None
    if arm_cfg.get("baseline"):
        from checkpointing import baselines

        baseline_policy = baselines.resolve(arm_cfg)

    rows = []
    for class_name, spec in classes.items():
        wall_seconds = iteration_wall_seconds(spec)
        default_every = int(spec.get("checkpoint_every", 0))
        l1 = interval(default_every, wall_seconds)
        l2 = interval(default_every, wall_seconds)
        l3 = "scenario/backstop dependent"
        placement = (
            f"cross-job / k={int(spec.get('kpeers', 0))}"
            if int(spec.get("kpeers", 0)) > 0
            else "local"
        )

        if configured.kind == "ideal":
            l1 = l2 = l3 = "disabled"
            placement = "none"
        elif configured.kind.startswith("external:"):
            decision = policy_by_class.get(class_name) or {}
            if isinstance(decision, dict) and decision:
                l1 = interval(decision.get("capture_every"), wall_seconds)
                l2 = interval(decision.get("checkpoint_every"), wall_seconds)
                l3 = interval(decision.get("store_every"), wall_seconds)
                peers = decision.get("kpeers")
                placement = (
                    f"policy-selected / k={peers}"
                    if peers not in (None, "")
                    else "policy-selected"
                )
            else:
                l1 = l2 = l3 = "external policy; not embedded"
                placement = "see observed trace"
        elif baseline_policy is not None:
            baseline_name = str(arm_cfg["baseline"])
            every = (
                baseline_policy.every_rule(spec, cluster)
                if baseline_policy.every_rule is not None
                else default_every
            )
            cadence = interval(every, wall_seconds)
            if baseline_name == "gemini":
                l1 = cadence
                l2 = f"in-job DRAM replica; {cadence}"
                store_every = (
                    baseline_policy.store_every_rule(spec, cluster)
                    if baseline_policy.store_every_rule is not None
                    else None
                )
                l3 = f"remote store; {interval(store_every, wall_seconds)}"
                placement = f"in-job replication / k={int(spec.get('kpeers', 0))}"
            elif baseline_name == "checkfreq":
                l1 = cadence
                l2 = f"local SSD; {cadence}"
                l3 = "none"
                placement = "local SSD"
            elif baseline_name == "checknrun":
                l1 = cadence
                l2 = "none"
                l3 = f"0.1x object-store write; {cadence}"
                placement = "object store"
            elif baseline_name == "megascale":
                l1 = cadence
                l2 = "none"
                l3 = f"full-size object-store write; {cadence}"
                placement = "object store"
            elif baseline_name == "jit":
                l1 = l2 = l3 = "no periodic checkpoint"
                placement = "survivor drain on failure"
        elif configured.kind == "naive-all-tiers":
            cadence_s = float(arm_cfg["cadence_s"])
            every = max(1, round(cadence_s / max(wall_seconds, 1e-9)))
            cadence = f"{cadence_s:g}s ({interval(every, wall_seconds)})"
            l1 = l2 = l3 = cadence
            placement = (
                f"cross-job / k={int(spec.get('kpeers', 0))}"
                if arm_cfg.get("kpeers", True)
                else "local"
            )
        elif configured.kind == "naive-tiered":
            l2_s = float(arm_cfg["cadence_l2_s"])
            l1_s = float(arm_cfg.get("cadence_l1_s", l2_s))
            l3_s = float(arm_cfg.get("cadence_l3_s", l2_s))

            def seconds_interval(seconds: float) -> str:
                every = max(1, round(seconds / max(wall_seconds, 1e-9)))
                return f"{seconds:g}s ({interval(every, wall_seconds)})"

            l1 = seconds_interval(l1_s)
            l2 = seconds_interval(l2_s)
            l3 = seconds_interval(l3_s)
            placement = (
                f"cross-job / k={int(spec.get('kpeers', 0))}"
                if arm_cfg.get("kpeers", True)
                else "local"
            )
        elif configured.kind == "fixed-iterations":
            every = int(arm_cfg["checkpoint_every"])
            l1 = l2 = interval(every, wall_seconds)

        rows.append(
            (
                class_name,
                int(spec.get("count", 0)),
                int(spec.get("ranks", 0)),
                l1,
                l2,
                l3,
                placement,
                float(spec.get("checkpoint_gb_per_rank", 0.0)),
                float(spec.get("rpo_s", 0.0)),
            )
        )
    figure = go.Figure(
        go.Table(
            header={
                "values": [
                    "Class",
                    "Jobs",
                    "Nodes/job",
                    "L1 capture",
                    "L2 persistence",
                    "L3 durability",
                    "Placement",
                    "Shard GB/node",
                    "RPO target (s)",
                ],
                "fill_color": "#fff3bf",
                "align": "left",
                "font": {"color": "#102a43", "size": 12},
                "height": 30,
            },
            cells={
                "values": list(zip(*rows)),
                "format": [None, ",d", ",d", None, None, None, None, ",.1f", ",.1f"],
                "fill_color": "#ffffff",
                "align": "left",
                "height": 27,
            },
        )
    )
    figure.update_layout(
        **_base_layout(
            f"Configured policy by workload class: {configured.label}",
            max(340, 155 + 29 * len(rows)),
        )
    )
    return figure


def reliability_figure(summary: TraceSummary, context: RunContext) -> go.Figure:
    failures = (context.result.get("failures") or {}) if context.result else {}
    recoveries = (
        (context.result.get("recovery_source_tiers") or {})
        if context.result
        else dict(summary.recovery_tiers)
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Failure / preemption incidents", "Recoveries by source"),
        horizontal_spacing=0.14,
    )
    failure_items = sorted(failures.items(), key=lambda item: int(item[1]))
    recovery_items = sorted(recoveries.items(), key=lambda item: int(item[1]))
    if failure_items:
        figure.add_trace(
            go.Bar(
                x=[int(value) for _, value in failure_items],
                y=[str(name).replace("_", " ") for name, _ in failure_items],
                orientation="h",
                marker_color="#e03131",
                text=[f"{int(value):,}" for _, value in failure_items],
                textposition="outside",
                hovertemplate="%{y}<br>%{x:,} incidents<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    else:
        figure.add_annotation(
            text="No result-sidecar failure counts supplied",
            x=0.2,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    if recovery_items:
        total = sum(int(value) for _, value in recovery_items)
        figure.add_trace(
            go.Bar(
                x=[int(value) for _, value in recovery_items],
                y=[str(name).replace("_", " ") for name, _ in recovery_items],
                orientation="h",
                marker_color="#0ca678",
                customdata=[
                    100 * int(value) / total if total else 0
                    for _, value in recovery_items
                ],
                text=[
                    f"{int(value):,} ({100 * int(value) / total if total else 0:.1f}%)"
                    for _, value in recovery_items
                ],
                textposition="outside",
                hovertemplate=(
                    "%{y}<br>%{x:,} recoveries"
                    "<br>%{customdata:.2f}% of all<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    figure.update_layout(
        **_base_layout("Failure and recovery outcomes", 430),
        bargap=0.28,
    )
    figure.update_xaxes(title_text="Count", row=1, col=1)
    figure.update_xaxes(title_text="Count", row=1, col=2)
    return figure


def service_table_figure(summary: TraceSummary) -> go.Figure:
    classes = sorted(
        summary.service_classes,
        key=lambda name: summary.class_node_seconds[name],
        reverse=True,
    )
    node_totals: Counter = Counter()
    for (service, _node), value in summary.node_weights.items():
        node_totals[service_class(service)] += value
    values = [
        classes,
        [len(summary.class_jobs[name]) for name in classes],
        [node_totals[name] for name in classes],
        [summary.class_events[name] for name in classes],
        [summary.class_node_seconds[name] for name in classes],
        [summary.class_data_gb[name] for name in classes],
    ]
    figure = go.Figure(
        go.Table(
            header={
                "values": [
                    "Service class",
                    "Jobs",
                    "Modeled nodes",
                    "Event records",
                    "Node-seconds",
                    "Transferred GB",
                ],
                "fill_color": "#e7f5ff",
                "align": "left",
                "font": {"color": "#102a43", "size": 12},
                "height": 30,
            },
            cells={
                "values": values,
                "format": [None, ",d", ",.0f", ",d", ",.1f", ",.1f"],
                "fill_color": "#ffffff",
                "align": "left",
                "height": 27,
            },
        )
    )
    figure.update_layout(
        **_base_layout("Service inventory", max(320, 150 + 28 * len(classes)))
    )
    return figure


def _format_number(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def _ideal_runtime_seconds(class_name: str, context: RunContext) -> float | None:
    spec = (context.scenario.get("classes") or {}).get(class_name)
    cluster = context.scenario.get("cluster") or {}
    if not spec:
        return None
    ranks = int(spec.get("ranks", 0))
    if ranks <= 0:
        return None
    nic = float(
        (spec.get("rates") or {}).get(
            "nic", cluster.get("network_bandwidth_gbps", 0.0)
        )
    )
    if nic <= 0:
        return None
    all_reduce = (
        2
        * (ranks - 1)
        / ranks
        * float(spec.get("gradient_gb_per_rank", 4.0))
        / nic
    )
    return int(spec.get("iterations", 0)) * (
        float(spec.get("iteration_seconds", 0.0)) + all_reduce
    )


def _service_node_totals(summary: TraceSummary) -> Counter:
    totals: Counter = Counter()
    for (service, _node), weight in summary.node_weights.items():
        totals[service] += weight
    return totals


def _job_metric_rows(
    summary: TraceSummary, context: RunContext
) -> list[dict[str, Any]]:
    service_nodes = _service_node_totals(summary)
    rows = []
    for service, training_end in summary.service_training_end.items():
        class_name = service_class(service)
        start = summary.service_arrival.get(
            service, summary.service_start.get(service, summary.min_time)
        )
        end = summary.service_departure.get(service, training_end)
        actual = max(0.0, end - start)
        ideal = _ideal_runtime_seconds(class_name, context)
        overhead = actual - ideal if ideal is not None else None
        goodput = 100.0 * ideal / actual if ideal is not None and actual else None
        checkpoint_gb = summary.service_checkpoint_gb[service]
        recoveries = {
            tier: int(value)
            for (job, tier), value in summary.service_recovery_tiers.items()
            if job == service
        }
        rows.append(
            {
                "service": service,
                "class": class_name,
                "nodes": service_nodes[service],
                "ideal": ideal,
                "actual": actual,
                "overhead": overhead,
                "overhead_pct": (
                    100.0 * overhead / ideal if ideal and overhead is not None else None
                ),
                "goodput": goodput,
                "checkpoint_gb": checkpoint_gb,
                "checkpoint_rate": checkpoint_gb / actual if actual else 0.0,
                "recoveries": sum(recoveries.values()),
                "recovery_mix": ", ".join(
                    f"{tier}: {count}" for tier, count in sorted(recoveries.items())
                )
                or "—",
            }
        )
    rows.sort(key=lambda row: (-row["nodes"], row["service"]))
    return rows


def _aggregate_metrics_table(summary: TraceSummary, context: RunContext) -> str:
    job_rows = _job_metric_rows(summary, context)
    result = context.result
    actual_makespan = float(
        result.get("makespan_s", max((row["actual"] for row in job_rows), default=summary.duration))
    )
    ideals = [row["ideal"] for row in job_rows if row["ideal"] is not None]
    ideal_makespan = max(ideals, default=None)
    ideal_node_seconds = sum(
        row["ideal"] * row["nodes"]
        for row in job_rows
        if row["ideal"] is not None
    )
    actual_node_seconds = sum(row["actual"] * row["nodes"] for row in job_rows)
    initial_nodes = sum(
        row["nodes"]
        for row in job_rows
        if summary.service_arrival.get(
            row["service"], summary.service_start.get(row["service"], summary.min_time)
        ) <= summary.min_time + 1e-9
    )
    cluster_nodes = int((context.scenario.get("cluster") or {}).get("node_count", 0))
    useful_node_iterations = sum(
        int(spec.get("count", 0))
        * int(spec.get("ranks", 0))
        * int(spec.get("iterations", 0))
        for spec in (context.scenario.get("classes") or {}).values()
    )
    checkpoint_gb = sum(
        stat.data_gb
        for (category, _operation), stat in summary.operations.items()
        if category == "Checkpoint"
    )
    local_ssd_write_gb = sum(summary.service_local_ssd_write_gb.values())
    routed_gb = sum(summary.routes.values())
    same_class_gb = sum(
        value for (source, destination), value in summary.routes.items()
        if source == destination
    )
    entries: list[tuple[str, str, str]] = []
    if ideal_makespan is not None:
        overhead = actual_makespan - ideal_makespan
        entries.extend(
            [
                (
                    "Ideal critical-path runtime",
                    f"{ideal_makespan:,.1f} s",
                    "Longest job with compute + modeled all-reduce only; no checkpoints, failures, recovery, queueing, or rollback.",
                ),
                (
                    "Actual training makespan",
                    f"{actual_makespan:,.1f} s",
                    "Time until the final training job completed.",
                ),
                (
                    "Critical-path overhead",
                    f"{overhead:,.1f} s ({100 * overhead / ideal_makespan:.2f}%)",
                    "(actual makespan − ideal makespan) / ideal makespan.",
                ),
                (
                    "Weighted training goodput",
                    f"{100 * ideal_node_seconds / actual_node_seconds:.2f}%"
                    if actual_node_seconds
                    else "—",
                    "Ideal useful node-seconds / actual allocated node-seconds across all jobs. Higher is better.",
                ),
                (
                    "Useful training throughput",
                    f"{useful_node_iterations / actual_makespan:,.2f} node-iterations/s"
                    if actual_makespan
                    else "—",
                    "Configured completed iterations × job nodes / actual makespan.",
                ),
                (
                    "Average allocated nodes",
                    f"{actual_node_seconds / actual_makespan:,.1f}"
                    if actual_makespan
                    else "—",
                    "Sum of per-job allocated node-seconds / training makespan.",
                ),
            ]
        )
    if cluster_nodes:
        entries.append(
            (
                "Initial training allocation",
                f"{initial_nodes:,.0f} / {cluster_nodes:,} nodes ({100 * initial_nodes / cluster_nodes:.2f}%)",
                f"Workload nodes admitted at simulated time zero; "
                f"{cluster_nodes - initial_nodes:,.0f} configured cluster nodes "
                "were initially unclaimed.",
            )
        )
    entries.extend(
        [
            (
                "Checkpoint traffic",
                f"{checkpoint_gb:,.1f} GB",
                "Bytes recorded across all checkpoint stages; the same payload is counted again when it moves to another tier.",
            ),
            (
                "Owner-local DRAM → SSD writes",
                f"{local_ssd_write_gb:,.1f} GB",
                "Cumulative completed local checkpoint writes. Rewrites count again; this is traffic, not retained SSD footprint.",
            ),
            (
                "Aggregate checkpoint throughput",
                f"{checkpoint_gb / actual_makespan:,.2f} GB/s"
                if actual_makespan
                else "—",
                "All checkpoint-stage GB / training makespan. This is cluster-wide traffic rate, not one-link bandwidth.",
            ),
            (
                "Same-class cross-job traffic",
                f"{same_class_gb:,.1f} GB ({100 * same_class_gb / routed_gb if routed_gb else 0:.2f}%)",
                "Transfers between different job IDs that share a workload class, such as frontier0 → frontier4. True same-job transfers are excluded.",
            ),
        ]
    )
    if result:
        failures = sum(int(value) for value in (result.get("failures") or {}).values())
        entries.extend(
            [
                (
                    "Durable checkpoint fraction",
                    f"{100 * float(result.get('durable_frac') or 0):.2f}%",
                    "Completed rank flushes judged durable by the scenario analyzer.",
                ),
                (
                    "Failures / preemptions",
                    f"{failures:,}",
                    "Failure outcomes from the result summary; these are not emitted as standalone trace spans.",
                ),
                (
                    "Restores",
                    f"{int(result.get('restores') or 0):,}",
                    "Recovery attempts counted by restored cohort/shard.",
                ),
                (
                    "Aborted flushes",
                    f"{int(result.get('aborted_flushes') or 0):,}",
                    "Checkpoint flushes interrupted before successful persistence.",
                ),
                (
                    "Partial peer grants",
                    f"{int(result.get('partial_grants') or 0):,}",
                    "Flushes that received fewer peer donors than requested.",
                ),
                (
                    "Successful rank flushes",
                    f"{int(result.get('rank_flushes') or 0):,}",
                    "Logical rank-level checkpoint persist outcomes.",
                ),
                (
                    "Peer flushes",
                    f"{int(result.get('peer_flushes') or 0):,}",
                    "Logical flushes placed on cross-job peer storage.",
                ),
                (
                    "Local fallback flushes",
                    f"{int(result.get('fallback_flushes') or 0):,}",
                    "Logical flushes kept on owner-local SSD when peer placement was unavailable.",
                ),
                (
                    "L3 donor-drain traffic",
                    f"{float(result.get('drain_bytes_gb') or 0):,.1f} GB",
                    "Bytes forwarded from donor SSDs to object storage for the durability backstop.",
                ),
                (
                    "L3 drain outcomes",
                    f"{int(result.get('l3_drains') or 0):,} complete / {int(result.get('l3_drain_aborted') or 0):,} aborted",
                    "Completed and interrupted donor-side L3 drain pieces.",
                ),
                (
                    "Donor-drain NIC share",
                    f"{float(result.get('donor_drain_nic_pct') or 0):.4f}%",
                    "Scenario analyzer's donor NIC utilization attributed to L3 drain traffic.",
                ),
                (
                    "Rollback iterations",
                    f"p50 {int(result.get('lost_iters_p50') or 0):,} / max {int(result.get('lost_iters_max') or 0):,}",
                    "Iterations re-executed after restoration; median and worst observed rollback.",
                ),
            ]
        )
    recovery_sources = (
        result.get("recovery_source_tiers") if result else None
    ) or dict(summary.recovery_tiers)
    recovery_total = sum(int(value) for value in recovery_sources.values())
    initial_resets = int(recovery_sources.get("initial_state", 0))
    if recovery_total:
        entries.append(
            (
                "Recovered from checkpoint state",
                f"{recovery_total - initial_resets:,} ({100 * (recovery_total - initial_resets) / recovery_total:.2f}%)",
                "Recoveries served by DRAM, peer, SSD, or object-store checkpoint state rather than resetting to initial state.",
            )
        )
    for source, count in sorted(
        recovery_sources.items(), key=lambda item: int(item[1]), reverse=True
    ):
        entries.append(
            (
                f"Recovery from {str(source).replace('_', ' ')}",
                f"{int(count):,} ({100 * int(count) / recovery_total if recovery_total else 0:.2f}%)",
                "Count and share of all recoveries by selected source tier.",
            )
        )
    if result:
        for failure_type, count in sorted(
            (result.get("failures") or {}).items(),
            key=lambda item: int(item[1]),
            reverse=True,
        ):
            entries.append(
                (
                    f"Failure: {str(failure_type).replace('_', ' ')}",
                    f"{int(count):,}",
                    "Incident count from the scenario result summary.",
                )
            )
    body = "".join(
        "<tr><td>"
        + html.escape(metric)
        + "</td><td><strong>"
        + html.escape(value)
        + "</strong></td><td>"
        + html.escape(definition)
        + "</td></tr>"
        for metric, value, definition in entries
    )
    return (
        "<div class='table-wrap metric-table-wrap'><table class='metric-table'>"
        "<thead><tr><th>Metric</th><th>Aggregate value</th><th>Definition</th>"
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _job_metrics_table(summary: TraceSummary, context: RunContext) -> str:
    rows = _job_metric_rows(summary, context)

    def number(value: float | None, pattern: str) -> str:
        return "—" if value is None else format(value, pattern)

    body = "".join(
        "<tr data-job='"
        + html.escape(f"{row['service']} {row['class']}".lower(), quote=True)
        + "'><td><code>"
        + html.escape(row["service"])
        + "</code></td><td>"
        + html.escape(row["class"])
        + "</td><td>"
        + f"{row['nodes']:,.0f}"
        + "</td><td>"
        + number(row["ideal"], ",.1f")
        + "</td><td>"
        + number(row["actual"], ",.1f")
        + "</td><td>"
        + number(row["overhead"], ",.1f")
        + "</td><td>"
        + number(row["overhead_pct"], ",.2f")
        + "%</td><td>"
        + number(row["goodput"], ",.2f")
        + "%</td><td>"
        + f"{row['checkpoint_gb']:,.1f}"
        + "</td><td>"
        + f"{row['checkpoint_rate']:,.2f}"
        + "</td><td>"
        + f"{row['recoveries']:,}"
        + "</td><td>"
        + html.escape(row["recovery_mix"])
        + "</td></tr>"
        for row in rows
    )
    return f"""
      <div class="trace-head"><h2>Per-job efficiency and recovery</h2><input id="job-search" placeholder="Filter job or class…"></div>
      <div class="table-wrap job-table-wrap"><table><thead><tr><th>Job</th><th>Class</th><th>Nodes</th><th>Ideal (s)</th><th>Actual (s)</th><th>Overhead (s)</th><th>Overhead %</th><th>Goodput %</th><th>Checkpoint GB</th><th>Checkpoint GB/s</th><th>Recoveries</th><th>Recovery sources</th></tr></thead><tbody id="job-rows">{body}</tbody></table></div>
      <p class="panel-note">Per-job goodput is ideal compute+all-reduce runtime divided by observed training completion time. Checkpoint GB counts every recorded tier-to-tier movement, so it measures traffic rather than unique model state.</p>
    """


def _cards(summary: TraceSummary, context: RunContext) -> str:
    cards = [
        ("Trace duration", f"{summary.duration:,.1f} s"),
        ("Modeled nodes", _format_number(summary.physical_nodes)),
        ("Services / jobs", f"{len(summary.services):,}"),
        ("Event records", _format_number(summary.event_count)),
        ("Logical spans", _format_number(summary.logical_event_count)),
        ("Transferred data", f"{summary.total_data_gb:,.1f} GB"),
        ("Standalone failure spans", _format_number(summary.failure_count)),
    ]
    result = context.result
    if result:
        failures = sum(int(value) for value in (result.get("failures") or {}).values())
        cards.extend(
            [
                ("Training makespan", f"{float(result.get('makespan_s') or 0):,.1f} s"),
                ("Failures / preemptions", f"{failures:,}"),
                ("Restores", f"{int(result.get('restores') or 0):,}"),
                ("Durable checkpoints", f"{100 * float(result.get('durable_frac') or 0):.2f}%"),
            ]
        )
    return "".join(
        "<div class='metric'><span>"
        + html.escape(label)
        + "</span><strong>"
        + html.escape(value)
        + "</strong></div>"
        for label, value in cards
    )


def _checkpoint_panel(summary: TraceSummary, context: RunContext) -> str:
    configured = configured_checkpoint_policy(context)
    observed = _observed_strategy_label(summary)
    arm_cfg = _arm_config(context)
    result_policy = _result_policy_metadata(context)
    flags = arm_cfg or result_policy.get("flags") or {}
    arm_text = html.escape(json.dumps(flags, sort_keys=True)) if flags else "not embedded"
    paths = [
        item
        for item in summary.checkpoint_paths.most_common()
        if item[0] != "l3_drain"
    ]
    path_total = sum(value for _, value in paths)
    l3_drains = summary.checkpoint_paths.get("l3_drain", 0)
    controller = context.scenario.get("controller") or {}
    slot_period = controller.get("slot_period_s")
    epoch = flags.get("controller_epoch_s") if flags else None
    scheduling = (
        f"{float(slot_period):g}s clock slots, repacked every {float(epoch):g}s"
        if flags.get("slots") and slot_period is not None and epoch is not None
        else "no controller schedule configured or embedded"
    )
    path_rows = "".join(
        "<tr><td><code>"
        + html.escape(name)
        + "</code></td><td>"
        + f"{value:,.0f}"
        + "</td><td>"
        + f"{100 * value / path_total if path_total else 0:.2f}%"
        + "</td></tr>"
        for name, value in paths
    )
    if not path_rows:
        path_rows = "<tr><td colspan='3'>No completed placement paths recorded</td></tr>"
    pipeline = "<b>&rarr;</b>".join(
        f"<span>{html.escape(stage)}</span>" for stage in configured.pipeline
    )
    pipeline_html = f'<div class="pipeline">{pipeline}</div>' if pipeline else ""
    path_explanations = {
        "peer": "cross-job peer placement",
        "peer_partial": "cross-job placement with fewer donors than requested",
        "local_fallback": "owner-local SSD fallback",
        "gemini_replica": "in-job peer-DRAM replication",
        "store": "direct object-store persistence",
        "store_backstop": "object-store durability backstop",
    }
    observed_paths = "; ".join(
        f"{html.escape(name)} = "
        f"{html.escape(path_explanations.get(name, 'strategy-defined path'))}"
        for name, _ in paths
    ) or "No primary placement labels were emitted."
    l3_note = (
        f" A separate downstream L3 stage recorded {l3_drains:,.0f} drain spans."
        if l3_drains
        else ""
    )
    return f"""
      <div class="explain-card featured">
        <span class="eyebrow">Configured policy</span>
        <h3>{html.escape(configured.label)}</h3>
        <p>{html.escape(configured.description)}</p>
        {pipeline_html}
        <p class="small"><strong>Arm:</strong> {html.escape(context.arm or 'not supplied')} &nbsp; <strong>Scheduling:</strong> {html.escape(scheduling)}<br><strong>Flags:</strong> <code>{arm_text}</code></p>
      </div>
      <div class="explain-card">
        <span class="eyebrow">Observed trace behavior</span>
        <h3>{html.escape(observed)}</h3>
        <p class="small">This label comes from completed trace records and is intentionally separate from configured arm intent.</p>
        <table class="compact"><thead><tr><th>Path</th><th>Logical spans</th><th>Share</th></tr></thead><tbody>{path_rows}</tbody></table>
        <p class="small">{observed_paths}.{l3_note}</p>
      </div>
    """


def _priority_panel(summary: TraceSummary) -> str:
    strategy = summary.checkpoint_strategies.most_common(1)
    if strategy and strategy[0][0] == "crossjob_peer":
        details = """
          <div class="priority-row"><b>-10 · highest</b><span>Recovery DRAM→GPU tail; CPU + GPU, foreground</span></div>
          <div class="priority-row"><b>0 · normal</b><span>GPU→DRAM snapshot/capture; CPU + GPU and can delay the next iteration</span></div>
          <div class="priority-row"><b>10 · lowest</b><span>Peer persistence transfer; CPU + network, asynchronous but coupled to all-reduce slowdown</span></div>
          <div class="priority-row"><b>QoS 0.05</b><span>Donor→L3 drain; no CPU hold, background weighted network sharing</span></div>
        """
    else:
        details = "<p>Operation priority details are not available for this strategy.</p>"
    return f"""
      <div class="explain-card warning">
        <span class="eyebrow">Job scheduling priority</span>
        <h3>Equal / not configured</h3>
        <p>This scenario does not define per-job priority, queue order, preemption rank, or weighted fair-share. Workload classes differ in size, checkpoint cadence, hardware rates, and failure exposure—not scheduler importance.</p>
      </div>
      <div class="explain-card">
        <span class="eyebrow">Operation-level CPU priority</span>
        {details}
        <p class="small">Lower numeric values are served first by the simulator's priority resource. These priorities affect CPU acquisition only; shared network capacity is allocated by the capacity model.</p>
      </div>
    """


def _findings_panel(summary: TraceSummary, context: RunContext) -> str:
    service_nodes = _service_node_totals(summary)
    class_nodes: Counter = Counter()
    class_ends: dict[str, float] = {}
    for service, training_end in summary.service_training_end.items():
        class_name = service_class(service)
        end = summary.service_departure.get(service, training_end)
        class_nodes[class_name] += service_nodes[service]
        class_ends[class_name] = max(class_ends.get(class_name, end), end)
    last_class = max(class_ends, key=class_ends.get) if class_ends else "unknown"
    remaining = class_nodes[last_class]
    released = sum(class_nodes.values()) - remaining
    earlier_end = max(
        (end for name, end in class_ends.items() if name != last_class),
        default=summary.min_time,
    )
    top_operation, top_stat = max(
        summary.operations.items(),
        key=lambda item: item[1].node_seconds,
        default=(("Other", "unknown"), OperationStat()),
    )
    result = context.result
    failures = sum(int(value) for value in (result.get("failures") or {}).values())
    restores = int(result.get("restores") or sum(summary.recovery_tiers.values()))
    configured = configured_checkpoint_policy(context)
    observed = _observed_strategy_label(summary)
    return f"""
      <div class="finding"><span>Why training nodes fall</span><strong>{released:,.0f} nodes became idle checkpoint donors</strong><p>Earlier classes finished training by about {earlier_end:,.1f}s. Their fixed nodes remain available for checkpointing only. The remaining {remaining:,.0f} <code>{html.escape(last_class)}</code> nodes trained until {class_ends.get(last_class, 0):,.1f}s.</p></div>
      <div class="finding"><span>Makespan owner</span><strong>{html.escape(last_class)}</strong><p>This class finished last and therefore determined the run's training makespan.</p></div>
      <div class="finding"><span>Checkpoint technique</span><strong>{html.escape(configured.label)}</strong><p>Configured arm intent. The trace observed <strong>{html.escape(observed)}</strong>; see Policy and paths for both views.</p></div>
      <div class="finding"><span>Largest recorded work</span><strong>{html.escape(operation_label(top_operation[1]))}</strong><p>{top_stat.node_seconds / 3600:,.1f} aggregate node-hours. Operations overlap, so this is work—not wall-clock share.</p></div>
      <div class="finding"><span>Reliability outcome</span><strong>{restores:,} restores</strong><p>{failures:,} failures/preemptions are reported by the run summary. Failure events are not standalone trace spans.</p></div>
    """


def _span_table(summary: TraceSummary) -> str:
    spans = [item[2] for item in sorted(summary.top_spans, reverse=True)]
    rows = []
    for span in spans:
        rows.append(
            "<tr data-text='"
            + html.escape(
                " ".join(
                    str(span[key])
                    for key in ("service", "class", "category", "operation", "node", "status")
                ).lower(),
                quote=True,
            )
            + "' data-duration='"
            + f"{span['duration']:.12g}"
            + "'><td>"
            + html.escape(str(span["service"]))
            + "</td><td><span class='pill pill-"
            + html.escape(str(span["category"]).lower())
            + "'>"
            + html.escape(str(span["category"]))
            + "</span></td><td><code>"
            + html.escape(str(span["operation"]))
            + "</code></td><td>"
            + f"{span['start']:.3f}"
            + "</td><td>"
            + f"{span['duration']:.4g}"
            + "</td><td>"
            + html.escape(str(span["iteration"] if span["iteration"] is not None else "—"))
            + "</td><td>"
            + html.escape(str(span["rank"] if span["rank"] is not None else "—"))
            + "</td><td>"
            + (f"{span['data_gb']:.3g}" if span["data_gb"] else "—")
            + "</td><td>"
            + html.escape(str(span["status"]))
            + "</td></tr>"
        )
    return "".join(rows)


def _figure_html(figure: go.Figure, *, include_plotly: bool) -> str:
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=True if include_plotly else False,
        config={"responsive": True, "displaylogo": False},
    )


def write_dashboard(
    summary: TraceSummary,
    activity: Activity,
    *,
    output_path: Path,
    source_log: Path,
    context: RunContext,
) -> None:
    figures = [
        allocated_nodes_figure(activity, summary),
        completion_figure(summary),
        operation_work_figure(summary),
        latency_figure(summary),
        checkpoint_policy_figure(summary, context),
        path_figure(summary),
        reliability_figure(summary, context),
        activity_figure(activity),
        heatmap_figure(activity, summary),
        service_table_figure(summary),
        local_ssd_write_figure(summary),
        topology_figure(summary),
        route_figure(summary),
    ]
    rendered = [
        _figure_html(figure, include_plotly=index == 0)
        for index, figure in enumerate(figures)
    ]
    source = html.escape(str(source_log))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aggregate Trace Explorer</title>
  <style>
    :root {{ --ink:#102a43; --muted:#627d98; --line:#d9e2ec; --bg:#edf2f7; --brand:#0b7285; --blue:#1971c2; --gold:#f59f00; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--bg); font-family:Inter,Segoe UI,sans-serif; }}
    header {{ background:linear-gradient(120deg,#073b4c,#0b7285); color:white; padding:22px 30px 18px; position:sticky; top:0; z-index:5; box-shadow:0 2px 9px #102a4340; }}
    header h1 {{ margin:0; font-size:24px; letter-spacing:.2px; }}
    header p {{ margin:6px 0 0; opacity:.82; font-family:Consolas,monospace; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    nav {{ margin-top:13px; display:flex; gap:16px; flex-wrap:wrap; }}
    nav a {{ color:#e3fafc; text-decoration:none; font-size:13px; font-weight:600; }}
    main {{ max-width:1560px; margin:0 auto; padding:28px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:12px; margin-bottom:18px; }}
    .metric,.panel {{ background:white; border:1px solid var(--line); border-radius:8px; box-shadow:0 1px 3px #102a4314; }}
    .metric {{ padding:14px 16px; }}
    .metric span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.7px; }}
    .metric strong {{ display:block; margin-top:6px; font-size:21px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:18px; }}
    .findings {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; margin-bottom:22px; }}
    .finding {{ background:#073b4c; color:white; border-radius:10px; padding:17px; box-shadow:0 2px 8px #102a4320; }}
    .finding span,.eyebrow {{ display:block; text-transform:uppercase; letter-spacing:.8px; font-size:10px; font-weight:700; color:#99e9f2; }}
    .finding strong {{ display:block; font-size:20px; margin:7px 0; }}
    .finding p {{ color:#d9f3f5; font-size:12px; line-height:1.5; margin:0; }}
    .finding code {{ color:#fff; }}
    .section-title {{ margin:30px 0 12px; }}
    .section-title h2 {{ margin:0 0 5px; font-size:22px; }}
    .section-title p {{ color:var(--muted); margin:0; max-width:1050px; line-height:1.5; }}
    .panel {{ overflow:hidden; }}
    .panel.full {{ margin-bottom:18px; }}
    .panel-note {{ color:var(--muted); font-size:12px; padding:0 18px 14px; margin-top:-8px; }}
    .trace-head {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:17px 18px; border-bottom:1px solid var(--line); }}
    .trace-head h2 {{ margin:0 auto 0 0; font-size:18px; }}
    input {{ border:1px solid #bcccdc; border-radius:5px; padding:8px 10px; min-width:240px; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th {{ position:sticky; top:0; background:#e7f5ff; text-align:left; padding:9px; border-bottom:1px solid var(--line); }}
    td {{ padding:8px 9px; border-bottom:1px solid #edf2f7; white-space:nowrap; }}
    tr:hover {{ background:#f8fafc; }}
    .table-wrap {{ max-height:620px; overflow:auto; }}
    .metric-table-wrap {{ max-height:none; }}
    .metric-table td:nth-child(1) {{ font-weight:600; }}
    .metric-table td:nth-child(2) {{ color:#0b7285; font-size:13px; }}
    .metric-table td:nth-child(3) {{ white-space:normal; color:var(--muted); line-height:1.45; min-width:360px; }}
    .job-table-wrap {{ max-height:570px; }}
    .explain-grid {{ display:grid; grid-template-columns:1.2fr 1fr; gap:18px; margin-bottom:18px; }}
    .explain-card {{ background:white; border:1px solid var(--line); border-radius:9px; padding:20px; box-shadow:0 1px 3px #102a4314; }}
    .explain-card.featured {{ border-top:4px solid var(--blue); }}
    .explain-card.warning {{ border-top:4px solid var(--gold); }}
    .explain-card .eyebrow {{ color:#0b7285; }}
    .explain-card h3 {{ margin:7px 0 8px; font-size:21px; }}
    .explain-card p {{ color:#486581; line-height:1.55; }}
    .pipeline {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:16px 0; }}
    .pipeline span {{ background:#e7f5ff; border:1px solid #a5d8ff; border-radius:18px; padding:7px 11px; font-size:12px; font-weight:600; }}
    .pipeline b {{ color:#0b7285; }}
    .small {{ font-size:11px; }}
    .compact th,.compact td {{ position:static; padding:7px; }}
    .priority-row {{ display:grid; grid-template-columns:120px 1fr; gap:12px; padding:9px 0; border-bottom:1px solid #edf2f7; font-size:12px; }}
    code {{ color:#334e68; }}
    .pill {{ padding:2px 7px; border-radius:10px; background:#e9ecef; }}
    .pill-checkpoint {{ background:#fff3bf; }} .pill-training {{ background:#dbeafe; }}
    .pill-recovery,.pill-failure {{ background:#ffe3e3; }} .pill-communication {{ background:#e5dbff; }}
    footer {{ color:var(--muted); text-align:center; padding:12px 24px 30px; font-size:12px; }}
    @media(max-width:900px) {{ .grid,.explain-grid {{ grid-template-columns:1fr; }} main {{ padding:14px; }} .metric-table td:nth-child(3) {{ min-width:260px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Aggregate Trace Explorer</h1>
    <p>{source}</p>
    <nav><a href="#summary">Summary</a><a href="#capacity">Capacity</a><a href="#operations">Operations</a><a href="#checkpointing">Checkpointing</a><a href="#local-ssd">Local SSD</a><a href="#reliability">Reliability</a><a href="#priorities">Priorities</a><a href="#jobs">Per job</a><a href="#topology">Topology</a><a href="#spans">Slow spans</a></nav>
  </header>
  <main>
    <section id="summary" class="section-title"><h2>What happened?</h2><p>The short version first: completion, overhead, checkpoint policy, dominant work, and reliability outcomes.</p></section>
    <section class="findings">{_findings_panel(summary, context)}</section>
    <section class="metrics">{_cards(summary, context)}</section>
    <section class="panel full">{_aggregate_metrics_table(summary, context)}</section>

    <section id="capacity" class="section-title"><h2>Capacity and completion</h2><p>Allocated nodes are inferred from each job's first recorded activity through its final training span. This is separate from concurrent operation spans, which may overlap.</p></section>
    <section class="panel full">{rendered[0]}<p class="panel-note"><strong>Why it drops:</strong> a step down means one or more jobs completed and released their modeled nodes. It does not mean the simulator randomly disabled nodes.</p></section>
    <section class="panel full">{rendered[1]}<p class="panel-note">Completion times come from each job's final coalesced training span; background checkpoint drains can continue afterward.</p></section>

    <section id="operations" class="section-title"><h2>Where time went</h2><p>Aggregate node-time answers how much fleet work each operation consumed. It is not a wall-clock decomposition because asynchronous operations overlap.</p></section>
    <section class="panel full">{rendered[2]}<p class="panel-note">Node-hours = span duration × represented nodes. Nominal time comes from <code>details.base_duration</code>; excess is observed duration above nominal. Some cohort events lack a logical weight, so their aggregate work can be conservative.</p></section>
    <section class="panel full">{rendered[3]}<p class="panel-note">Percentiles use a deterministic bounded reservoir per operation. p95 describes individual-span latency, while node-hours describes aggregate cost.</p></section>

    <section id="checkpointing" class="section-title"><h2>Checkpoint policy and observed behavior</h2><p>Configured intent and trace-observed placement are shown separately.</p></section>
    <section class="explain-grid">{_checkpoint_panel(summary, context)}</section>
    <section class="panel full">{rendered[4]}</section>
    <section class="grid"><div class="panel">{rendered[5]}</div><div id="inventory" class="panel">{rendered[9]}</div></section>

    <section id="local-ssd" class="section-title"><h2>Owner-local checkpoint writes</h2><p>This view restores same-job DRAM → SSD traffic that the cross-service dependency map intentionally omits.</p></section>
    <section class="panel full">{rendered[10]}<p class="panel-note"><strong>What is counted:</strong> completed <code>checkpoint_dram_to_local_ssd_chunk</code> bytes, attributed to the logical checkpoint-owner job. Peer SSD writes, object-store traffic, and recovery reads are excluded. Values are cumulative physical write traffic—not unique checkpoint state or current SSD occupancy—so overwrites and replay after rollback count again.</p></section>

    <section id="reliability" class="section-title"><h2>Failures, rollback, and recovery</h2><p>The result sidecar provides incident counts; completed restore tails in the trace provide per-job recovery sources. One whole-job preemption can restore several cohorts, so restore count can exceed incident count.</p></section>
    <section class="panel full">{rendered[6]}<p class="panel-note">A failure timeline is not available because this simulation version did not emit standalone failure/restart spans. Aggregate incident counts remain exact in the result JSON.</p></section>

    <section id="priorities" class="section-title"><h2>Scheduling priorities and QoS</h2><p>The model has operation-level CPU priority, but this scenario has no business/job-priority scheduler.</p></section>
    <section class="explain-grid">{_priority_panel(summary)}</section>

    <section id="jobs" class="panel full">{_job_metrics_table(summary, context)}</section>

    <section id="activity" class="section-title"><h2>Recorded span occupancy</h2><p>This is diagnostic concurrency, not allocated-node count. Training and asynchronous checkpoint spans can overlap and therefore add more than one span for the same cohort.</p></section>
    <section class="panel full">{rendered[7]}<p class="panel-note">Values are time-weighted and scaled by <code>details.represents</code> where present. Dips can mean completion, recovery/wait gaps, or missing standalone wait spans.</p></section>
    <section class="panel full">{rendered[8]}</section>

    <section id="topology" class="section-title"><h2>Data movement topology</h2><p>Use the dependency map for structure and the ranked route chart for exact comparisons.</p></section>
    <section class="panel full">{rendered[11]}<p class="panel-note"><strong>How to read it:</strong> each ribbon runs from source to destination and its width is cumulative GB. Purple routes ending at labels such as <code>frontier (other jobs)</code> mean different jobs in the same class exchanged data. True same-job tier movements are omitted. This measures traffic volume, not latency.</p></section>
    <section id="routes" class="panel full">{rendered[12]}<p class="panel-note">Purple bars are same-class, cross-job routes. Percentages use every retained inter-job or inter-service route, even when only the top routes are displayed.</p></section>
    <section id="spans" class="panel full">
      <div class="trace-head"><h2>Slow-span explorer</h2><input id="span-search" placeholder="Filter service, operation, status…"><input id="min-duration" type="number" min="0" step="0.1" value="0" placeholder="Minimum seconds"></div>
      <div class="table-wrap"><table><thead><tr><th>Service</th><th>Kind</th><th>Operation</th><th>Start (s)</th><th>Duration (s)</th><th>Iteration</th><th>Rank/cohort</th><th>GB</th><th>Status</th></tr></thead><tbody id="span-rows">{_span_table(summary)}</tbody></table></div>
      <p class="panel-note">This bounded table contains the slowest {len(summary.top_spans):,} recorded spans, not every node-level event.</p>
    </section>
  </main>
  <footer>Generated from {source} · {summary.event_count:,} event records summarized without per-node lanes</footer>
  <script>
    function filterSpans() {{
      const query = document.getElementById('span-search').value.toLowerCase().trim();
      const minimum = Number(document.getElementById('min-duration').value || 0);
      document.querySelectorAll('#span-rows tr').forEach((row) => {{
        row.hidden = !(row.dataset.text.includes(query) && Number(row.dataset.duration) >= minimum);
      }});
    }}
    document.getElementById('span-search').addEventListener('input', filterSpans);
    document.getElementById('min-duration').addEventListener('input', filterSpans);
    document.getElementById('job-search').addEventListener('input', (event) => {{
      const query = event.target.value.toLowerCase().trim();
      document.querySelectorAll('#job-rows tr').forEach((row) => {{
        row.hidden = !row.dataset.job.includes(query);
      }});
    }});
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def generate_dashboard(
    log_path: Path,
    output_path: Path,
    *,
    buckets: int = 240,
    top_spans: int = 500,
    scenario_path: Path | None = None,
    result_path: Path | None = None,
    arm: str | None = None,
    seed: int | None = None,
) -> TraceSummary:
    context = load_run_context(
        scenario_path=scenario_path,
        result_path=result_path,
        arm=arm,
        seed=seed,
    )
    summary = summarize_trace(log_path, top_spans=top_spans)
    if summary.event_count == 0:
        raise ValueError(f"Trace contains no events: {log_path}")
    activity = aggregate_activity(log_path, summary, buckets=buckets)
    write_dashboard(
        summary,
        activity,
        output_path=output_path,
        source_log=log_path,
        context=context,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a service-oriented aggregate dashboard without per-node lanes."
        )
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("results/simulation_log.jsonl"),
        help="Plain or gzip-compressed simulator JSONL trace",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/simulation_aggregate.html"),
        help="Output standalone HTML dashboard",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=240,
        help="Number of time buckets (20-1000; default 240)",
    )
    parser.add_argument(
        "--top-spans",
        type=int,
        default=500,
        help="Number of slow spans embedded in the explorer (default 500)",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        help="Optional scenario YAML for policy, ideal runtime, and class metadata",
    )
    parser.add_argument(
        "--result",
        type=Path,
        help="Optional scenario result JSON for failures and recovery outcomes",
    )
    parser.add_argument("--arm", help="Arm to select from scenario/result metadata")
    parser.add_argument("--seed", type=int, help="Seed to select from result metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_dashboard(
        args.log,
        args.output,
        buckets=args.buckets,
        top_spans=args.top_spans,
        scenario_path=args.scenario,
        result_path=args.result,
        arm=args.arm,
        seed=args.seed,
    )
    print(
        f"Aggregate visualization written to {args.output} "
        f"({summary.event_count:,} records, "
        f"{summary.physical_nodes:,.0f} modeled nodes)"
    )


if __name__ == "__main__":
    main()
