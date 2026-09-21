from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation import run_simulation


GIB = 1024**3
SCHEMA = "gcp_2x2_sim_validation.v1"
EVENT_SCHEMA = "gcp_2x2_sim_event.v1"
BASELINE_RUN = "baseline-large-20260804T223437Z"
SYNC_SMOKE_RUN = "smoke-sync-20260804T225054Z"
ASYNC_SMOKE_RUN = "smoke-async-20260804T225407Z"
ARM_SPECS = {
    "baseline": {
        "label": "No checkpointing",
        "mode": "none",
        "every": 0,
        "real_run_id": BASELINE_RUN,
    },
    "sync_f3": {
        "label": "Synchronous SSD every 3 iterations",
        "mode": "sync",
        "every": 3,
        "real_run_id": "sync-f3-20260804T235437Z",
    },
    "sync_f5": {
        "label": "Synchronous SSD every 5 iterations",
        "mode": "sync",
        "every": 5,
        "real_run_id": "sync-f5-20260804T225934Z",
    },
    "async_f3": {
        "label": "Asynchronous SSD every 3 iterations",
        "mode": "async",
        "every": 3,
        "real_run_id": "async-f3-20260805T003038Z",
    },
    "async_f5": {
        "label": "Asynchronous SSD every 5 iterations",
        "mode": "async",
        "every": 5,
        "real_run_id": "async-f5-20260804T233123Z",
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = REPO_ROOT
    default_real = (
        repo_root.parent
        / "checkpointing"
        / "realgpu_coefficient"
        / "results"
        / "gcp"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the configured simulator from the GCP baseline and "
            "checkpoint smokes, then validate against held-out f3/f5 runs."
        )
    )
    parser.add_argument("--real-root", type=Path, default=default_real)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "gcp_two_node_validation",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: event is not an object")
            rows.append(value)
    return rows


def rank_logs(real_root: Path, run_id: str) -> list[Path]:
    paths = sorted(
        path
        for path in real_root.glob(f"node*/logs/{run_id}/{run_id}/*.jsonl")
        if path.name.startswith(("A_rank", "B_rank"))
    )
    if len(paths) != 4:
        raise FileNotFoundError(
            f"Expected four rank logs for {run_id}, found {len(paths)} under {real_root}"
        )
    return paths


def load_training_run(real_root: Path, run_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in rank_logs(real_root, run_id):
        rows = read_jsonl(path)
        for row in rows:
            if row.get("run_id") != run_id:
                raise ValueError(f"{path}: event has wrong run_id {row.get('run_id')!r}")
        events.extend(rows)
    return events


def selected(events: Iterable[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [row for row in events if row.get("event") == event_name]


def mean_ns(events: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) / 1e9 for row in events]
    if not values:
        raise ValueError(f"No values found for {key}")
    return statistics.mean(values)


def counter_at(rows: list[dict[str, Any]], timestamp: float) -> int:
    before = [row for row in rows if float(row["t"]) <= timestamp]
    selected_row = before[-1] if before else rows[0]
    return int(selected_row["tx"])


def measured_network_tx_gib(
    real_root: Path,
    run_id: str,
    training_events: list[dict[str, Any]],
) -> float:
    starts = [int(row["emitted_utc_ns"]) for row in selected(training_events, "measurement_start")]
    ends = [int(row["emitted_utc_ns"]) for row in selected(training_events, "training_end")]
    if not starts or not ends:
        raise ValueError(f"{run_id}: missing measurement_start/training_end")
    start_s = min(starts) / 1e9
    end_s = max(ends) / 1e9
    total = 0
    for path in sorted(real_root.glob(f"node*/logs/{run_id}/network.jsonl")):
        rows = [row for row in read_jsonl(path) if row.get("iface") == "ens7"]
        rows.sort(key=lambda row: float(row["t"]))
        if not rows:
            continue
        total += max(0, counter_at(rows, end_s) - counter_at(rows, start_s))
    if total <= 0:
        raise ValueError(f"{run_id}: no positive ens7 TX delta in measurement window")
    return total / GIB


def calibrate(real_root: Path) -> dict[str, Any]:
    baseline = load_training_run(real_root, BASELINE_RUN)
    sync_smoke = load_training_run(real_root, SYNC_SMOKE_RUN)
    async_smoke = load_training_run(real_root, ASYNC_SMOKE_RUN)

    training_ends = selected(baseline, "training_end")
    if len(training_ends) != 4:
        raise ValueError("Baseline must have four training_end events")
    iterations = int(training_ends[0]["measured_iterations"])
    iteration_seconds = statistics.mean(
        float(row["training_duration_ns"]) / 1e9 / iterations
        for row in training_ends
    )

    sync_commits = selected(sync_smoke, "checkpoint_commit")
    async_stages = selected(async_smoke, "checkpoint_staged")
    async_persists = selected(async_smoke, "checkpoint_persist_end")
    if len(sync_commits) != 4 or len(async_stages) != 8 or len(async_persists) != 8:
        raise ValueError(
            "Calibration smokes require 4 sync commits, 8 async stages, and 8 async persists"
        )

    sync_stage_seconds = mean_ns(sync_commits, "stage_or_sync_duration_ns")
    sync_total_seconds = mean_ns(sync_commits, "total_duration_ns")
    async_stage_seconds = mean_ns(async_stages, "stage_or_sync_duration_ns")
    async_persist_seconds = mean_ns(async_persists, "persist_duration_ns")
    occupancy_seconds = statistics.mean(
        (
            int(row["emitted_monotonic_ns"])
            - int(row["persist_start_monotonic_ns"])
        )
        / 1e9
        for row in async_persists
    )
    async_cleanup_seconds = max(0.0, occupancy_seconds - async_persist_seconds)
    checkpoint_bytes = statistics.mean(
        [int(row["bytes"]) for row in sync_commits]
        + [int(row["bytes"]) for row in async_persists]
    )
    checkpoint_gib = checkpoint_bytes / GIB

    network_tx_gib = measured_network_tx_gib(real_root, BASELINE_RUN, baseline)
    transfers = iterations * 2 * 2  # iterations × jobs × ranks
    network_gib_per_rank_iteration = network_tx_gib / transfers
    network_capacity_gib_s = 16e9 / 8 / GIB
    # Two jobs concurrently use each physical node NIC. The simulator's
    # max-min sharing doubles each collective's isolated transfer duration.
    all_reduce_seconds = (
        2.0 * network_gib_per_rank_iteration / network_capacity_gib_s
    )
    residual = iteration_seconds - all_reduce_seconds
    if residual <= 0:
        raise ValueError(
            "Measured NIC volume leaves no positive compute time; check counter window"
        )

    return {
        "source_run_ids": [BASELINE_RUN, SYNC_SMOKE_RUN, ASYNC_SMOKE_RUN],
        "held_out_run_ids": [
            ARM_SPECS[key]["real_run_id"]
            for key in ("sync_f3", "sync_f5", "async_f3", "async_f5")
        ],
        "strict_holdout": True,
        "warmups_excluded": 10,
        "measured_iterations": iterations,
        "iteration_seconds": iteration_seconds,
        "sync_stage_seconds": sync_stage_seconds,
        "sync_total_seconds": sync_total_seconds,
        "async_stage_seconds": async_stage_seconds,
        "async_persist_seconds": async_persist_seconds,
        "async_worker_occupancy_seconds": occupancy_seconds,
        "async_cleanup_seconds": async_cleanup_seconds,
        "checkpoint_bytes_per_rank": checkpoint_bytes,
        "checkpoint_gib_per_rank": checkpoint_gib,
        "baseline_network_tx_gib": network_tx_gib,
        "network_gib_per_rank_iteration": network_gib_per_rank_iteration,
        "physical_nic_capacity_gbps": 16.0,
        "physical_nic_capacity_gib_s": network_capacity_gib_s,
        "all_reduce_seconds_with_two_job_share": all_reduce_seconds,
        "forward_seconds": residual * 0.30,
        "backward_seconds": residual * 0.60,
        "optimizer_seconds": residual * 0.10,
        "notes": [
            "Only the baseline and one-checkpoint sync/two-checkpoint async smokes calibrate coefficients.",
            "The 100-iteration cadence-3/cadence-5 runs are held out.",
            "Training operation decomposition is inferred from aggregate ens7 TX; no CUDA/NCCL profiler was collected.",
            "Async durable service ends at persist_end; worker occupancy additionally includes cleanup/pruning through event emission.",
        ],
    }


def config_text(arm: dict[str, Any], calibration: dict[str, Any]) -> str:
    mode = str(arm["mode"])
    cadence = int(arm["every"]) if int(arm["every"]) > 0 else 101
    checkpoint_gib = float(calibration["checkpoint_gib_per_rank"])
    if mode == "sync":
        stage_seconds = float(calibration["sync_stage_seconds"])
        disk_seconds = float(calibration["sync_total_seconds"]) - stage_seconds
        cleanup_seconds = 0.0
        checkpoint_mode = "synchronous"
        backpressure = "false"
    elif mode == "async":
        stage_seconds = float(calibration["async_stage_seconds"])
        disk_seconds = float(calibration["async_persist_seconds"])
        cleanup_seconds = float(calibration["async_cleanup_seconds"])
        checkpoint_mode = "asynchronous"
        backpressure = "true"
    else:
        stage_seconds = float(calibration["async_stage_seconds"])
        disk_seconds = float(calibration["async_persist_seconds"])
        cleanup_seconds = 0.0
        checkpoint_mode = "synchronous"
        backpressure = "false"

    gpu_cpu_gib_s = checkpoint_gib / stage_seconds
    # There are two simultaneous writers per node. Shared-local-SSD fairness
    # gives each stream half of this physical aggregate rate.
    local_ssd_gib_s = 2.0 * checkpoint_gib / disk_seconds
    model_weight_gib = 774_030_080 * 2 / GIB

    job_blocks: list[str] = []
    for job_id in ("A", "B"):
        job_blocks.append(
            f'''[[jobs]]
id = "{job_id}"
strategy = "data_parallel"
failure_rate = 0.0
data_parallel_replicas = 2
pipeline_stages = 1
microbatches = 4

[jobs.model]
weights_gb = {model_weight_gib:.12f}
gradient_gb = {float(calibration["network_gib_per_rank_iteration"]):.12f}
activation_gb = 1.0

[jobs.timing]
forward_seconds = {float(calibration["forward_seconds"]):.12f}
backward_seconds = {float(calibration["backward_seconds"]):.12f}
optimizer_seconds = {float(calibration["optimizer_seconds"]):.12f}

[jobs.checkpoint]
strategy = "local_tiered"
mode = "{checkpoint_mode}"
every = {cadence}
size_gb = {checkpoint_gib:.12f}
chunk_gb = {checkpoint_gib:.12f}
upload_gpu_slowdown = 1.0
backpressure = {backpressure}
cleanup_seconds = {cleanup_seconds:.12f}
'''
        )

    return f'''[simulation]
seed = 17
iterations = 100

[cluster]
node_count = 2
cpu_cores_per_node = 24
cpu_memory_gb = 153.0
gpus_per_node = 2
gpu_memory_gb = 16.0
gpu_cpu_bandwidth_gbps = {gpu_cpu_gib_s:.12f}
network_bandwidth_gbps = {float(calibration["physical_nic_capacity_gib_s"]):.12f}
communication_launch_seconds = 0.000001
local_ssd_bandwidth_gbps = {local_ssd_gib_s:.12f}
object_store_bandwidth_gbps = 1.0
object_store_concurrency = 4

[failures]
probability_per_second = 0.0
process_weight = 1.0
node_weight = 0.0
spot_weight = 0.0
process_restart_seconds = 3.0
node_restart_seconds = 8.0
spot_restart_seconds = 15.0

{''.join(job_blocks)}'''


def event_kind(operation: str, category: str) -> str:
    if operation in {
        "data_parallel_forward",
        "data_parallel_backward",
        "data_parallel_optimizer",
    }:
        return "training_compute"
    if operation == "data_parallel_all_reduce":
        return "training_communication"
    if operation == "checkpoint_stage_gpu_to_dram":
        return "checkpoint_stage"
    if "local_ssd" in operation:
        return "checkpoint_persist"
    if operation in {"checkpoint_async_backpressure", "checkpoint_async_cleanup"}:
        return "wait"
    if category == "Failure":
        return "failure"
    if category == "Recovery":
        return "recovery"
    if category == "Synchronization":
        return "wait"
    return "other"


def enrich_trace(
    native_path: Path,
    output_path: Path,
    arm_key: str,
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    placement = {
        (str(row["job_id"]), int(row["rank"])): row for row in placements
    }
    by_job_rank = {
        str(row["job_id"]): {
            int(candidate["rank"]): candidate
            for candidate in placements
            if candidate["job_id"] == row["job_id"]
        }
        for row in placements
    }
    canonical: list[dict[str, Any]] = []
    for row in read_jsonl(native_path):
        job_id = str(row.get("job_id") or "unknown")
        rank = int(row.get("rank") or 0)
        operation = str(row.get("operation") or "unknown")
        details = dict(row.get("details") or {})
        mapped = placement.get((job_id, rank), {})
        event: dict[str, Any] = {
            "schema": EVENT_SCHEMA,
            "arm": arm_key,
            "event_id": int(row.get("event_id") or len(canonical) + 1),
            "start": float(row["start"]),
            "end": float(row["end"]),
            "duration": float(row["duration"]),
            "job_id": job_id,
            "rank": rank,
            "physical_node": str(row.get("physical_node") or mapped.get("physical_node") or "unknown"),
            "gpu_index": int(mapped.get("gpu_index", 0)),
            "iteration": int(row.get("iteration") or 0),
            "category": str(row.get("category") or "Other"),
            "operation": operation,
            "resources": list(row.get("resources") or []),
            "source": row.get("source"),
            "destination": row.get("destination"),
            "data_gib": row.get("data_gb"),
            "kind": event_kind(operation, str(row.get("category") or "")),
            "details": details,
        }
        data_gib = float(row.get("data_gb") or 0.0)
        if operation == "data_parallel_all_reduce" and data_gib > 0:
            peers = by_job_rank.get(job_id, {})
            peer = peers.get(1 - rank, {})
            event["network"] = {
                "source": event["physical_node"],
                "destination": peer.get("physical_node", row.get("destination") or "peer"),
                "gib": data_gib,
                "contention_s": float(details.get("bandwidth_contention_seconds") or 0.0),
            }
        if "local_ssd" in operation and data_gib > 0:
            event["disk"] = {
                "host": event["physical_node"],
                "tier": "local-ssd",
                "direction": "write",
                "gib": data_gib,
                "contention_s": float(details.get("bandwidth_contention_seconds") or 0.0),
            }
        canonical.append(event)

    canonical.sort(key=lambda row: (row["start"], row["end"], row["event_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in canonical:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")
    return canonical


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def simulation_metrics(events: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    training_kinds = {"training_compute", "training_communication"}
    iteration_spans: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for event in events:
        if event["kind"] not in training_kinds or int(event["iteration"]) <= 0:
            continue
        iteration_spans.setdefault((event["job_id"], int(event["iteration"])), []).append(
            (float(event["start"]), float(event["end"]))
        )
    durations = [
        max(end for _, end in spans) - min(start for start, _ in spans)
        for spans in iteration_spans.values()
    ]

    training_end_candidates = [
        float(event["end"])
        for event in events
        if event["kind"] in training_kinds | {"checkpoint_stage"}
        or (mode == "sync" and event["kind"] == "checkpoint_persist")
    ]
    training_completion = max(training_end_candidates, default=0.0)
    persist_ends = [
        float(event["end"])
        for event in events
        if event["kind"] == "checkpoint_persist"
    ]
    durable_completion = max([training_completion, *persist_ends])
    logical = {
        (event["job_id"], int(event["iteration"]))
        for event in events
        if event["kind"] == "checkpoint_stage"
    }
    writes = {
        (event["job_id"], int(event["rank"]), int(event["iteration"]))
        for event in events
        if event["kind"] == "checkpoint_persist"
    }
    checkpoint_gib = sum(
        float(event.get("data_gib") or 0.0)
        for event in events
        if event["kind"] == "checkpoint_persist"
    )
    network_gib = sum(
        float((event.get("network") or {}).get("gib") or 0.0)
        for event in events
    )
    return {
        "training_completion_s": training_completion,
        "durable_completion_s": durable_completion,
        "goodput_pct": None,
        "iteration_p50_s": percentile(durations, 0.50),
        "iteration_p95_s": percentile(durations, 0.95),
        "logical_checkpoints": len(logical),
        "physical_writes": len(writes),
        "checkpoint_gib": checkpoint_gib,
        "network_gib": network_gib,
        "simulated_harness_finish_s": max(
            (float(event["end"]) for event in events), default=training_completion
        ),
    }


def real_metrics(real_root: Path, run_id: str) -> dict[str, Any]:
    events = load_training_run(real_root, run_id)
    starts = selected(events, "measurement_start")
    training_ends = selected(events, "training_end")
    if len(starts) != 4 or len(training_ends) != 4:
        raise ValueError(f"{run_id}: incomplete measurement boundary events")
    origin = min(int(row["emitted_utc_ns"]) for row in starts)
    training_end = max(int(row["emitted_utc_ns"]) for row in training_ends)
    durable_end = training_end
    commits = selected(events, "checkpoint_commit")
    persists = selected(events, "checkpoint_persist_end")
    if commits:
        durable_end = max(
            durable_end,
            max(int(row["emitted_utc_ns"]) for row in commits),
        )
    if persists:
        durable_end = max(
            durable_end,
            max(int(row["persist_end_utc_ns"]) for row in persists),
        )
    iteration_durations = [
        float(row["duration_ns"]) / 1e9
        for row in selected(events, "iteration")
        if row.get("phase") == "measure"
    ]
    logical = {
        (str(row["job_id"]), int(row["measured_iteration"]))
        for row in [*commits, *persists]
    }
    checkpoint_bytes = sum(int(row["bytes"]) for row in [*commits, *persists])
    return {
        "training_completion_s": (training_end - origin) / 1e9,
        "durable_completion_s": (durable_end - origin) / 1e9,
        "iteration_p50_s": percentile(iteration_durations, 0.50),
        "iteration_p95_s": percentile(iteration_durations, 0.95),
        "logical_checkpoints": len(logical),
        "physical_writes": len(commits) + len(persists),
        "checkpoint_gib": checkpoint_bytes / GIB,
    }


def error_pct(simulated: float, real: float) -> float:
    return 100.0 * (simulated - real) / real if real else 0.0


def main() -> int:
    args = parse_args()
    real_root = args.real_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = calibrate(real_root)
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest_runs: dict[str, Any] = {}
    for run_id, (arm_key, arm) in enumerate(ARM_SPECS.items(), start=1):
        arm_dir = output_dir / arm_key
        config_path = arm_dir / "simulator.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_text(arm, calibration), encoding="utf-8")
        result = run_simulation(
            config_path,
            results_dir=arm_dir / "native",
            run_id=run_id,
            quiet=True,
        )
        if result.event_log_path is None:
            raise RuntimeError(f"{arm_key}: simulator did not write an event log")
        canonical_path = arm_dir / "simulated_trace.jsonl"
        events = enrich_trace(
            result.event_log_path,
            canonical_path,
            arm_key,
            result.placements,
        )
        sim = simulation_metrics(events, str(arm["mode"]))
        real = real_metrics(real_root, str(arm["real_run_id"]))
        validation = {
            "status": (
                "calibration_source"
                if str(arm["real_run_id"]) in calibration["source_run_ids"]
                else "held_out"
            ),
            "training_completion_error_pct": error_pct(
                sim["training_completion_s"], real["training_completion_s"]
            ),
            "durable_completion_error_pct": error_pct(
                sim["durable_completion_s"], real["durable_completion_s"]
            ),
            "real_metrics": real,
            "errors": [],
        }
        manifest_runs[arm_key] = {
            "arm": arm["label"],
            "mode": arm["mode"],
            "checkpoint_every": arm["every"],
            "real_run_id": arm["real_run_id"],
            "event_log": str(canonical_path.relative_to(output_dir)),
            "config": str(config_path.relative_to(output_dir)),
            "metrics": sim,
            "validation": validation,
        }

    baseline_time = float(manifest_runs["baseline"]["metrics"]["training_completion_s"])
    for spec in manifest_runs.values():
        completion = float(spec["metrics"]["training_completion_s"])
        spec["metrics"]["goodput_pct"] = 100.0 * baseline_time / completion

    manifest = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "topology": {
            "nodes": 2,
            "gpus_per_node": 2,
            "gpu": "Tesla V100-SXM2-16GB",
            "cpu_cores_per_node": 24,
            "ram_gib_per_node": 153.0,
            "local_ssd_gb_per_node": 375,
            "jobs": {
                "A": "GPU 0 on both nodes",
                "B": "GPU 1 on both nodes",
            },
            "measured_iterations": 100,
            "warmups_before_measurement": 10,
            "failures": "disabled in completed real arms",
        },
        "calibration": calibration,
        "runs": manifest_runs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "runs": len(manifest_runs),
                "held_out": {
                    key: value["validation"]
                    for key, value in manifest_runs.items()
                    if value["validation"]["status"] == "held_out"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
