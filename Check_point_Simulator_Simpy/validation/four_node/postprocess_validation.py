from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


GB_PER_MB = 1 / 1024
IDLE_GAP_THRESHOLD_SECONDS = 0.001
CORE_REAL_OPERATIONS = {
    "data_parallel_forward",
    "data_parallel_backward",
    "data_parallel_all_reduce",
    "data_parallel_optimizer",
    "pipeline_activation_receive",
    "pipeline_activation_send",
    "pipeline_forward",
    "pipeline_gradient_receive",
    "pipeline_backward",
    "pipeline_gradient_send",
    "pipeline_optimizer",
    "checkpoint_stage_gpu_to_dram",
    "checkpoint_stage_dram_to_object_store",
}
REAL_JOB_IDS = {
    ("data_parallel", "synchronous"): "real-data-parallel-sync",
    ("pipeline_parallel", "synchronous"): "real-pipeline-parallel-sync",
    ("data_parallel", "asynchronous"): "real-data-parallel-async",
    ("pipeline_parallel", "asynchronous"): "real-pipeline-parallel-async",
}
LEGACY_REAL_JOB_IDS = {
    "data_parallel": "real-data-parallel",
    "pipeline_parallel": "real-pipeline-parallel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build simulator inputs and side-by-side validation HTML."
    )
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--microbatches", type=int, required=True)
    parser.add_argument("--run-simulator", action="store_true")
    return parser.parse_args()


def load_jsonl_files(root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        if path.name in {"real_trace.jsonl", "simulated_trace.jsonl"}:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    if not events:
        raise FileNotFoundError(f"No JSONL traces were found under {root}")

    events.sort(
        key=lambda event: (
            str(event.get("job_id") or ""),
            float(event.get("start") or 0.0),
            float(event.get("end") or 0.0),
            int(event.get("rank") or 0),
            int(event.get("event_id") or 0),
        )
    )
    for index, event in enumerate(events, start=1):
        event["event_id"] = index
    return events


def write_jsonl(events: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")


def clone_reindexed(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned = [dict(event) for event in events]
    for index, event in enumerate(cloned, start=1):
        event["event_id"] = index
    return cloned


def comparison_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [
        event for event in events if str(event.get("operation") or "") in CORE_REAL_OPERATIONS
    ]
    if not filtered:
        return clone_reindexed(events)
    return clone_reindexed(filtered)


def load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    if not events:
        raise ValueError(f"No events were found in {path}")
    return events


def idle_reason(previous_operation: str, next_operation: str) -> str:
    if "barrier" in next_operation:
        return "waiting_for_other_ranks_at_synchronization_barrier"
    if next_operation in {"pipeline_activation_receive", "pipeline_gradient_receive"}:
        return "waiting_for_neighbor_pipeline_stage_to_send_tensor"
    if "receive" in next_operation:
        return "waiting_for_remote_rank_or_network_receive"
    if "all_reduce" in next_operation:
        return "waiting_for_data_parallel_collective_to_start"
    if "checkpoint" in next_operation:
        return "checkpoint_cpu_disk_or_object_store_path"
    if "iteration_start" in next_operation and "complete" in previous_operation:
        return "between_iterations_after_rank_completed_previous_work"
    if next_operation.startswith("benchmark_"):
        return "benchmark_setup_or_teardown_gap"
    return "no_pytorch_event_recorded_on_this_rank_during_gap"


def idle_analysis(events: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        rank = event.get("rank")
        if rank is None:
            continue
        grouped[(str(event.get("job_id") or "unknown"), int(rank))].append(event)

    gaps: list[dict[str, Any]] = []
    total_gap_by_reason: dict[str, float] = defaultdict(float)
    for (job_id, rank), rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row.get("start") or 0.0),
                float(row.get("end") or 0.0),
                int(row.get("event_id") or 0),
            ),
        )
        if len(ordered) < 2:
            continue
        previous = ordered[0]
        previous_end = float(previous.get("end") or 0.0)
        for current in ordered[1:]:
            current_start = float(current.get("start") or 0.0)
            if current_start > previous_end + IDLE_GAP_THRESHOLD_SECONDS:
                previous_operation = str(previous.get("operation") or "unknown")
                next_operation = str(current.get("operation") or "unknown")
                reason = idle_reason(previous_operation, next_operation)
                duration = current_start - previous_end
                gaps.append(
                    {
                        "job_id": job_id,
                        "rank": rank,
                        "start": round(previous_end, 9),
                        "end": round(current_start, 9),
                        "duration": round(duration, 9),
                        "previous_operation": previous_operation,
                        "next_operation": next_operation,
                        "likely_reason": reason,
                    }
                )
                total_gap_by_reason[reason] += duration
            previous = current
            previous_end = max(previous_end, float(current.get("end") or 0.0))

    gaps.sort(key=lambda row: float(row["duration"]), reverse=True)
    return {
        "threshold_seconds": IDLE_GAP_THRESHOLD_SECONDS,
        "gap_count": len(gaps),
        "total_gap_seconds": round(sum(float(row["duration"]) for row in gaps), 9),
        "total_gap_seconds_by_reason": {
            reason: round(duration, 9)
            for reason, duration in sorted(
                total_gap_by_reason.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "largest_gaps": gaps[:200],
    }


def load_summaries(root: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.rglob("*.summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        key = str(
            summary.get("job_id")
            or REAL_JOB_IDS.get(
                (
                    str(summary["mode"]),
                    str(summary.get("checkpoint_mode", "synchronous")),
                )
            )
            or LEGACY_REAL_JOB_IDS[str(summary["mode"])]
        )
        grouped[key].append(summary)
    if not grouped:
        raise FileNotFoundError(f"No summary files were found under {root}")
    return {
        key: sorted(rows, key=lambda row: int(row.get("rank", 0)))[0]
        for key, rows in grouped.items()
    }


def values(
    events: list[dict[str, Any]],
    *,
    job_id: str,
    operation: str,
) -> list[float]:
    return [
        float(event["duration"])
        for event in events
        if event.get("job_id") == job_id
        and event.get("operation") == operation
        and float(event.get("duration", 0.0)) > 0.0
    ]


def data_values(
    events: list[dict[str, Any]],
    *,
    job_id: str,
    operation: str,
) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for event in events:
        if event.get("job_id") != job_id or event.get("operation") != operation:
            continue
        duration = float(event.get("duration") or 0.0)
        data_gb = float(event.get("data_gb") or 0.0)
        if duration > 0.0 and data_gb > 0.0:
            rows.append((data_gb, duration))
    return rows


def mean_or_default(samples: list[float], default: float) -> float:
    return statistics.mean(samples) if samples else default


def median_bandwidth(samples: list[tuple[float, float]], default: float) -> float:
    bandwidths = [data_gb / duration for data_gb, duration in samples if duration > 0]
    return statistics.median(bandwidths) if bandwidths else default


def all_reduce_bandwidth(
    events: list[dict[str, Any]],
    *,
    job_id: str,
    replicas: int,
    default: float,
) -> float:
    samples = data_values(events, job_id=job_id, operation="data_parallel_all_reduce")
    bandwidths = [
        2 * (replicas - 1) / replicas * data_gb / duration
        for data_gb, duration in samples
        if duration > 0.0 and replicas > 1
    ]
    return statistics.median(bandwidths) if bandwidths else default


def object_store_bandwidth(events: list[dict[str, Any]], default: float) -> float:
    samples = [
        (float(event.get("data_gb") or 0.0), float(event.get("duration") or 0.0))
        for event in events
        if event.get("operation") == "checkpoint_stage_dram_to_object_store"
    ]
    bandwidths = [data_gb / duration for data_gb, duration in samples if duration > 0]
    return statistics.median(bandwidths) if bandwidths else default


def gpu_cpu_bandwidth(events: list[dict[str, Any]], default: float) -> float:
    samples = [
        (float(event.get("data_gb") or 0.0), float(event.get("duration") or 0.0))
        for event in events
        if event.get("operation") == "checkpoint_stage_gpu_to_dram"
    ]
    bandwidths = [data_gb / duration for data_gb, duration in samples if duration > 0]
    return statistics.median(bandwidths) if bandwidths else default


def summary_for(
    summaries: dict[str, dict[str, Any]],
    mode: str,
    checkpoint_mode: str,
) -> dict[str, Any]:
    job_id = REAL_JOB_IDS[(mode, checkpoint_mode)]
    if job_id in summaries:
        return summaries[job_id]
    legacy = LEGACY_REAL_JOB_IDS[mode]
    return summaries.get(legacy, {})


def real_job_id_for_events(
    events: list[dict[str, Any]],
    mode: str,
    checkpoint_mode: str,
) -> str:
    job_id = REAL_JOB_IDS[(mode, checkpoint_mode)]
    if any(event.get("job_id") == job_id for event in events):
        return job_id
    return LEGACY_REAL_JOB_IDS[mode]


def summary_float(
    summaries: dict[str, dict[str, Any]],
    path: tuple[str, ...],
    default: float,
) -> float:
    values_: list[float] = []
    for summary in summaries.values():
        value: Any = summary
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is None:
            continue
        try:
            values_.append(float(value))
        except (TypeError, ValueError):
            continue
    return statistics.median(values_) if values_ else default


def summary_int(
    summaries: dict[str, dict[str, Any]],
    key: str,
    default: int,
) -> int:
    values_: list[int] = []
    for summary in summaries.values():
        try:
            values_.append(int(summary[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return int(statistics.median(values_)) if values_ else default


def checkpoint_mode_slug(checkpoint_mode: str) -> str:
    return "sync" if checkpoint_mode == "synchronous" else "async"


def positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def median_positive(values_: list[float], default: float) -> float:
    positives = [value for value in values_ if value > 0.0]
    return statistics.median(positives) if positives else default


def first_median_positive(sources: list[list[float]], default: float) -> float:
    for values_ in sources:
        positives = [value for value in values_ if value > 0.0]
        if positives:
            return statistics.median(positives)
    return default


def trace_data_gb(
    events: list[dict[str, Any]],
    *,
    job_id: str,
    operations: set[str],
) -> list[float]:
    values_: list[float] = []
    for event in events:
        if event.get("job_id") != job_id:
            continue
        if str(event.get("operation") or "") not in operations:
            continue
        value = positive_float(event.get("data_gb"))
        if value is not None:
            values_.append(value)
    return values_


def trace_detail_gb(
    events: list[dict[str, Any]],
    *,
    job_id: str,
    operations: set[str],
    keys: tuple[str, ...],
) -> list[float]:
    values_: list[float] = []
    for event in events:
        if event.get("job_id") != job_id:
            continue
        if str(event.get("operation") or "") not in operations:
            continue
        details = event.get("details") or {}
        if not isinstance(details, dict):
            continue
        for key in keys:
            value = positive_float(details.get(key))
            if value is not None:
                values_.append(value)
                break
    return values_


def summary_size(summary: dict[str, Any], key: str, default: float) -> float:
    value = positive_float(summary.get(key))
    return value if value is not None else default


def infer_checkpoint_every(
    events: list[dict[str, Any]],
    *,
    job_id: str,
    summary: dict[str, Any],
) -> int:
    value = summary.get("checkpoint_every")
    try:
        configured = int(value)
    except (TypeError, ValueError):
        configured = 0
    if configured > 0:
        return configured

    iterations = sorted(
        {
            int(event["iteration"])
            for event in events
            if event.get("job_id") == job_id
            and event.get("operation") == "checkpoint_stage_gpu_to_dram"
            and event.get("iteration") is not None
        }
    )
    if len(iterations) >= 2:
        gaps = [b - a for a, b in zip(iterations, iterations[1:]) if b > a]
        if gaps:
            return min(gaps)
    if iterations and iterations[0] > 0:
        return iterations[0]
    return 1


def simulator_job_block(
    *,
    mode: str,
    checkpoint_mode: str,
    nodes: int,
    microbatches: int,
    events: list[dict[str, Any]],
    sizing_events: list[dict[str, Any]],
    summary: dict[str, Any],
    checkpoint_every: int,
) -> str:
    real_job_id = real_job_id_for_events(sizing_events, mode, checkpoint_mode)
    sim_kind = "data-parallel" if mode == "data_parallel" else "pipeline-parallel"
    sim_id = (
        f"sim-{sim_kind}-{checkpoint_mode_slug(checkpoint_mode)}-from-real"
    )
    sim_checkpoint_every = checkpoint_every if checkpoint_every > 0 else 1_000_000

    model_setup_ops = {
        "data_parallel_setup_complete",
        "pipeline_setup_complete",
    }
    checkpoint_shard_gb = median_positive(
        trace_data_gb(
            sizing_events,
            job_id=real_job_id,
            operations={"checkpoint_stage_gpu_to_dram"},
        ),
        summary_size(summary, "checkpoint_gb", 16 * GB_PER_MB),
    )

    if mode == "data_parallel":
        forward_seconds = mean_or_default(
            values(events, job_id=real_job_id, operation="data_parallel_forward"),
            0.01,
        )
        backward_seconds = mean_or_default(
            values(events, job_id=real_job_id, operation="data_parallel_backward"),
            0.01,
        )
        optimizer_seconds = mean_or_default(
            values(events, job_id=real_job_id, operation="data_parallel_optimizer"),
            0.01,
        )
        weights_gb = median_positive(
            trace_detail_gb(
                sizing_events,
                job_id=real_job_id,
                operations=model_setup_ops,
                keys=("model_parameter_gb", "gradient_gb"),
            ),
            summary_size(summary, "model_parameter_gb", 0.01),
        )
        gradient_gb = first_median_positive(
            [
                trace_data_gb(
                    sizing_events,
                    job_id=real_job_id,
                    operations={"data_parallel_all_reduce"},
                ),
                trace_detail_gb(
                    sizing_events,
                    job_id=real_job_id,
                    operations=model_setup_ops,
                    keys=("gradient_gb",),
                ),
            ],
            summary_size(summary, "gradient_gb", weights_gb),
        )
        activation_gb = first_median_positive(
            [
                trace_data_gb(
                    sizing_events,
                    job_id=real_job_id,
                    operations={"data_parallel_input_allocate"},
                ),
                trace_detail_gb(
                    sizing_events,
                    job_id=real_job_id,
                    operations=model_setup_ops,
                    keys=("tensor_gb", "activation_gb"),
                ),
            ],
            summary_size(summary, "tensor_gb", 0.01),
        )
        checkpoint_gb = max(checkpoint_shard_gb, 0.001)
        data_parallel_replicas = nodes
        pipeline_stages = 1
        job_microbatches = 1
        strategy = "data_parallel"
    else:
        pipeline_forward_stage = mean_or_default(
            values(events, job_id=real_job_id, operation="pipeline_forward"),
            0.01,
        )
        pipeline_backward_stage = mean_or_default(
            values(events, job_id=real_job_id, operation="pipeline_backward"),
            0.01,
        )
        pipeline_optimizer_stage = mean_or_default(
            values(events, job_id=real_job_id, operation="pipeline_optimizer"),
            0.01,
        )
        forward_seconds = max(pipeline_forward_stage * nodes * microbatches, 0.000001)
        backward_seconds = max(pipeline_backward_stage * nodes * microbatches, 0.000001)
        optimizer_seconds = max(pipeline_optimizer_stage * nodes, 0.000001)
        per_stage_weights_gb = median_positive(
            trace_detail_gb(
                sizing_events,
                job_id=real_job_id,
                operations=model_setup_ops,
                keys=("model_parameter_gb",),
            ),
            summary_size(summary, "model_parameter_gb", 0.01),
        )
        per_stage_gradient_gb = median_positive(
            trace_detail_gb(
                sizing_events,
                job_id=real_job_id,
                operations=model_setup_ops,
                keys=("gradient_gb", "model_parameter_gb"),
            ),
            summary_size(summary, "gradient_gb", per_stage_weights_gb),
        )
        weights_gb = per_stage_weights_gb * nodes
        gradient_gb = per_stage_gradient_gb * nodes
        per_microbatch_activation_gb = first_median_positive(
            [
                trace_data_gb(
                    sizing_events,
                    job_id=real_job_id,
                    operations={
                        "pipeline_activation_send",
                        "pipeline_activation_receive",
                        "pipeline_gradient_send",
                        "pipeline_gradient_receive",
                    },
                ),
                trace_detail_gb(
                    sizing_events,
                    job_id=real_job_id,
                    operations=model_setup_ops,
                    keys=("tensor_gb", "activation_gb"),
                ),
            ],
            summary_size(summary, "tensor_gb", 0.01),
        )
        activation_gb = max(per_microbatch_activation_gb * microbatches, 0.01)
        checkpoint_gb = max(checkpoint_shard_gb * nodes, 0.001)
        data_parallel_replicas = 1
        pipeline_stages = nodes
        job_microbatches = microbatches
        strategy = "pipeline_parallel"

    return f"""
[[jobs]]
id = "{sim_id}"
strategy = "{strategy}"
failure_rate = 0.0
data_parallel_replicas = {data_parallel_replicas}
pipeline_stages = {pipeline_stages}
microbatches = {job_microbatches}

[jobs.model]
weights_gb = {weights_gb:.9f}
gradient_gb = {gradient_gb:.9f}
activation_gb = {activation_gb:.9f}

[jobs.timing]
forward_seconds = {max(forward_seconds, 0.000001):.9f}
backward_seconds = {max(backward_seconds, 0.000001):.9f}
optimizer_seconds = {max(optimizer_seconds, 0.000001):.9f}

[jobs.checkpoint]
strategy = "object_store"
mode = "{checkpoint_mode}"
every = {sim_checkpoint_every}
size_gb = {checkpoint_gb:.9f}
upload_gpu_slowdown = 1.0
"""


def build_simulator_config(
    *,
    events: list[dict[str, Any]],
    sizing_events: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    output_path: Path,
    nodes: int,
    iterations: int,
    microbatches: int,
) -> None:
    network_samples = []
    for checkpoint_mode in ("synchronous", "asynchronous"):
        data_job_id = real_job_id_for_events(
            events,
            "data_parallel",
            checkpoint_mode,
        )
        pipeline_job_id = real_job_id_for_events(
            events,
            "pipeline_parallel",
            checkpoint_mode,
        )
        network_samples.extend(
            [
                all_reduce_bandwidth(
                    events,
                    job_id=data_job_id,
                    replicas=nodes,
                    default=1.0,
                ),
                median_bandwidth(
                    data_values(
                        events,
                        job_id=pipeline_job_id,
                        operation="pipeline_activation_send",
                    ),
                    default=1.0,
                ),
                median_bandwidth(
                    data_values(
                        events,
                        job_id=pipeline_job_id,
                        operation="pipeline_gradient_send",
                    ),
                    default=1.0,
                ),
            ]
        )
    network_gbps = max(
        0.001,
        statistics.median(network_samples),
    )
    object_store_gbps = max(0.001, object_store_bandwidth(events, default=1.0))
    gpu_cpu_gbps = max(0.001, gpu_cpu_bandwidth(events, default=12.5))
    cpu_cores_per_node = max(1, summary_int(summaries, "cpu_cores", 4))
    cpu_memory_gb = max(
        0.001,
        summary_float(summaries, ("cpu_memory", "total_gb"), 15.0),
    )
    gpu_memory_gb = max(
        0.001,
        summary_float(summaries, ("gpu_memory", "total_gb"), 16.0),
    )
    job_blocks = [
        simulator_job_block(
            mode=mode,
            checkpoint_mode=checkpoint_mode,
            nodes=nodes,
            microbatches=microbatches,
            events=events,
            sizing_events=sizing_events,
            summary=summary_for(summaries, mode, checkpoint_mode),
            checkpoint_every=infer_checkpoint_every(
                sizing_events,
                job_id=real_job_id_for_events(
                    sizing_events,
                    mode,
                    checkpoint_mode,
                ),
                summary=summary_for(summaries, mode, checkpoint_mode),
            ),
        )
        for checkpoint_mode in ("synchronous", "asynchronous")
        for mode in ("data_parallel", "pipeline_parallel")
    ]

    config = f"""[simulation]
seed = 17
iterations = {iterations}

[cluster]
node_count = {nodes * 4}
cpu_cores_per_node = {cpu_cores_per_node}
cpu_memory_gb = {cpu_memory_gb:.9f}
gpus_per_node = 1
gpu_memory_gb = {gpu_memory_gb:.9f}
gpu_cpu_bandwidth_gbps = {gpu_cpu_gbps:.9f}
network_bandwidth_gbps = {network_gbps:.9f}
communication_launch_seconds = 0.000001
local_ssd_bandwidth_gbps = 1.0
object_store_bandwidth_gbps = {object_store_gbps:.9f}
object_store_concurrency = {nodes * 4}

[failures]
probability_per_second = 0.0
process_weight = 1.0
node_weight = 0.0
spot_weight = 0.0
process_restart_seconds = 1.0
node_restart_seconds = 1.0
spot_restart_seconds = 1.0
{''.join(job_blocks)}
"""
    output_path.write_text(config, encoding="utf-8")


def run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def generate_visualization(repo_root: Path, log_path: Path, output_path: Path) -> None:
    run_command(
        [
            sys.executable,
            str(repo_root / "visualise.py"),
            "--log",
            str(log_path),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
    )


def run_simulator(repo_root: Path, config_path: Path, output_dir: Path) -> Path:
    workdir = output_dir / "simulator_workdir"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    run_command(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--config",
            str(config_path),
            "--quiet",
        ],
        cwd=workdir,
        env=env,
    )
    trace_path = workdir / "results" / "simulation_log.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Simulator did not write {trace_path}")
    copied = output_dir / "simulated_trace.jsonl"
    shutil.copy2(trace_path, copied)
    config_json = workdir / "results" / "simulation_config.json"
    if config_json.exists():
        shutil.copy2(config_json, output_dir / "simulated_config.json")
    return copied


def metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_job[str(event.get("job_id") or "unknown")].append(event)
    jobs = {}
    for job_id, rows in by_job.items():
        start = min(float(row["start"]) for row in rows)
        end = max(float(row["end"]) for row in rows)
        operations: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            operations[str(row.get("operation") or "unknown")].append(
                float(row.get("duration") or 0.0)
            )
        jobs[job_id] = {
            "wall_time_s": round(end - start, 6),
            "events": len(rows),
            "operation_mean_s": {
                operation: round(statistics.mean(samples), 6)
                for operation, samples in sorted(operations.items())
            },
        }
    return {"jobs": jobs}


def iframe(src: str, title: str) -> str:
    return (
        f'<section><h2>{html.escape(title)}</h2>'
        f'<iframe title="{html.escape(title)}" src="{html.escape(src)}"></iframe>'
        "</section>"
    )


def write_comparison_html(
    *,
    output_path: Path,
    real_html: Path,
    simulated_html: Path,
    real_metrics: dict[str, Any],
    simulated_metrics: dict[str, Any],
    title: str = "Four-node validation trace comparison",
    description: str = (
        "Real PyTorch traces from GCE T4 VMs are shown beside simulator traces "
        "generated from the measured timings."
    ),
    idle_summary: dict[str, Any] | None = None,
) -> None:
    summary_json = html.escape(
        json.dumps(
            {"real": real_metrics, "simulated": simulated_metrics},
            indent=2,
            sort_keys=True,
        )
    )
    idle_json = (
        html.escape(json.dumps(idle_summary, indent=2, sort_keys=True))
        if idle_summary is not None
        else ""
    )
    idle_details = (
        f"""
  <details>
    <summary>Idle and wait gap analysis JSON</summary>
    <pre>{idle_json}</pre>
  </details>
"""
        if idle_summary is not None
        else ""
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #172033;
      background: #f6f8fb;
    }}
    header {{
      padding: 18px 22px;
      border-bottom: 1px solid #d7deea;
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    p {{
      margin: 0;
      color: #536179;
    }}
    .frames {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 12px;
      height: calc(100vh - 230px);
      min-height: 640px;
    }}
    section {{
      min-width: 0;
      background: #ffffff;
      border: 1px solid #d7deea;
      border-radius: 8px;
      overflow: hidden;
    }}
    h2 {{
      margin: 0;
      padding: 10px 12px;
      font-size: 16px;
      border-bottom: 1px solid #d7deea;
      background: #eef3f8;
    }}
    iframe {{
      width: 100%;
      height: calc(100% - 42px);
      border: 0;
      background: #ffffff;
    }}
    details {{
      margin: 0 12px 12px;
      padding: 10px 12px;
      background: #ffffff;
      border: 1px solid #d7deea;
      border-radius: 8px;
    }}
    pre {{
      overflow: auto;
      white-space: pre-wrap;
    }}
    @media (max-width: 1100px) {{
      .frames {{
        grid-template-columns: 1fr;
        height: auto;
      }}
      section {{
        height: 760px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(description)}</p>
  </header>
  <main class="frames">
    {iframe(real_html.name, "Real PyTorch on GCE")}
    {iframe(simulated_html.name, "Simulator from measured timings")}
  </main>
  <details>
    <summary>Timing summary JSON</summary>
    <pre>{summary_json}</pre>
  </details>
  {idle_details}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.real_root = args.real_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.repo_root = args.repo_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_events = load_jsonl_files(args.real_root)
    events = comparison_events(detailed_events)
    summaries = load_summaries(args.real_root)
    detailed_trace = args.output_dir / "real_trace_detailed.jsonl"
    write_jsonl(detailed_events, detailed_trace)
    real_trace = args.output_dir / "real_trace.jsonl"
    write_jsonl(events, real_trace)
    idle_summary = idle_analysis(detailed_events)
    (args.output_dir / "idle_analysis.json").write_text(
        json.dumps(idle_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    simulator_config = args.output_dir / "simulator_from_real.toml"
    build_simulator_config(
        events=events,
        sizing_events=detailed_events,
        summaries=summaries,
        output_path=simulator_config,
        nodes=args.nodes,
        iterations=args.iterations,
        microbatches=args.microbatches,
    )

    real_html = args.output_dir / "real_pytorch_visualization.html"
    generate_visualization(args.repo_root, real_trace, real_html)
    real_metrics = metrics(events)
    detailed_real_html = args.output_dir / "real_pytorch_detailed_visualization.html"
    generate_visualization(args.repo_root, detailed_trace, detailed_real_html)
    detailed_real_metrics = metrics(detailed_events)

    simulated_trace = args.output_dir / "simulated_trace.jsonl"
    if args.run_simulator:
        simulated_trace = run_simulator(args.repo_root, simulator_config, args.output_dir)
    if not simulated_trace.exists():
        raise FileNotFoundError(
            f"{simulated_trace} does not exist; rerun with --run-simulator"
        )
    simulated_html = args.output_dir / "simulator_visualization.html"
    generate_visualization(args.repo_root, simulated_trace, simulated_html)
    simulated_metrics = metrics(load_jsonl_file(simulated_trace))

    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(
            {
                "real": real_metrics,
                "real_detailed": detailed_real_metrics,
                "simulated": simulated_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_comparison_html(
        output_path=args.output_dir / "trace_comparison.html",
        real_html=real_html,
        simulated_html=simulated_html,
        real_metrics=real_metrics,
        simulated_metrics=simulated_metrics,
        title="Four-node validation trace comparison",
        description=(
            "Simulator-comparable PyTorch operations are shown beside simulator "
            "traces generated from the measured timings."
        ),
    )
    write_comparison_html(
        output_path=args.output_dir / "trace_comparison_detailed.html",
        real_html=detailed_real_html,
        simulated_html=simulated_html,
        real_metrics=detailed_real_metrics,
        simulated_metrics=simulated_metrics,
        title="Four-node full detailed trace comparison",
        description=(
            "Every detailed PyTorch setup, iteration, wait, communication, "
            "checkpoint, and synchronization event is shown beside the simulator."
        ),
        idle_summary=idle_summary,
    )


if __name__ == "__main__":
    main()
