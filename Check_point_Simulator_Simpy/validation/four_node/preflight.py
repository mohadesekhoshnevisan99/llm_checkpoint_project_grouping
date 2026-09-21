from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-cloud validation smoke test for the four-node harness."
        )
    )
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/validation_four_node/preflight"),
    )
    return parser.parse_args()


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.current = 0

    def step(self, message: str) -> None:
        self.current += 1
        width = 28
        filled = int(width * self.current / self.total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{bar}] {self.current}/{self.total} {message}", flush=True)


def event(
    event_id: int,
    *,
    job_id: str,
    rank: int,
    operation: str,
    start: float,
    duration: float,
    iteration: int,
    category: str = "Computation",
    stage: int = 0,
    data_parallel_rank: int = 0,
    data_gb: float | None = None,
    resources: list[str] | None = None,
    source: str | None = None,
    destination: str | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "start": start,
        "end": start + duration,
        "duration": duration,
        "rank": rank,
        "node": f"rank-{rank}",
        "category": category,
        "operation": operation,
        "run_id": 0,
        "job_id": job_id,
        "pipeline_stage": stage,
        "data_parallel_rank": data_parallel_rank,
        "physical_node": f"preflight-node-{rank}",
        "resources": resources or ["GPU"],
        "iteration": iteration,
        "source": source,
        "destination": destination,
        "data_gb": data_gb,
        "failure_type": None,
        "details": {
            "source_system": "preflight",
            "base_duration": duration,
            "effective_slowdown": 1.0,
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def build_synthetic_remote_logs(
    *,
    remote_root: Path,
    nodes: int,
    iterations: int,
    microbatches: int,
) -> None:
    shutil.rmtree(remote_root, ignore_errors=True)
    event_id = 1
    checkpoint_modes = (
        ("synchronous", "sync"),
        ("asynchronous", "async"),
    )
    for rank in range(nodes):
        log_dir = remote_root / f"preflight-node-{rank}" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for checkpoint_mode, suffix in checkpoint_modes:
            data_job = f"real-data-parallel-{suffix}"
            pipeline_job = f"real-pipeline-parallel-{suffix}"
            for iteration in range(1, iterations + 1):
                offset = (iteration - 1) * 0.5
                rows.extend(
                    [
                        event(
                            event_id,
                            job_id=data_job,
                            rank=rank,
                            operation="data_parallel_forward",
                            start=offset,
                            duration=0.015 + rank * 0.001,
                            iteration=iteration,
                            data_parallel_rank=rank,
                        ),
                        event(
                            event_id + 1,
                            job_id=data_job,
                            rank=rank,
                            operation="data_parallel_backward",
                            start=offset + 0.02,
                            duration=0.025 + rank * 0.001,
                            iteration=iteration,
                            data_parallel_rank=rank,
                        ),
                        event(
                            event_id + 2,
                            job_id=data_job,
                            rank=rank,
                            operation="data_parallel_all_reduce",
                            start=offset + 0.05,
                            duration=0.012,
                            iteration=iteration,
                            category="Communication",
                            data_parallel_rank=rank,
                            data_gb=0.02,
                            resources=["GPU", "NETWORK"],
                            source=f"rank-{rank}",
                            destination=f"{data_job}-stage-0-replicas",
                        ),
                        event(
                            event_id + 3,
                            job_id=data_job,
                            rank=rank,
                            operation="data_parallel_optimizer",
                            start=offset + 0.07,
                            duration=0.009,
                            iteration=iteration,
                            data_parallel_rank=rank,
                        ),
                    ]
                )
                event_id += 4
                if rank == 0:
                    rows.extend(
                        [
                            event(
                                event_id,
                                job_id=data_job,
                                rank=rank,
                                operation="checkpoint_stage_gpu_to_dram",
                                start=offset + 0.085,
                                duration=0.010,
                                iteration=iteration,
                                category="Checkpoint",
                                data_parallel_rank=rank,
                                data_gb=0.02,
                                resources=["CPU", "GPU"],
                                source=f"rank-{rank}",
                                destination=f"rank-{rank}",
                            ),
                            event(
                                event_id + 1,
                                job_id=data_job,
                                rank=rank,
                                operation="checkpoint_stage_dram_to_object_store",
                                start=offset + 0.098,
                                duration=0.012,
                                iteration=iteration,
                                category="Checkpoint",
                                data_parallel_rank=rank,
                                data_gb=0.02,
                                resources=["CPU", "NETWORK"],
                                source=f"rank-{rank}",
                                destination="object-store",
                            ),
                        ]
                    )
                    event_id += 2

                for microbatch in range(microbatches):
                    mb_offset = offset + 0.12 + microbatch * 0.08
                    rows.append(
                        event(
                            event_id,
                            job_id=pipeline_job,
                            rank=rank,
                            operation="pipeline_forward",
                            start=mb_offset,
                            duration=0.010 + rank * 0.001,
                            iteration=iteration,
                            stage=rank,
                        )
                    )
                    event_id += 1
                    if rank < nodes - 1:
                        rows.append(
                            event(
                                event_id,
                                job_id=pipeline_job,
                                rank=rank,
                                operation="pipeline_activation_send",
                                start=mb_offset + 0.012,
                                duration=0.006,
                                iteration=iteration,
                                category="Communication",
                                stage=rank,
                                data_gb=0.01,
                                resources=["GPU", "NETWORK"],
                                source=f"rank-{rank}",
                                destination=f"rank-{rank + 1}",
                            )
                        )
                        event_id += 1
                    rows.append(
                        event(
                            event_id,
                            job_id=pipeline_job,
                            rank=rank,
                            operation="pipeline_backward",
                            start=mb_offset + 0.03,
                            duration=0.018 + rank * 0.001,
                            iteration=iteration,
                            stage=rank,
                        )
                    )
                    event_id += 1
                    if rank > 0:
                        rows.append(
                            event(
                                event_id,
                                job_id=pipeline_job,
                                rank=rank,
                                operation="pipeline_gradient_send",
                                start=mb_offset + 0.052,
                                duration=0.006,
                                iteration=iteration,
                                category="Communication",
                                stage=rank,
                                data_gb=0.01,
                                resources=["GPU", "NETWORK"],
                                source=f"rank-{rank}",
                                destination=f"rank-{rank - 1}",
                            )
                        )
                        event_id += 1

                rows.extend(
                    [
                        event(
                            event_id,
                            job_id=pipeline_job,
                            rank=rank,
                            operation="pipeline_optimizer",
                            start=offset + 0.36,
                            duration=0.008,
                            iteration=iteration,
                            stage=rank,
                        ),
                        event(
                            event_id + 1,
                            job_id=pipeline_job,
                            rank=rank,
                            operation="checkpoint_stage_gpu_to_dram",
                            start=offset + 0.38,
                            duration=0.010,
                            iteration=iteration,
                            category="Checkpoint",
                            stage=rank,
                            data_gb=0.02,
                            resources=["CPU", "GPU"],
                            source=f"rank-{rank}",
                            destination=f"rank-{rank}",
                        ),
                        event(
                            event_id + 2,
                            job_id=pipeline_job,
                            rank=rank,
                            operation="checkpoint_stage_dram_to_object_store",
                            start=offset + 0.40,
                            duration=0.012,
                            iteration=iteration,
                            category="Checkpoint",
                            stage=rank,
                            data_gb=0.02,
                            resources=["CPU", "NETWORK"],
                            source=f"rank-{rank}",
                            destination="object-store",
                        ),
                    ]
                )
                event_id += 3

            for mode, job_id in (
                ("data_parallel", data_job),
                ("pipeline_parallel", pipeline_job),
            ):
                summary = {
                    "job_id": job_id,
                    "mode": mode,
                    "checkpoint_mode": checkpoint_mode,
                    "rank": rank,
                    "world_size": nodes,
                    "node_name": f"preflight-node-{rank}",
                    "iterations": iterations,
                    "microbatches": microbatches,
                    "checkpoint_every": 1,
                    "checkpoint_gb": 0.02,
                    "model_parameter_gb": 0.04,
                    "gradient_gb": 0.04,
                    "tensor_gb": 0.01,
                    "gpu_memory": {"cuda": True, "total_gb": 16.0},
                    "cpu_memory": {"available": True, "total_gb": 15.0},
                    "cpu_cores": 4,
                }
                (log_dir / f"{mode}_{checkpoint_mode}_rank_{rank}.summary.json").write_text(
                    json.dumps(summary, indent=2),
                    encoding="utf-8",
                )

        write_jsonl(rows, log_dir / f"rank-{rank}.jsonl")


def run_postprocess(
    *,
    repo_root: Path,
    remote_root: Path,
    output_dir: Path,
    nodes: int,
    iterations: int,
    microbatches: int,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "validation" / "four_node" / "postprocess_validation.py"),
            "--real-root",
            str(remote_root),
            "--output-dir",
            str(output_dir),
            "--repo-root",
            str(repo_root),
            "--nodes",
            str(nodes),
            "--iterations",
            str(iterations),
            "--microbatches",
            str(microbatches),
            "--run-simulator",
        ],
        cwd=repo_root,
        check=True,
    )


def require_outputs(output_dir: Path) -> None:
    expected = [
        "real_trace.jsonl",
        "simulated_trace.jsonl",
        "simulator_from_real.toml",
        "validation_metrics.json",
        "real_pytorch_visualization.html",
        "simulator_visualization.html",
        "trace_comparison.html",
    ]
    missing = [name for name in expected if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Preflight did not create expected outputs: " + ", ".join(missing)
        )


def main() -> None:
    args = parse_args()
    if args.nodes < 2:
        raise ValueError("--nodes must be at least 2")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.microbatches <= 0:
        raise ValueError("--microbatches must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (repo_root / args.output_dir).resolve()
    remote_root = output_dir / "synthetic_remote"
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = Progress(total=4)
    progress.step("Generating synthetic rank logs")
    build_synthetic_remote_logs(
        remote_root=remote_root,
        nodes=args.nodes,
        iterations=args.iterations,
        microbatches=args.microbatches,
    )

    progress.step("Running simulator and visualization pipeline")
    run_postprocess(
        repo_root=repo_root,
        remote_root=remote_root,
        output_dir=output_dir,
        nodes=args.nodes,
        iterations=args.iterations,
        microbatches=args.microbatches,
    )

    progress.step("Checking generated artifacts")
    require_outputs(output_dir)

    progress.step("Preflight complete")
    print(f"comparison HTML: {output_dir / 'trace_comparison.html'}")


if __name__ == "__main__":
    main()
