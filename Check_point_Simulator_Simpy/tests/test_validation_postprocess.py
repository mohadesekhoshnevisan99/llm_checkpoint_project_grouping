from __future__ import annotations

import tomllib
from pathlib import Path

from validation.four_node.postprocess_validation import build_simulator_config


def event(
    job_id: str,
    operation: str,
    *,
    duration: float = 0.1,
    data_gb: float | None = None,
    iteration: int = 1,
    details: dict | None = None,
) -> dict:
    return {
        "job_id": job_id,
        "operation": operation,
        "duration": duration,
        "data_gb": data_gb,
        "iteration": iteration,
        "details": details or {},
    }


def test_simulator_config_extracts_sizes_from_detailed_trace(tmp_path: Path) -> None:
    nodes = 4
    summaries = {
        "real-data-parallel-sync": {
            "checkpoint_every": 99,
            "checkpoint_gb": 100.0,
            "model_parameter_gb": 100.0,
            "gradient_gb": 100.0,
            "tensor_gb": 100.0,
        },
        "real-pipeline-parallel-sync": {
            "checkpoint_every": 99,
            "checkpoint_gb": 100.0,
            "model_parameter_gb": 100.0,
            "gradient_gb": 100.0,
            "tensor_gb": 100.0,
        },
    }
    comparison_events = [
        event("real-data-parallel-sync", "data_parallel_forward"),
        event("real-data-parallel-sync", "data_parallel_backward"),
        event("real-data-parallel-sync", "data_parallel_optimizer"),
        event("real-data-parallel-sync", "data_parallel_all_reduce", data_gb=4.0),
        event(
            "real-data-parallel-sync",
            "checkpoint_stage_dram_to_object_store",
            data_gb=3.0,
        ),
        event("real-pipeline-parallel-sync", "pipeline_forward"),
        event("real-pipeline-parallel-sync", "pipeline_backward"),
        event("real-pipeline-parallel-sync", "pipeline_optimizer"),
        event(
            "real-pipeline-parallel-sync",
            "pipeline_activation_send",
            data_gb=0.5,
        ),
        event(
            "real-pipeline-parallel-sync",
            "pipeline_gradient_send",
            data_gb=0.5,
        ),
        event(
            "real-pipeline-parallel-sync",
            "checkpoint_stage_dram_to_object_store",
            data_gb=2.0,
        ),
    ]
    sizing_events = [
        *comparison_events,
        event(
            "real-data-parallel-sync",
            "data_parallel_setup_complete",
            details={
                "model_parameter_gb": 10.0,
                "gradient_gb": 10.0,
                "tensor_gb": 1.0,
            },
        ),
        event(
            "real-data-parallel-sync",
            "checkpoint_stage_gpu_to_dram",
            data_gb=3.0,
            iteration=2,
        ),
        event(
            "real-pipeline-parallel-sync",
            "pipeline_setup_complete",
            details={
                "model_parameter_gb": 5.0,
                "gradient_gb": 5.0,
                "tensor_gb": 0.5,
            },
        ),
        event(
            "real-pipeline-parallel-sync",
            "checkpoint_stage_gpu_to_dram",
            data_gb=2.0,
            iteration=2,
        ),
    ]

    output_path = tmp_path / "simulator_from_real.toml"
    build_simulator_config(
        events=comparison_events,
        sizing_events=sizing_events,
        summaries=summaries,
        output_path=output_path,
        nodes=nodes,
        iterations=2,
        microbatches=2,
    )

    raw = tomllib.loads(output_path.read_text(encoding="utf-8"))
    jobs = {job["id"]: job for job in raw["jobs"]}
    data_job = jobs["sim-data-parallel-sync-from-real"]
    pipeline_job = jobs["sim-pipeline-parallel-sync-from-real"]

    assert data_job["model"]["weights_gb"] == 10.0
    assert data_job["model"]["gradient_gb"] == 4.0
    assert data_job["model"]["activation_gb"] == 1.0
    assert data_job["checkpoint"]["size_gb"] == 3.0
    assert data_job["checkpoint"]["every"] == 99

    assert pipeline_job["model"]["weights_gb"] == 5.0 * nodes
    assert pipeline_job["model"]["gradient_gb"] == 5.0 * nodes
    assert pipeline_job["model"]["activation_gb"] == 0.5 * 2
    assert pipeline_job["checkpoint"]["size_gb"] == 2.0 * nodes
