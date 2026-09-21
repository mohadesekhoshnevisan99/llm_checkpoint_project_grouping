from __future__ import annotations

from pathlib import Path

import simpy

import simulation  # noqa: F401
from jobs import load_config, run_configured_jobs
from nodes.node import EventLogger


def write_config(path: Path, *, checkpoint_mode: str) -> None:
    path.write_text(
        f"""
[simulation]
seed = 7
iterations = 2

[cluster]
node_count = 1
cpu_cores_per_node = 1
cpu_memory_gb = 3.75
gpus_per_node = 1
gpu_memory_gb = 16.0
gpu_cpu_bandwidth_gbps = 10.0
network_bandwidth_gbps = 10.0
communication_launch_seconds = 0.000001
local_ssd_bandwidth_gbps = 1.0
object_store_bandwidth_gbps = 1.0
object_store_concurrency = 1

[failures]
probability_per_second = 0.0
process_weight = 1.0
node_weight = 0.0
spot_weight = 0.0
process_restart_seconds = 1.0
node_restart_seconds = 1.0
spot_restart_seconds = 1.0

[[jobs]]
id = "checkpoint-mode-test"
strategy = "data_parallel"
failure_rate = 0.0
data_parallel_replicas = 1
pipeline_stages = 1
microbatches = 1

[jobs.model]
weights_gb = 1.0
gradient_gb = 1.0
activation_gb = 1.0

[jobs.timing]
forward_seconds = 0.1
backward_seconds = 0.1
optimizer_seconds = 0.1

[jobs.checkpoint]
strategy = "object_store"
mode = "{checkpoint_mode}"
every = 1
size_gb = 10.0
upload_gpu_slowdown = 1.0
""".lstrip(),
        encoding="utf-8",
    )


def run_mode(tmp_path: Path, checkpoint_mode: str):
    config_path = tmp_path / f"{checkpoint_mode}.toml"
    write_config(config_path, checkpoint_mode=checkpoint_mode)
    env = simpy.Environment()
    logger = EventLogger()
    completion, _placements = run_configured_jobs(
        env,
        run_id=0,
        config=load_config(config_path),
        logger=logger,
        verbose=False,
    )
    env.run(until=completion)
    upload = next(
        event
        for event in logger.events
        if event.operation == "checkpoint_stage_dram_to_object_store"
        and event.iteration == 1
    )
    second_forward = next(
        event
        for event in logger.events
        if event.operation == "data_parallel_forward" and event.iteration == 2
    )
    return upload, second_forward


def test_synchronous_checkpoint_blocks_next_iteration(tmp_path: Path) -> None:
    upload, second_forward = run_mode(tmp_path, "synchronous")

    assert second_forward.start >= upload.end
    assert upload.details["checkpoint_mode"] == "synchronous"


def test_asynchronous_checkpoint_overlaps_next_iteration(tmp_path: Path) -> None:
    upload, second_forward = run_mode(tmp_path, "asynchronous")

    assert second_forward.start < upload.end
    assert upload.details["checkpoint_mode"] == "asynchronous"


def test_concurrent_jobs_use_disjoint_physical_nodes(tmp_path: Path) -> None:
    config_path = tmp_path / "placement.toml"
    config_path.write_text(
        """
[simulation]
seed = 7
iterations = 1

[cluster]
node_count = 4
cpu_cores_per_node = 1
cpu_memory_gb = 3.75
gpus_per_node = 1
gpu_memory_gb = 16.0
gpu_cpu_bandwidth_gbps = 10.0
network_bandwidth_gbps = 10.0
communication_launch_seconds = 0.000001
local_ssd_bandwidth_gbps = 1.0
object_store_bandwidth_gbps = 1.0
object_store_concurrency = 4

[failures]
probability_per_second = 0.0
process_weight = 1.0
node_weight = 0.0
spot_weight = 0.0
process_restart_seconds = 1.0
node_restart_seconds = 1.0
spot_restart_seconds = 1.0

[[jobs]]
id = "job-a"
strategy = "data_parallel"
failure_rate = 0.0
data_parallel_replicas = 2
pipeline_stages = 1
microbatches = 1

[jobs.model]
weights_gb = 1.0
gradient_gb = 1.0
activation_gb = 1.0

[jobs.timing]
forward_seconds = 0.1
backward_seconds = 0.1
optimizer_seconds = 0.1

[jobs.checkpoint]
strategy = "object_store"
mode = "synchronous"
every = 1
size_gb = 1.0
upload_gpu_slowdown = 1.0

[[jobs]]
id = "job-b"
strategy = "pipeline_parallel"
failure_rate = 0.0
data_parallel_replicas = 1
pipeline_stages = 2
microbatches = 1

[jobs.model]
weights_gb = 2.0
gradient_gb = 2.0
activation_gb = 1.0

[jobs.timing]
forward_seconds = 0.1
backward_seconds = 0.1
optimizer_seconds = 0.1

[jobs.checkpoint]
strategy = "object_store"
mode = "asynchronous"
every = 1
size_gb = 1.0
upload_gpu_slowdown = 1.0
""".lstrip(),
        encoding="utf-8",
    )
    env = simpy.Environment()
    completion, placements = run_configured_jobs(
        env,
        run_id=0,
        config=load_config(config_path),
        logger=EventLogger(),
        verbose=False,
    )

    physical_nodes = [placement["physical_node"] for placement in placements]

    assert len(physical_nodes) == len(set(physical_nodes))
    assert set(physical_nodes) == {"node-0", "node-1", "node-2", "node-3"}
    assert completion is not None
