from __future__ import annotations

import json

import simulator as simulator_module
from simulation import SimulationResult, run_simulation


def test_public_simulator_runs_toml_config(tmp_path) -> None:
    config_path = tmp_path / "jobs.toml"
    config_path.write_text(
        """
[simulation]
seed = 7
iterations = 1

[cluster]
node_count = 2
cpu_cores_per_node = 1
gpus_per_node = 1
gpu_memory_gb = 80.0
gpu_cpu_bandwidth_gbps = 31.5
network_bandwidth_gbps = 25.0
communication_launch_seconds = 0.01
local_ssd_bandwidth_gbps = 4.0
object_store_bandwidth_gbps = 8.0
object_store_concurrency = 2

[failures]
probability_per_second = 0.0
process_weight = 1.0
node_weight = 0.0
spot_weight = 0.0
process_restart_seconds = 3.0
node_restart_seconds = 8.0
spot_restart_seconds = 15.0

[[jobs]]
id = "toml-data"
strategy = "data_parallel"
failure_rate = 0.0
data_parallel_replicas = 2
pipeline_stages = 1
microbatches = 1

[jobs.model]
weights_gb = 26.0
gradient_gb = 26.0
activation_gb = 2.0

[jobs.timing]
forward_seconds = 1.0
backward_seconds = 1.0
optimizer_seconds = 0.5

[jobs.checkpoint]
strategy = "object_store"
every = 2
size_gb = 30.0
upload_gpu_slowdown = 1.15
""".lstrip(),
        encoding="utf-8",
    )

    result = run_simulation(config_path, results_dir=tmp_path / "results")

    assert isinstance(result, SimulationResult)
    assert simulator_module.run is simulator_module.simulator
    assert result.logical_ranks == 2
    assert result.physical_nodes == 2
    assert result.event_count > 0
    assert result.event_log_path is not None
    assert result.config_output_path is not None
    assert result.event_log_path.exists()
    assert result.config_output_path.exists()

    logged_events = result.event_log_path.read_text(
        encoding="utf-8"
    ).splitlines()
    resolved_config = json.loads(
        result.config_output_path.read_text(encoding="utf-8")
    )
    assert len(logged_events) == result.event_count
    assert resolved_config["config_path"] == str(config_path)
    assert len(resolved_config["placements"]) == 2
