# Simulator configurations

This directory contains concrete inputs for the general configured simulator.
These files are parsed by `jobs.config.load_config` and run by `main.py`,
`simulation.run_simulation`, or `run_crossjob.py`.

[Back to the repository overview](../README.md)

Do not confuse this schema with the experiment-level YAML schema in
[`scenarios/`](../scenarios/README.md). A config enumerates individual jobs; a
scenario defines reusable job classes, arm comparisons, seeds, controller
settings, and optional preemption.

## Files

| File | Purpose |
| --- | --- |
| `jobs.toml` | Default broad simulation used by `make run`; includes data, pipeline, hybrid, tiered, failure-free, and very-large jobs. |
| `crossjob_smoke.toml` | Small two-job `crossjob_peer` setup for direct cross-job smoke runs. |

## Schema

```toml
[simulation]
seed = 17
iterations = 20

[cluster]
node_count = 64
cpu_cores_per_node = 1
cpu_memory_gb = 3.75
gpus_per_node = 1
gpu_memory_gb = 80.0
gpu_cpu_bandwidth_gbps = 31.5
network_bandwidth_gbps = 25.0
communication_launch_seconds = 0.01
local_ssd_bandwidth_gbps = 4.0
object_store_bandwidth_gbps = 8.0
object_store_concurrency = 4

[failures]
probability_per_second = 0.002
process_weight = 0.50
node_weight = 0.30
spot_weight = 0.20
process_restart_seconds = 3.0
node_restart_seconds = 8.0
spot_restart_seconds = 15.0

[[jobs]]
id = "example"
strategy = "data_parallel"
data_parallel_replicas = 4
pipeline_stages = 1
microbatches = 1
failure_rate = 0.001

[jobs.model]
weights_gb = 26.0
gradient_gb = 26.0
activation_gb = 2.0

[jobs.timing]
forward_seconds = 5.0
backward_seconds = 9.0
optimizer_seconds = 1.0

[jobs.checkpoint]
strategy = "object_store"
mode = "asynchronous"
every = 2
size_gb = 30.0
chunk_gb = 4.0
upload_gpu_slowdown = 1.15
```

## Important semantics

- A job's rank count is `data_parallel_replicas * pipeline_stages`.
- Physical nodes are assigned in config order and are not shared between jobs.
- A job-level `failure_rate` overrides the global probability; otherwise the
  global value is inherited.
- Checkpoint mode accepts synchronous/blocking and asynchronous/background
  spelling variants and normalizes them.
- TOML works with the Python standard library. YAML requires PyYAML.
- `pipeline_stages = 0` asks the loader to choose a feasible stage count from
  model weight size and usable GPU memory.
- The cluster needs at least the sum of all job rank counts.

## Adding a config

Prefer a focused file that demonstrates one behavior. Keep units in field
names: sizes are GB, bandwidths are GB/s, and durations are seconds. Add a short
comment above unusual jobs and run it through the public API test or a targeted
smoke command before committing.
