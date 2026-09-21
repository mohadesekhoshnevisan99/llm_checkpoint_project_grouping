# Distributed-training checkpoint simulator

This repository models distributed training, checkpoint placement, resource
contention, failures, and recovery. It contains a reusable discrete-event
simulation engine and a higher-level experiment harness for comparing cross-job
checkpoint policies.

## Start here

Create the virtual environment, install dependencies, run the default
configuration, and build the HTML timeline:

```powershell
make install
make run
make visualise
make visualise-aggregate
make check
```

The equivalent direct commands are:

```powershell
python main.py --config configs/jobs.toml
python visualise.py --log results/simulation_log.jsonl --output results/simulation_visualization.html
python visualise_aggregate.py --log results/simulation_log.jsonl --output results/simulation_aggregate.html
python -m ruff check .
python -m pytest
```

The detailed visualizer creates rank/resource lanes and is best for small runs.
The aggregate visualizer is the wide-run report: it explains allocation drops,
ideal-versus-actual overhead, weighted training goodput, operation node-hours,
checkpoint policy and placement, priority/QoS rules, failures and recovery
sources, per-job efficiency, data movement, and bounded slow spans. It accepts
plain `.jsonl` and compressed `.jsonl.gz` traces. Pass `--scenario` and
`--result` to enrich trace spans with configuration and exact run outcomes.

After generating one aggregate dashboard per policy, build a comparison landing
page from the scenario result files:

```powershell
python visualise_policy_matrix.py --result results/named.json `
  --result results/ours.json --dashboard-dir results/dashboards `
  --output results/index.html
```

Visualize an existing scenario trace without rerunning the simulation:

```powershell
make visualise-scenario-aggregate SCENARIO=scenarios/realnode26.yaml ARM=ours_full SEED=7
```

Run and then visualize explicitly when a fresh simulation is wanted:

```powershell
make run-and-visualise-scenario-aggregate SCENARIO=scenarios/realnode26.yaml ARM=ours_full SEED=7
```

Independent arms and seeds can run in parallel OS processes:

```powershell
python run_scenario.py --scenario scenarios/realnode26.yaml --workers 4
```

Parallelism is across complete arm/seed simulations. A single arm/seed keeps
one ordered event queue because its jobs share donor availability and bandwidth
capacity; running those events in threads would change simulation causality.

Scenario runs print a cohort-weighted progress bar and ETA every ten wall-clock
seconds. Change the cadence with `PROGRESS_INTERVAL=30` in Make or
`--progress-interval 30` on `run_scenario.py`. The same samples are written next
to the result JSON as a `.progress` file.

Scenario traces are retained in bounded sorted batches instead of keeping the
complete event history in memory. The default batch is 50,000 events; adjust it
with `EVENT_BATCH_SIZE=10000` in Make or `--event-batch-size 10000`. Smaller
batches use less memory and create more temporary merge chunks.

American-spelling aliases are available as `make visualize`,
`make visualize-aggregate`, and `make visualize-scenario-aggregate`.

To call the simulator from Python:

```python
from simulation import run_simulation

result = run_simulation("configs/jobs.toml")
print(result.simulated_time, result.event_count)
```

`simulator.run(...)` is a compatibility shortcut for the same public API.

## Which workflow should I use?

There are two entry paths built on the same simulation concepts:

| Goal | Entry point | Input | Output |
| --- | --- | --- | --- |
| Run configured data, pipeline, or hybrid jobs | `main.py` or `simulation.run_simulation` | `configs/*.toml`, `.yaml`, or `.yml` | JSONL trace and resolved JSON config |
| Compare cross-job checkpoint policies | `run_scenario.py` | `scenarios/*.yaml` and optional policy JSON | Per-arm metrics JSON and optional compressed traces |
| Run a small cross-job configuration directly | `run_crossjob.py` | simulator config plus optional policy JSON | Cross-job trace and summary |
| Solve controller policy parameters | `gp_policy.py` | scenario YAML | policy JSON |
| Inspect or validate a cross-job trace | `trace_report.py`, `trace_validator.py` | JSONL or JSONL.GZ trace | console, PNG, HTML, and invariant results |
| Explore a wide trace without node lanes | `visualise_aggregate.py` | JSONL or JSONL.GZ trace | Standalone service-oriented HTML dashboard |
| Compare simulation with real hardware | `validation/four_node/` | GCE/PyTorch run parameters | calibrated config and comparison reports |

Use `main.py` when you are developing the simulator itself or testing one
concrete job mix. Use `run_scenario.py` when the experimental unit is an arm ×
seed matrix and jobs are described as workload classes.

## Repository map

```text
HHHH/
|-- checkpointing/   Checkpoint placement, persistence, capacity, and recovery
|-- configs/         Small simulator-level TOML configurations
|-- docs/            Design history, gap analysis, and verification notes
|-- jobs/            Config parsing, runtime planning, orchestration, and failures
|-- nodes/           Event schema and legacy resource-level machine operations
|-- results/         Generated traces, reports, and validation artifacts
|-- scenarios/       Experiment-level YAML workloads and policy-arm definitions
|-- simpy/           Repository-local SimPy-compatible scheduler implementation
|-- simulation/      Public API, backend abstraction, and shared progress engine
|-- tests/           Unit, integration, scale, trace, and validation tests
|-- validation/      Real-hardware calibration and comparison tooling
|-- main.py          Main CLI; configured mode plus the legacy single-job mode
|-- run_scenario.py  Scenario expansion and arm/seed experiment driver
|-- run_crossjob.py  Lower-level cross-job runner
|-- gp_policy.py     Controller policy optimizer
|-- visualise.py     Interactive Plotly timeline generator
|-- visualise_aggregate.py  Service-oriented aggregate trace explorer
|-- trace_report.py  Cross-job trace summaries and figures
|-- trace_validator.py  Machine-checkable physical trace invariants
|-- simulator.py     Thin compatibility facade over simulation.runner
|-- Makefile         Common install, run, validation, test, and lint commands
|-- pyproject.toml   pytest and Ruff configuration
`-- requirements.txt Runtime and development dependencies
```

Every maintained directory has its own README with a file-by-file guide. The
exceptions are `.git/`, caches, bytecode folders, virtual environments, and
timestamped folders below `results/`; those are internal or generated state.

## How a configured run moves through the code

```text
config file
   |
   v
jobs.config.load_config
   |
   v
simulation.runner.run_simulation
   |
   +--> jobs.runtime.ExecutionPlanner
   |       |-- detailed runtime for ordinary jobs
   |       |-- chunked individual-rank runtime when supported at large scale
   |       `-- analytical aggregate runtime for other jobs above 10,000 ranks
   |
   +--> checkpointing.create_checkpoint_strategy
   +--> jobs.failures.FailureController
   `--> nodes.node.EventLogger
              |
              v
       results/simulation_log.jsonl
              |
              v
          visualise.py
```

The `simulation` package owns the stable public boundary. `jobs` owns workload
execution. `checkpointing` owns durable-state behavior. `nodes` owns the event
record and the older resource-level model. This separation lets domain code use
the small `SimulationBackend` contract instead of depending directly on a
particular scheduler.

## Configuration model

A configured run has four layers:

1. `[simulation]` sets the seed and iteration count.
2. `[cluster]` describes per-node resources and shared object-store capacity.
3. `[failures]` defines failure probability, type weights, and restart times.
4. Each `[[jobs]]` block describes parallelism, model sizes, operation timing,
   checkpoint strategy, cadence, and optional per-job failure rate.

The rank mapping is:

```text
global_rank = data_parallel_rank * pipeline_stages + pipeline_stage
```

Jobs run concurrently on disjoint configured nodes. Supported execution
strategies include data parallel, pipeline parallel, and hybrid. Supported
checkpoint names are registered in `checkpointing/__init__.py` and currently
include object-store, CPU-tiered, SSD-tiered, paired-tiered, local-tiered, and
cross-job peer variants.

See [configs/README.md](configs/README.md) for the simulator schema and
[scenarios/README.md](scenarios/README.md) for the separate experiment schema.

## Event and artifact contract

The main simulator writes:

- `results/simulation_log.jsonl`: one structured interval or failure event per
  line, sorted by simulated start/end time;
- `results/simulation_config.json`: the normalized input plus resolved physical
  placement;
- `results/simulation_visualization.html`: produced separately by
  `visualise.py`.

Events identify the run, job, global rank, data-parallel rank, pipeline stage,
physical node, operation, resources, iteration, duration, byte count, and
operation-specific details. Tests treat this schema and the dependency order as
part of the public behavior.

## Checkpoint and failure behavior

Checkpoint capture always begins with a serialized GPU-to-DRAM stage. Later
persistence may be synchronous or asynchronous, depending on the job config.
Tiered strategies write configurable chunks and may retain local, paired, peer,
SSD, or remote-store copies.

Failure severity is `process < node < spot`. Process failures retain local
memory and disk, node failures flush DRAM, and spot failures flush DRAM and
local SSD. Overlapping failure signals are coalesced at the highest severity.
Recovery selects the newest completed surviving checkpoint; tier speed breaks
ties. Without a surviving copy, execution restarts from iteration zero.

Interrupted operations are discarded and retried after restart and restore:

```text
interrupted work -> restart delay -> fetch checkpoint -> DRAM-to-GPU restore
                 -> retry the interrupted operation
```

## Common development tasks

| Change | Primary location | Tests to start with |
| --- | --- | --- |
| Add a config field | `jobs/config.py` | `tests/test_public_api.py`, config-using integration tests |
| Add a checkpoint strategy | `checkpointing/` and its factory in `checkpointing/__init__.py` | checkpoint, failure, and trace tests |
| Change job scheduling or dependencies | `jobs/runtime.py` | `tests/test_simulation_log.py` |
| Change scale behavior | `jobs/chunked_runtime.py`, `jobs/aggregate_runtime.py` | `tests/test_large_scale.py` |
| Change scheduler primitives | `simpy/`, `simulation/backend.py` | `tests/test_backend.py` |
| Add an experiment arm | scenario YAML plus `run_scenario.py` or `checkpointing/baselines.py` | cross-job and donor-drain tests |
| Change event fields or visualization | `nodes/node.py`, `visualise.py` | log and visualization assertions |
| Change hardware calibration | `validation/four_node/` | `tests/test_validation_postprocess.py` and preflight |

Run `make check` before committing. For changes to cross-job physics, also run
`trace_validator.py` on a representative generated trace.

## Further documentation

- [checkpointing/README.md](checkpointing/README.md)
- [jobs/README.md](jobs/README.md)
- [simulation/README.md](simulation/README.md)
- [nodes/README.md](nodes/README.md)
- [simpy/README.md](simpy/README.md)
- [configs/README.md](configs/README.md)
- [scenarios/README.md](scenarios/README.md)
- [tests/README.md](tests/README.md)
- [validation/README.md](validation/README.md)
- [docs/README.md](docs/README.md)
- [results/README.md](results/README.md)


## Before you push (Sam + Atharva)

```bash
./check.sh              # quick gate, ~2 min — tests, hand-check, validator, pencil math
./check.sh full         # ~8 min — both engines, all hand-checks, mini-C smoke
./check.sh install-hook # make the quick gate run automatically on every git push
```

Ground truth: handcheck2's big0 peer flush is EXACTLY 3.4483 s (fluid pencil math).
If your change moves it, stop and talk before pushing. Per-seed differences between
the event/polling engines (or across logger changes) at decision boundaries are the
documented distribution-level-equivalence class — see ENGINE_REWRITE_VERIFICATION.md.
