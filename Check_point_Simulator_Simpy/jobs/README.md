# Jobs package

This package turns normalized job configuration into concurrent training
processes. It owns placement, execution order, parallelism dependencies,
failure injection, and selection of the appropriate scale runtime.

[Back to the repository overview](../README.md)

## Package structure

| File | Responsibility |
| --- | --- |
| `__init__.py` | Exposes `SimulatorConfig`, `load_config`, and `run_configured_jobs`. |
| `config.py` | Parses TOML/YAML, applies defaults, validates values, resolves automatic pipeline stages, and returns frozen dataclasses. |
| `runtime.py` | Detailed data/pipeline/hybrid execution, physical placement, execution planning, and configured-run orchestration. |
| `failures.py` | Per-node failure sampling, severity coalescing, restart delays, failure logging, and recovery dispatch. |
| `chunked_runtime.py` | Large data-parallel execution that retains individual rank behavior while scheduling ranks in bounded chunks. |
| `aggregate_runtime.py` | Analytical cohort execution for very large supported jobs when the chunked runtime does not apply. |

## Configuration objects

`config.py` reduces input files to four immutable types:

- `SimulationSettings`: seed and iteration count;
- `ClusterConfig`: node resources, transfer rates, and object-store capacity;
- `FailureSettings`: failure mix and restart durations;
- `JobConfig`: parallel layout, tensor sizes, timing, checkpoint policy, and
  effective per-node failure rate.

`SimulatorConfig` groups them and calculates the number of required physical
nodes. Keeping these objects immutable makes a resolved run reproducible and
safe to serialize into `simulation_config.json`.

## Runtime planning

`ExecutionPlanner` visits jobs in config order, assigns disjoint contiguous
physical-node ranges, and selects a runtime:

```text
job
|-- ChunkedDataParallelRuntime.supports(job) -> chunked individual ranks
|-- otherwise rank_count > 10,000            -> analytical aggregate runtime
`-- otherwise                                -> detailed distributed runtime
```

The planner also collects placement records for output. If you change the
threshold or support matrix, update scale tests and any README claims about
trace granularity.

## Detailed runtime

`DistributedJobRuntime` creates one `JobWorker` per `(data_parallel_rank,
pipeline_stage)` pair. Each worker owns simulated GPU and priority CPU
resources. The runtime coordinates:

- forward and backward compute;
- activation and gradient transfers between pipeline stages;
- data-parallel synchronization and ring all-reduce;
- optimizer work;
- synchronous or asynchronous checkpoint launch;
- dynamic slowdown from overlapping work;
- checkpoint-aware failure recovery.

The global rank is data-parallel-major:

```text
rank = data_parallel_rank * pipeline_stages + pipeline_stage
```

`run_configured_jobs` builds all runtimes, starts them concurrently, and starts
one shared `FailureController` for detailed workers.

## Failures

The controller samples each active worker every simulated integer second using
the job's effective failure probability. Multiple failures on one tick form a
batch. Signals received while a worker is already down extend the recovery
window and can raise severity from process to node to spot.

The checkpoint strategy owns data loss and restore mechanics; the failure
controller owns timing, coalescing, and lifecycle state. Keep that boundary
intact when adding new failure types.

## Import boundary

`runtime.py` lazily imports the checkpointing factory because checkpointing,
simulation, and jobs refer to one another. Avoid adding new eager imports that
recreate the cycle. Shared scheduling behavior belongs in `simulation`, and
checkpoint placement belongs in `checkpointing`.

## Tests to run

- Config/API changes: `tests/test_public_api.py` and checkpoint-mode tests.
- Dependency or logging changes: `tests/test_simulation_log.py`.
- Failure changes: failure sections of `test_simulation_log.py`.
- Planner or scale changes: `tests/test_large_scale.py`.
- Backend-facing changes: `tests/test_backend.py`.
