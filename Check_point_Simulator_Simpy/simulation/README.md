# Simulation package

This is the stable public boundary around the simulator. It separates scheduler
mechanics from domain behavior and exposes the supported Python API.

[Back to the repository overview](../README.md)

## Package structure

| File | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports the supported backend, progress, and runner API. |
| `backend.py` | Defines the minimal scheduler/resource protocols and the current `SimPyBackend` adapter. |
| `progress.py` | Advances interruptible work under dynamic slowdown and combines named contention sources. |
| `runner.py` | Loads config, creates the environment, launches jobs, runs to completion, and writes standard artifacts. |

## Public API

The normal call is:

```python
from simulation import run_simulation

result = run_simulation(
    "configs/jobs.toml",
    results_dir="results",
    run_id=0,
    verbose=False,
    write_artifacts=True,
)
```

`SimulationResult` includes the normalized config, config path, run ID,
simulated completion time, placements, output paths, and in-memory logger.
`run` and `simulator` are aliases retained for simple callers.

## Backend contract

Domain code depends on `SimulationBackend`, which supplies only:

- current simulated time;
- timeouts and generator processes;
- all-of synchronization and manually completed events;
- ordinary and priority resources.

`SimPyBackend` adapts those operations to the repository-local `simpy` package.
A different scheduler can be introduced by implementing the same protocol and
passing it through orchestration code; checkpoint and job logic should not need
direct scheduler imports.

## Dynamic progress

`advance_work` repeatedly measures the current slowdown functions while work is
active. It returns completed work, elapsed time, and interruption state. Named
`Slowdown` values make independent contention sources composable and visible in
event details.

Use this helper for operations whose rate can change while they run. A single
fixed timeout is appropriate only when duration cannot be affected by newly
arriving work or failure.

## Artifact behavior

With `write_artifacts=True`, `runner.py` creates:

- `simulation_log.jsonl` through `EventLogger`;
- `simulation_config.json` containing normalized configuration and placement.

HTML visualization is intentionally a separate step handled by
`../visualise.py`, so library callers can run without Plotly rendering work.

## Design rules

- Keep public exports explicit in `__init__.py`.
- Do not put job or checkpoint policy in this package.
- Add backend operations only when multiple domain layers genuinely need them.
- Preserve deterministic ordering and run IDs in artifacts.
- Test public return values as well as generated files.
