# Validation tooling

This directory connects simulated timing to real distributed-training
measurements. Validation is separate from the normal simulator because it can
require GPUs, cloud credentials, object storage, and resource cleanup.

[Back to the repository overview](../README.md)

## Structure

```text
validation/
`-- four_node/
    |-- README.md                    Full GCE workflow and safety notes
    |-- validate.sh                  SSH-oriented multi-VM launcher
    |-- validate_startup.sh          Startup-script/GCS launcher
    |-- real_training_benchmark.py   PyTorch distributed workload and tracer
    |-- postprocess_validation.py    Trace merge, calibration, simulation, reports
    `-- preflight.py                 Local synthetic end-to-end validation
```

The repository currently has one maintained validation target: a four-node
Google Compute Engine T4 baseline. The scripts accept other node counts, but
the folder name identifies the reference setup.

## Data flow

```text
real PyTorch rank traces
        |
        v
postprocess_validation.py
        |-- combined real JSONL
        |-- measured timing and bandwidth summaries
        |-- generated simulator_from_real.toml
        v
simulation.run_simulation
        |
        v
real-vs-simulated HTML comparison and metrics
```

Use `make validate-preflight` before any cloud run. It exercises the same
postprocessing and simulator path with synthetic local traces and creates no
cloud resources.

See [four_node/README.md](four_node/README.md) before launching validation. It
documents prerequisites, created resources, cleanup behavior, flags, and output
files.

## Change rules

- Preserve per-run resource naming and scoped cleanup in shell launchers.
- Never broaden deletion from recorded validation resources to a project-wide
  pattern.
- Keep raw measured data distinct from inferred/calibrated values.
- Update `tests/test_validation_postprocess.py` when calibration fields change.
- Run preflight after changes to benchmark output, postprocessing, config
  generation, visualization, or expected artifact names.
