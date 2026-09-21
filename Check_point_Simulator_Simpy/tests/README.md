# Test suite

The suite combines unit tests for resource algorithms with integration tests
for complete traces. Run everything from the repository root:

```powershell
make test
make check
```

[Back to the repository overview](../README.md)

## Test map

| File | Coverage |
| --- | --- |
| `test_backend.py` | Backend primitives and compatibility behavior. |
| `test_capacity.py` | Duplex NIC/disk capacities, weighted max-min fairness, and flow reallocation. |
| `test_checkpoint_modes.py` | Synchronous versus asynchronous persistence and disjoint multi-job placement. |
| `test_crossjob.py` | Donor eligibility, reservation, refusal, hosted caps, and peer-set recovery selection. |
| `test_donor_drain.py` | Donor-drain L3 pieces, stitched recovery, parity reconstruction, and owner-push fallback. |
| `test_large_scale.py` | 100K and one-million-rank runtime selection and individual/chunked execution behavior. |
| `test_public_api.py` | `simulation` and `simulator` facades plus standard output artifacts. |
| `test_simulation_log.py` | End-to-end event schema, dependencies, checkpointing, contention, failures, recovery, scale, and visualization. |
| `test_validation_postprocess.py` | Conversion of measured traces into a calibrated simulator configuration. |

## Test layers

1. Algorithm tests isolate capacity and donor registries.
2. Runtime tests construct small in-memory configs and inspect worker behavior.
3. Trace tests run full simulations and assert causal ordering in emitted JSONL.
4. Public API tests verify what external Python callers receive.
5. Validation tests protect the bridge between real measurements and simulated
   parameters.

`test_simulation_log.py` is intentionally broad because the trace is the shared
contract between execution, visualization, analysis, and validation.

## Writing tests

- Use `tmp_path` or `tmp_path_factory` for artifacts.
- Prefer deterministic seeds and small simulated workloads.
- Assert causal facts and resource limits, not incidental event IDs.
- When adding an event field, test both its value and at least one consumer.
- Reset run-scoped `DonorRegistry` and `CapacityRegistry` state in isolated
  tests that construct strategies directly.
- Put scale-only behavior in `test_large_scale.py`; avoid making every test
  instantiate a large job.

Pytest discovers this directory through `pyproject.toml`. Ruff checks the test
code with the same basic error and import rules as production code.
