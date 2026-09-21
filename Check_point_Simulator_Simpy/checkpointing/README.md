# Checkpointing package

This package owns how checkpoint state is captured, placed, contended, lost,
and restored. Job runtimes call a strategy through a common behavioral
interface; they do not implement persistence policy themselves.

[Back to the repository overview](../README.md)

## Package structure

| File | Responsibility |
| --- | --- |
| `__init__.py` | Public exports and `create_checkpoint_strategy`, the name-to-implementation factory used by job runtimes. |
| `base.py` | `CheckpointWorker` protocol: the worker state every strategy is allowed to read or update. |
| `object_store.py` | GPU-to-DRAM capture followed by synchronous or asynchronous remote upload. Remote completed copies survive all modeled local failures. |
| `tiered.py` | Local/paired DRAM and SSD copies, chunked persistence, failure flushing, and newest-copy recovery. |
| `capacity.py` | Run-wide duplex NIC and disk-write capacity registry with weighted max-min fair allocation. It supports polling and event-driven transfers. |
| `crossjob.py` | Cross-job donor registry, peer selection, sharded flushes, backstops, donor draining, placement modes, and cross-job recovery. |
| `baselines.py` | Named published-policy baselines and their cadence rules, implemented as experiment arms. |

## Strategy lifecycle

A strategy instance is created once per job by `jobs.runtime`. It receives the
backend, cluster configuration, shared event logger, object-store resource, and
the workers belonging to that job.

The runtime then uses these operations:

1. `checkpoint(worker, job, iteration=...)` captures and persists a shard.
2. `gpu_slowdown(...)` and `network_slowdown(...)` expose active contention to
   training work.
3. `handle_failure(worker, failure_type)` removes invalid in-flight or stored
   state and reports flushed bytes.
4. `recover(worker, job)` selects a surviving copy, fetches it if necessary,
   and restores it to the GPU.

`CheckpointWorker` deliberately uses structural typing. Detailed workers and
scenario cohort workers can participate as long as they expose the protocol's
fields.

## Registered strategy names

The factory currently recognizes:

| Config value | Implementation | Placement |
| --- | --- | --- |
| `object_store` | `ObjectStoreCheckpointStrategy` | DRAM staging plus remote store |
| `cpu_tiered` | `TieredCheckpointStrategy` | local and adjacent-peer DRAM |
| `ssd_tiered` | `TieredCheckpointStrategy` | local/peer DRAM and SSD |
| `paired_tiered` | `TieredCheckpointStrategy` | explicitly paired DRAM and SSD |
| `local_tiered` | `TieredCheckpointStrategy` | local-tiered alias normalized to SSD tiering |
| `crossjob_peer` | `CrossJobPeerStrategy` | donors from other jobs plus configured backstop |

Add new names in `checkpointing/__init__.py`; otherwise config loading may
succeed but runtime construction will reject the strategy.

## Copy and failure model

`CheckpointCopy` records owner rank, location rank, storage tier, iteration,
size, and completion time. A copy is recoverable only after its write event has
completed.

| Failure type | DRAM | Local SSD | Remote/other-node copy |
| --- | --- | --- | --- |
| process | retained | retained | retained |
| node | flushed | retained | retained |
| spot | flushed | flushed | retained if its holder remains available |

Recovery maximizes checkpoint iteration first. Storage speed is only a
tie-breaker. In-flight state is never treated as durable.

## Capacity model

`CapacityRegistry` gives each node independent `in`, `out`, and `disk`
capacities. A transfer opens one or more `Flow` objects listing every resource
it traverses. Weighted progressive filling recomputes allocations when flows
start or end, so a flow is limited by all of its bottlenecks without also
applying the older task-count slowdown.

Registry state is keyed by `run_id`. Tests and independent runs must call the
corresponding `reset` method when bypassing the normal driver setup.

## Cross-job-specific structure

`DonorRegistry` is the cluster-level availability board. Donors publish whether
they are available, whether their own job is flushing, and how many hosted
pieces they hold. Matching can use the centralized registry or the bounded
probe baseline. Scenario arms further select spreading, window bias, in-job
placement, donor-drain versus owner-push backstops, and controller phasing.

Most cross-job knobs are class attributes set by `run_scenario.py` or
`run_crossjob.py`. Drivers must set every relevant knob for each run to avoid
leaking state from a preceding arm.

## Change checklist

- Keep capture, persistence, and recovery events causally ordered.
- Update the factory and this README when adding a strategy.
- Preserve failure-generation checks around interruptible transfers.
- Avoid combining capacity allocations with task-count bandwidth division.
- Reset run-scoped registries between experiments.
- Run checkpoint, capacity, cross-job, donor-drain, and simulation-log tests.
- Validate a representative cross-job trace with `trace_validator.py`.
