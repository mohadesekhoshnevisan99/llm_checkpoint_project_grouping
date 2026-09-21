# Nodes package

This package contains the event record used by all current runtimes and the
resource-level node model used by the legacy single-job path in `main.py`.

[Back to the repository overview](../README.md)

## Package structure

| File | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports event logging, links, base nodes, object storage, and training nodes. |
| `node.py` | `SimulationEvent`, `EventLogger`, shared `ObjectStore`, base `Node`, and point-to-point `Link`. |
| `training_node.py` | Reusable compute, communication, checkpoint, and restore operations for one legacy distributed-training rank. |

## Two roles in one package

`SimulationEvent` and `EventLogger` are shared infrastructure and are used by
the modern configured and scenario runtimes. `Node`, `Link`, `ObjectStore`, and
`TrainingNode` are the older concrete resource model used by `main.py` when it
runs without `--config`.

This distinction matters when editing the code: adding a method to
`TrainingNode` will not automatically affect the configured runtime in
`jobs/runtime.py`.

## Event schema

Each event records timing, rank and placement identity, operation/category,
resources, iteration, transfer endpoints and size, failure type, and a flexible
`details` mapping. `EventLogger.write_jsonl` sorts events by start, end, and
event ID before writing JSON Lines.

The schema is consumed by:

- `visualise.py` for timelines;
- `trace_report.py` for cross-job analysis;
- `trace_validator.py` for physical invariants;
- tests that assert operation ordering and capacity behavior.

Treat field names and operation names as a compatibility contract. When they
change, update all consumers together.

## Legacy resource model

`Node` owns CPU/GPU resources and DRAM/SSD containers. `Link` represents a
shared full-bandwidth connection between exactly two nodes. `TrainingNode`
builds higher-level resource operations and models CPU/GPU overlap slowdown in
small time quanta.

The legacy orchestration order, checkpoint cadence, and injected failures stay
in `main.py`; the node class should remain a reusable operation layer.
