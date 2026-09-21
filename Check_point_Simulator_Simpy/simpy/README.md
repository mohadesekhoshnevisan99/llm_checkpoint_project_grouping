# Local SimPy compatibility layer

This directory shadows the external `simpy` import with the small subset of the
API required by this repository. Its event queue is backed by
`SerializableSimpy.core.Environment`.

[Back to the repository overview](../README.md)

## What is implemented

The entire implementation currently lives in `__init__.py`:

- `Environment`, timeouts, processes, and manual events;
- `AllOf` and `AnyOf` conditions;
- process interrupts;
- FIFO `Resource` and priority-queued `PriorityResource`;
- capacity-tracking `Container`;
- compatibility module objects for `simpy.events` and
  `simpy.resources.resource/container` imports.

## Why this directory matters

Because the repository root is on Python's import path, `import simpy` resolves
here during normal local runs. This is intentional. Behavior changes in this
file can affect every simulator layer even if the caller appears to use the
third-party SimPy API.

`simulation/backend.py` wraps this API for most modern domain code. The legacy
node model still imports `simpy` directly.

## Scope and constraints

This is a compatibility subset, not a complete reimplementation of upstream
SimPy. Add primitives only when repository code needs them, and match the
upstream behavior relied upon by callers: callback ordering, request release,
interrupt propagation, and same-time event priority are especially sensitive.

Run `tests/test_backend.py` plus the full test suite after changes. Scheduler
bugs often appear as incorrect domain ordering rather than a direct exception.
