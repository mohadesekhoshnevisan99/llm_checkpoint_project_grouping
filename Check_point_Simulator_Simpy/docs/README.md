# Design documentation

This directory stores design history and research notes that explain why parts
of the simulator exist. Operational instructions live beside the code they
describe; start with the root README for current usage.

[Back to the repository overview](../README.md)

## Files

| File | Status | Purpose |
| --- | --- | --- |
| `CROSSJOB_GAPS.md` | Historical, updated over time | Tracks gaps discovered while adding and hardware-validating cross-job donor sharing, with dated resolution notes. |

`../ENGINE_REWRITE_VERIFICATION.md` is kept at the repository root because it
records a repository-wide engine rewrite and its verification, rather than one
subsystem's design.

## How to read these notes

Gap documents are decision history, not necessarily the current API contract.
Later dated updates supersede earlier planned behavior, and implementation plus
tests are authoritative when a historical statement conflicts with current
code.

When adding a new design note:

- state whether it is a proposal, decision, verification report, or open gap;
- include the date and the code paths involved;
- link to the relevant local README;
- append resolution evidence instead of silently rewriting history.
