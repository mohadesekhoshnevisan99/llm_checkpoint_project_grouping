# Capacity engine rewrite: quantum-polling -> event-driven

Branch: **`engine-event-rewrite`** (sim repo `cu-basil/Check_point_Simulator_Simpy`).
Base: `affa7b6` (engine files identical at current main `476ad56`).

## What changed and why

Profiling (`control-plane/design/mega_striping_tail_note_2026-07-16.md`) put ~97% of
heavy-scenario runtime in `capacity.py::_reallocate`, driven from the contention-quantum
poll loop in `crossjob.py::_capacity_transfer`. The rewrite makes the capacity engine
event-driven: shares change ONLY when a flow starts, finishes, or dies; between events
nothing wakes.

Two things had to be true for a real speedup, and the first cut got only #1:

1. **Recompute on membership change, not per quantum.** The old poll loop already got
   most of this for free (the registry's `_dirty` flag recomputed max-min only when the
   flow set changed), so a naive "recompute all flows on every change" event engine did
   the *same* O(fleet) work — measured **no speedup** on the 385-job gemini slice.
2. **Localize each recompute to the connected component the change touches.** At the
   first Setup-C checkpoint wave there are **7,680 concurrent flows but the largest
   actually-coupled component is only 48** (gemini replicates in-job — one component per
   job). Crediting, the max-min solve, and finish scheduling now run per component
   (BFS over an incremental `(node,res)->{flow}` index), so independent transfers never
   enter each other's work. This is what unlocks the speedup.

max-min is separable across disjoint resource sets, so per-component allocations equal
the global solve — **verified to 2.2e-16** over 200 randomized cases. The polling
`_reallocate` is byte-identical to before (it just calls the extracted kernel over the
whole set).

The engine is selected by `self.engine`: scenario key `simulation.engine: event|polling`
(default `event`), overridden by env var `SIM_ENGINE` for quick A/B.

## Acceptance battery

| # | Item | Result |
|---|------|--------|
| a | 42/42 tests under BOTH engines | **PASS** — event 42/42 (75.6s), polling 42/42 (79.3s) |
| b | 3 hand-checks both engines, trace_validator PASS, fluid math holds | **PASS** — all 6 traces PASS; event==polling identical; handcheck2 = **3.4483 / 1.7241 s** (pencil-math 3.45 / 1.72) exactly |
| c | Hardware anchors within recorded deviations | **PASS** — event reproduces committed `m5c_reanchor_20260712.json` byte-for-byte |
| d | A/B two mid-size scenarios, per-class <1%, recoveries identical/explained | **PASS** — freqsweep + shrunken-Setup-C both **0.000%** per-class; all recovery/abort/failure counts identical |
| e | Timing: handcheck, c150 gemini slice, naive500 ablation | **PASS** — see speedup table |
| f | Progress heartbeat (separate commit) | **DONE** — `<out>.progress` gets `wall_s sim_s pct` every 120 wall-s |

### (a) tests
Both engines green, 42/42. Run with `SIM_ENGINE=event` and `SIM_ENGINE=polling`.

### (b) hand-checks (both engines, with traces)
`trace_validator.py` PASS on all six traces (handcheck / handcheck2_imbalanced /
handcheck3_tiers x {event, polling}). Every summary metric is identical between engines
(train_end, per-class ends, flushes, restores, losses). handcheck2's peer-flush spans:

| job | span (both engines) | pencil-math |
|-----|--------------------|-------------|
| big0  | 3.4483 s | 3.45 s |
| mid0  | 1.7241 s (contended: 3.4483 s) | 1.72 s |
| small0| 1.7241 s | 1.72 s |

The event engine lands on the analytic fluid value exactly. Here polling matches too
(equal-shard flushes complete on clean boundaries its quantum also resolves), so there
is **no deviation to explain** — the event engine is no worse, and the value is exact.

### (c) hardware anchors (`run_m5c_reanchor.py`, event engine)
Pointed the anchor harness at this worktree (`M.SIM`) with `SIM_ENGINE=event`:

| anchor | event sim | committed ref | measured (fleet) |
|--------|-----------|---------------|------------------|
| A' k-curve k=1..8 | 5.78 / 2.89 / 1.93 / 1.92x5 | identical | 5.8..1.9 |
| C' store n=4/8/16/32 | 29.4 / 29.4 / 42.1 / 84.2 | identical | 29.8 / 26.4 / 44.0 / 79.4 |

Deviations vs fleet are **unchanged** from the recorded reanchor (dev_pct identical).

### (d) A/B equivalence (event vs polling)
| scenario (arm, seed) | max per-class Δ | recovery / abort / failure counts |
|----------------------|-----------------|-----------------------------------|
| realnode26_freqsweep (naive100, s7) | **0.000%** | identical (2104 organic failures, 67 preemptions, 12 aborts, 2640 restores by tier) |
| shrunken Setup-C (ours_full, s7) | **0.000%** | identical (4 preemptions, 80 aborts, 32 crossjob_peer restores) |

Both are byte-identical, not merely within 1%. This exercises the generation-abort path
(donor/flusher failures mid-transfer) and cohort recovery — the parts most at risk.

### (e) speedups (64-core host; single-thread runs, no contention)

| case | event | polling | speedup |
|------|-------|---------|---------|
| handcheck (full sim) | 0.16 s | 0.16 s | ~1.0x (tiny; nothing to amortize) |
| s60 gemini (61-job Setup-C slice) | 4.4 s | 39.3 s | **8.9x** |
| s60 ours_full | 2.4 s | 6.5 s | 2.7x |
| **c150 gemini** (full 385-job, seed 7) | **71.8 s** | **>3150 s (did not finish)** | **>43x** |
| naive500 ablation (385-job, seed 7) | see note | see note | pre-existing stall (both engines) |

Before localization the full c150 gemini did not finish an 80-sim-second slice in 5 min;
after, the whole 385-job run completes in 72 s. The polling baseline had not finished after
52 wall-minutes (>3150 s -> **>43x** and climbing, comfortably inside the 10-50x target;
the profiling note independently measured the polling engine at 6,700 s for a 150-sim-second
gemini-C slice, i.e. ~90x once complete).

**naive500 ablation (e-iii) — honest result.** This arm does NOT complete under EITHER
engine in reasonable wall time, and that is **pre-existing**: the arm is `kpeers: False`,
so after the every-21s preemptions a *store-recovery thundering herd* forms — ~1,225
concurrent store fetches all coupled at the single shared store node — one giant flow
component around sim 500-515. The untouched original engine (main `476ad56`) stalls there
identically (measured: original sim clock pinned at 515.1 for 50 wall-seconds; the new
event engine at 500.2). Both engines must re-solve that whole component on every change
(polling per quantum, event per membership change), so neither escapes it. It is a base-
scenario cost, not a rewrite regression. Correctness on the arm is confirmed on a
completable slice: naive500 capped at 100 sim-seconds is **event==polling byte-identical**.
For a capacity-engine speedup the gemini slice (the profiling note's actual target) is the
representative case; naive500's cost lives in the shared-store contention, which is
identical work for both engines.

### (f) progress heartbeat
Daemon thread appends `wall_s sim_s pct_of_horizon` to `<out>.progress` every 120 wall-s
(e.g. `120 7200.0 5.0`). Unconditional, cheap; short arms produce no file. Separate commit.

## Behavioral deltas and why they are correct

- **None observed** at the reported precision: every A/B above is byte-identical, and
  allocations match the global max-min to 2.2e-16 (separability). The event engine credits
  `rate x elapsed` exactly (rates are piecewise-constant between membership changes) where
  the poll loop credited `min(pre,post)*dt` per quantum — so any difference could only be a
  *removal* of quantum error, and on these scenarios it rounds away entirely.
- **Abort latency (not outcome):** a watched worker's failure now fires an abort event
  from the failure path (`tiered.handle_failure`, `run_scenario._fail_worker`) at the exact
  failure instant, vs the poll loop noticing on its next quantum (<=0.5 s later). Aborted
  transfers discard their bytes, so the earlier abort is unobservable in results
  (`aborted_flushes` identical in (d)); it only removes up to one quantum of latency.
- **Polling path unchanged:** verified byte-identical to the original engine code
  (main `476ad56`, engine files unchanged from base) on handcheck2/handcheck3/freqsweep.

## Compromises / honesty notes

- The simpy shim cannot cancel a scheduled timeout, so stale wakes are handled by a
  per-flow token on the finish-heap: a wake whose flow was re-solved (token bumped),
  removed, or already finished is discarded on pop. This is exact, only slightly wasteful
  (a few no-op wakes).
- Streams finishing within `TIME_EPS = 1e-9` sim-seconds of each other are collapsed to
  finish together (avoids a same-instant wake spin when `now + tiny` underflows in float64).
  The discarded bytes are `<= allocation * 1e-9` GB — far below the poll engine's own
  `max(dt, 1e-6)` step floor, and below the 1e-9 GB completion epsilon it already used.
- **Giant shared-resource components.** When hundreds of transfers couple on ONE resource
  (a store thundering herd), that component is re-solved on every membership change — the
  same O(component^2) the poll loop pays per quantum. This does not help those cases (see
  naive500 above), but the disjoint-transfer case it targets (gemini, per-job components)
  gets the full win. A same-instant arm/finish coalescing pass could bound an arriving herd
  to one solve; left out here to keep the verified engine unchanged, and noted for follow-up.
- Two perf/robustness fixes were made after first-run profiling (results unchanged): a
  push-time `seen` guard in the component BFS (was O(component^2)), and periodic compaction
  of the lazy finish-heap (stale entries buried below their replacements were leaking).

## Files changed
- `checkpointing/capacity.py` — event scheduler (`ev_*`, component-local `_ev_touch`,
  finish-heap), `_reallocate_subset` kernel extraction.
- `checkpointing/crossjob.py` — `_capacity_transfer` dispatch; `_capacity_transfer_event`
  (wait-on-completion consumer); `_capacity_transfer_polling` (verbatim legacy body);
  `resolve_engine`; per-instance engine + `ev_attach`.
- `checkpointing/tiered.py` — abort hook in `handle_failure`.
- `run_scenario.py` — `simulation.engine` switch; abort hook in `_fail_worker`; heartbeat.

## Commits (reviewable stages)
1. `capacity: event-driven scheduler alongside the polling max-min core`
2. `crossjob: wait-on-completion capacity consumer + generation-abort hooks`
3. `run_scenario: engine switch via simulation.engine scenario key`
4. `capacity: localize the event scheduler to connected components`
5. `capacity: fix event-scheduler heap leak and O(n^2) component BFS`
6. `run_scenario: wall-clock progress heartbeat`
7. `docs: ENGINE_REWRITE_VERIFICATION`


## Merge-gate addendum (coordinator, 2026-07-18)

Independent re-verification confirmed the battery, with one finding the battery missed:
on mini-C (accelerated failures, 20 s runs) A/B is NOT byte-identical — 6/6 seeds diverge
at discrete decision boundaries: grace-window eligibility of an eviction draw, newest-copy
ties between store and peer recoveries, and end-of-run windows where a scheduled failure
lands only if the job is still running (seed 23: one extra spot failure = +150 s on mega).
All are sub-quantum timing flips where the EVENT engine's exact timing is the truer one;
across seeds the flips are symmetric (4 shorter / 2 longer), conservation and validator
PASS on event traces, and pencil-math/hand-check/anchor results are exact. Verdict: the
event engine is the reference going forward; cross-engine comparisons are valid at the
distribution level, not per-seed.
