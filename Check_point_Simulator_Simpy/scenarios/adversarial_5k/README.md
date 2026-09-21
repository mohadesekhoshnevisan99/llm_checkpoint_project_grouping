# Adversarial 5,000-node scenarios

[Final seed-7 results and dashboards](RESULTS.md)

These scenarios deliberately stress failure modes of cross-job checkpointing.
Every file contains exactly 5,000 logical ranks/nodes and uses one deterministic
seed (`7`).  They are not representative production forecasts: each isolates a
specific mechanism so a bad result is diagnosable.

| Scenario | Jobs | Cohorts | Primary stress | Expected SysName failure signature |
|---|---:|---:|---|---|
| `failure_storm.yaml` | 74 | 295 | Elevated organic failures plus frequent best-effort eviction | Early/in-flight checkpoint loss, more rollback, aborted flushes, and possibly initial-state recovery |
| `donor_scarcity.yaml` | 11 | 50 | A 4,900-node owner has only 100 nodes in other jobs available as donors | Partial grants, local fallback, low durable fraction, and initial-state recovery after whole-job eviction |
| `bandwidth_storm.yaml` | 20 | 100 | Large 32 GB/rank checkpoints on 10 GB/s NICs, 1 GB/s disks, and a near-saturated 320 GB/s shared store | Overlapping flushes, queueing, low goodput, stale/aborted flushes, and possible censoring |

All three contain 600 seconds of useful computation per job. They are
time-compressed torture tests: organic hazards are multiplied by 12, and the
explicit eviction intervals are shortened, so a quick run still sees roughly
the incident pressure of a two-hour workload. Checkpoint sizes, restart costs,
and hardware rates are not reduced, intentionally making contention harsher.
Cohorts compress the event count but preserve logical bytes and per-node
capacity.

## Weak points under test

| Weak point | Why it becomes bad | Observable result |
|---|---|---|
| Cross-job donor exclusion | A job cannot place its own cross-job shards on another cohort of that same job. A fleet-dominating job therefore sees only the small outside population. | Partial/no grants, local-SSD fallback, low durable fraction, and initial-state recovery after eviction. |
| Correlated whole-job loss | Eviction destroys every local DRAM/SSD copy for all cohorts at once. | The job needs a coherent cross-job or L3 checkpoint; otherwise every cohort returns to iteration zero. |
| Checkpoint work exceeds service rate | Large state and short cadence can inject bytes faster than donor disks, owner NICs, or the shared store can drain them. | Growing queues, aborted/stale work, checkpoint contention, low goodput, and censoring. |
| Recovery outruns durability | A second incident can arrive before an asynchronous peer/L3 wave becomes usable. | Older recovery sources, larger rollback, or initial-state fallback despite high checkpoint attempt counts. |
| Solver feasibility floor | The current v1 solver chooses at least one peer even when the outside donor pool cannot satisfy the implied placement. | A seemingly valid policy with heavy runtime fallback; solver `overload_x` and realized durable fraction must be read together. |

The first runs also exposed simulator defects rather than policy behavior:
abandoned-timeline checkpoint work could land after rollback, donor drains did
not share aggregate store capacity, and the solved L3 cadence was initialized
before the job's strategy list existed (accidentally draining every L2 wave).
These paths now have regression tests; the final matrices use job-wide
checkpoint epochs, coherent restore iterations, the enacted solved L3 gate,
shared store ingress/disk resources, and an independent trace-capacity check.

## Why these are adversarial

`failure_storm.yaml` uses a per-node hazard of `9.6e-6` per second. Across 5,000
nodes and an ideal ten-minute run, that is about 28.8 organic incidents before
runtime extension and re-execution. Best-effort eviction is independently
accelerated, with a five-second livelock guard.

`donor_scarcity.yaml` leaves only 100 nodes outside the 4,900-node owner job.
Cross-job placement cannot use the owner's own nodes. The donor registry can
therefore fill quickly; waves receiving no grant fall back to the owner's local
SSD, which is erased by whole-job eviction.

`bandwidth_storm.yaml` removes donor scarcity but makes each persist expensive.
Twenty synchronized 250-node jobs collectively flush 160 TB of raw checkpoint
state per wave. The 320 GB/s aggregate store is intentionally only just large
enough for the expected L3 stream, while each donor still has a 1 GB/s disk.
SysName slots can stagger starts, but they cannot create more physical bandwidth
when a wave lasts longer than the slot period.

## Policy matrix workflow

For each scenario, first solve SysName's policy, then run the named policies and
the solved policy separately:

```powershell
$py = "C:\Users\rrath\AppData\Local\Programs\Python\Python313\python.exe"
$scenario = "failure_storm"
$base = "scenarios/adversarial_5k/$scenario"
$out = "results/adversarial_5k/$scenario"

& $py gp_policy.py --scenario "$base.yaml" --out "$base`_policy.json" --explain
& $py run_scenario.py --scenario "$base.yaml" --seed 7 --workers 4 `
  --out "$out/named_results.json" --trace-dir "$out/traces"
& $py run_scenario.py --scenario "$base.yaml" --policy "$base`_policy.json" `
  --seed 7 --out "$out/ours_results.json" --trace-dir "$out/traces"
```

Build individual aggregate dashboards with `visualise_aggregate.py`, then build
the comparison page with:

```powershell
& $py visualise_policy_matrix.py `
  --result "$out/named_results.json" `
  --result "$out/ours_results.json" `
  --dashboard-dir "$out/dashboards" `
  --output "$out/index.html"
```

## Interpretation limits

- One seed demonstrates a mechanism; it does not estimate a failure
  probability. Use multiple seeds before making a reliability claim.
- `initial_state` means no usable checkpoint was found and the job restarted at
  iteration zero. It does not mean other recoveries lost no work.
- Whole-job eviction restores every cohort from one common checkpoint iteration;
  an aborted cohort transfer fails the run instead of silently resuming it.
- Donor-drain L3 traffic contends on shared store ingress and disk capacity. The
  trace validator independently checks aggregate store-byte conservation.
