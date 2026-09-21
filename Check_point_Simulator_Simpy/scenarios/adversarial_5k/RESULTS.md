# Adversarial 5,000-node results

Final run: 2026-07-21, seed 7, event engine. Each scenario has exactly 5,000
logical nodes and 600 seconds of ideal useful computation per job. Failure
hazards are time-compressed; checkpoint sizes, restart times, and hardware
rates are not reduced. These are deterministic torture tests, not production
probability estimates.

Open the interactive policy comparisons:

- [Failure storm](../../results/adversarial_5k/failure_storm/index.html)
- [Donor scarcity](../../results/adversarial_5k/donor_scarcity/index.html)
- [Bandwidth storm](../../results/adversarial_5k/bandwidth_storm/index.html)

Every strategy name in those pages links to its aggregate and per-job trace
dashboard.

## How to read the tables

- Completion is the slowest job's training end. `>=10,000` means at least one
  job was censored at the simulation cap, so overhead is a lower bound and
  goodput is an upper bound.
- A failure is one modeled incident. A restore is one cohort recovery; a
  whole-job eviction can therefore create many restores.
- Initial is the number of cohort restores for which no usable checkpoint was
  available. Its percentage is of restores, not failures or nodes.
- Durable is the fraction of rank-weighted checkpoint outcomes that reached
  the configured durable path; aborted attempts are included in the
  denominator. It does not measure job coverage or checkpoint freshness by
  itself.
- Pressure is `aborted flushes / local fallbacks / partial grants`.

## SysName across the three adversaries

| Scenario | Completion | Overhead | Goodput | Censored | Failures / restores | Initial, count (%) | Durable | Rank flushes | Pressure | Worst rollback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Failure storm | >=10,000 s | >=1,534.5% | <=6.12% | `mega0` | 382 / 448 | 38 (8.5%) | 99.41% | 34,908 | 207 / 0 / 0 | 13 |
| Donor scarcity | >=10,000 s | >=1,520.0% | <=6.17% | `mega0` | 188 / 2,411 | 2,282 (94.6%) | 100% | 500 | 0 / 0 / 0 | 12 |
| Bandwidth storm | 837.7 s | 32.57% | 75.43% | none | 9 / 9 | 2 (22.2%) | 89.69% | 64,000 | 1,150 / 5,450 / 13 | 4 |

The headline is that the three metrics answer different questions. SysName's
99-100% durable fraction in the first two scenarios does not guarantee that
the dominant job has a reachable checkpoint. Donor scarcity is the clearest
counterexample: the run recorded 500 durable rank-weighted outcomes, while the
4,900-node owner recorded 2,282 initial-state restores and never completed.

## Scenario A: failure storm

Topology: 74 jobs and 295 cohorts, including one 3,306-node mega job. Organic
failure pressure is high and the 28 best-effort jobs also face independent
evictions.

| Strategy | Completion | Failures / restores | Recovery sources, count (% restores) | Initial | Durable | Rank flushes | Pressure | Worst rollback |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| SysName | >=10,000 s | 382 / 448 | DRAM 279 (62.3%); peer 131 (29.2%); initial 38 (8.5%) | 38 | 99.41% | 34,908 | 207 / 0 / 0 | 13 |
| JIT | >=10,000 s | 514 / 697 | JIT peer 329 (47.2%); initial 368 (52.8%) | 368 | n/a | 0 | 0 / 0 / 0 | 60 |
| CheckFreq | >=10,000 s | 499 / 683 | DRAM 281 (41.1%); SSD 4 (0.6%); initial 398 (58.3%) | 398 | 0% | 69,026 | 0 / 69,026 / 0 | 60 |
| Gemini | >=10,000 s | 513 / 696 | DRAM 296 (42.5%); replica 4 (0.6%); initial 396 (56.9%) | 396 | 0% | 59,108 | 0 / 0 / 0 | 60 |
| MegaScale | >=10,000 s | 403 / 494 | DRAM 272 (55.1%); store 104 (21.1%); initial 118 (23.9%) | 118 | 100% | 1,421 | 0 / 0 / 0 | 34 |
| Check-N-Run | >=10,000 s | 498 / 681 | DRAM 271 (39.8%); initial 410 (60.2%) | 410 | n/a | 0 | 0 / 0 / 0 | 60 |
| Naive 1000/all | >=10,000 s | 498 / 681 | DRAM 271 (39.8%); initial 410 (60.2%) | 410 | n/a | 0 | 0 / 0 / 0 | 60 |
| Naive 10/100 | >=10,000 s | 387 / 455 | DRAM 283 (62.2%); peer 103 (22.6%); stitch 12 (2.6%); initial 57 (12.5%) | 57 | 83.85% | 12,091 | 429 / 1,524 / 1 | 41 |
| Ideal | 611.8 s | 0 / 0 | none | 0 | n/a | 0 | 0 / 0 / 0 | n/a |

Every non-ideal strategy censored `mega0`, so their aggregate completion rows
are bounds. Per-class completion is more informative:

| Strategy | FXL | Frontier | Standard-G | Standard-C | Best-effort | Mega |
|---|---:|---:|---:|---:|---:|---:|
| SysName | 987.5 | 762.7 | 719.4 | 694.6 | 1,185.1 | 10,000 cap |
| JIT | 690.2 | 705.9 | 651.3 | 647.4 | 9,133.8 | 10,000 cap |
| CheckFreq | 789.8 | 689.3 | 674.0 | 670.2 | 9,140.1 | 10,000 cap |
| Gemini | 789.9 | 686.7 | 674.0 | 670.2 | 9,140.1 | 10,000 cap |
| MegaScale | 1,369.8 | 1,082.3 | 719.1 | 772.6 | 2,609.3 | 10,000 cap |
| Check-N-Run | 1,442.9 | 1,536.5 | 970.4 | 770.7 | 9,133.8 | 10,000 cap |
| Naive 1000/all | 1,442.9 | 1,536.5 | 970.4 | 770.7 | 9,133.8 | 10,000 cap |
| Naive 10/100 | 1,130.4 | 687.7 | 674.2 | 670.2 | 1,409.9 | 10,000 cap |
| Ideal | 603.2 | 604.7 | 605.9 | 611.8 | 609.6 | 601.6 |

SysName gives the best recovery quality here, but the mega job's aggregate
organic hazard still overwhelms forward progress. The fall in active nodes in
the dashboard is normal job completion: smaller classes finish and release
their nodes while `mega0` remains as the final censored straggler.

## Scenario B: donor scarcity

Topology: one 4,900-node owner plus ten 10-node jobs, represented by 50
cohorts. Cross-job placement excludes the owner's own job, leaving only 100
outside nodes as possible donors.

| Strategy | Completion | Recovery sources | Initial | Durable / rank flushes | Pressure | Worst rollback |
|---|---:|---|---:|---:|---:|---:|
| SysName | >=10,000 s | initial 2,282; DRAM 129 | 2,282 (94.6%) | 100% / 500 | 0 / 0 / 0 | 12 |
| JIT | >=10,000 s | initial 2,280; JIT DRAM 131 | 2,280 (94.6%) | n/a / 0 | 0 / 0 / 0 | 2 |
| CheckFreq | >=10,000 s | initial 2,282; DRAM 129 | 2,282 (94.6%) | 0% / 5,900 | 0 / 5,900 / 0 | 2 |
| Gemini | >=10,000 s | initial 2,282; DRAM 129 | 2,282 (94.6%) | n/a / 0 | 0 / 0 / 0 | 2 |
| MegaScale | >=10,000 s | initial 2,283; DRAM 128 | 2,283 (94.7%) | 100% / 200 | 0 / 0 / 0 | 23 |
| Check-N-Run | >=10,000 s | initial 2,283; DRAM 128 | 2,283 (94.7%) | n/a / 0 | 0 / 0 / 0 | 23 |
| Naive 1000/all | >=10,000 s | initial 2,283; DRAM 128 | 2,283 (94.7%) | n/a / 0 | 0 / 0 / 0 | 23 |
| Naive 10/100 | >=10,000 s | initial 2,282; DRAM 129 | 2,282 (94.6%) | 100% / 490 | 0 / 0 / 0 | 2 |
| Ideal | 617.3 s | none | 0 | n/a / 0 | 0 / 0 / 0 | 0 |

Every non-ideal policy censored `mega0`; all have 1,520% lower-bound overhead
and at most 6.17% goodput. SysName logged 22,142 `donor_failed` placement
refusals. Its solved `k=1` policy is therefore not physically sufficient even
though aggregate flush-outcome durability is 100%. The final f3-correct
run has zero successful SysName L3 drains, ten aborted drains, and zero drained
bytes.

## Scenario C: bandwidth storm

Topology: twenty homogeneous 250-node jobs and 100 cohorts. A raw synchronized
wave contains 160 TB. Each node has a 10 GB/s NIC and 1 GB/s disk; the shared
store is capped at 320 GB/s. The v1 solver reports `overload_x=2.67`.

| Strategy | Completion | Overhead | Goodput | Failures / restores | Recovery sources | Initial | Durable | Rank flushes | Pressure | Worst rollback |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| SysName | 837.7 s | 32.57% | 75.43% | 9 / 9 | DRAM 7; initial 2 | 2 | 89.69% | 64,000 | 1,150 / 5,450 / 13 | 4 |
| JIT | 761.9 s | 20.57% | 82.94% | 11 / 11 | JIT peer 11 | 0 | n/a | 0 | 0 / 0 / 0 | 1 |
| CheckFreq | 800.0 s | 26.60% | 78.99% | 10 / 10 | DRAM 9; initial 1 | 1 | 0% | 92,750 | 0 / 95,000 / 0 | 3 |
| Gemini | 724.7 s | 14.69% | 87.19% | 8 / 8 | DRAM 7; initial 1 | 1 | 0% | 194,250 | 0 / 0 / 0 | 2 |
| MegaScale | 1,195.7 s | 89.22% | 52.85% | 11 / 11 | DRAM 4; initial 7 | 7 | 100% | 9,250 | 0 / 0 / 0 | 20 |
| Check-N-Run | 1,268.8 s | 100.79% | 49.80% | 11 / 11 | initial 10; DRAM 1 | 10 | n/a | 0 | 0 / 0 / 0 | 39 |
| Naive 1000/all | 1,268.8 s | 100.79% | 49.80% | 11 / 11 | initial 10; DRAM 1 | 10 | n/a | 0 | 0 / 0 / 0 | 39 |
| Naive 10/100 | 860.4 s | 36.16% | 73.44% | 10 / 10 | DRAM 9; initial 1 | 1 | 97.29% | 29,500 | 800 / 0 / 0 | 2 |
| Ideal | 631.9 s | 0% | 100% | 0 / 0 | none | 0 | n/a | 0 | 0 / 0 / 0 | n/a |

This scenario distinguishes performance from protection. Gemini and JIT are
fastest, but their successful restores are volatile-memory paths rather than a
durable guarantee. SysName is slower but limits worst rollback to four
iterations with 89.69% admitted-flush durability. Its 1,150 aborts, 5,450 local
fallbacks, and 13 partial grants show the intended service-pressure failure.
Naive 10/100 is the closest durable competitor in this seed; it is 22.7 seconds
slower but has higher flush-outcome durability and one fewer worst-case lost
iteration.

## Simulator findings and fixes

The initial torture runs found three simulator problems that would have made
the comparison invalid:

1. Queued checkpoint work from an abandoned rollback timeline could publish a
   stale or future copy. Checkpoints now bind to a job-wide epoch, in-flight
   transfers are interrupted on rollback, future copies are rejected, and
   whole-job cohorts recover to one coherent iteration.
2. Donor-to-L3 drains bypassed aggregate store ingress/disk capacity. Every
   drain now consumes the shared store resources, and validator invariant I13
   independently checks byte conservation over overlapping store writes.
3. The policy's `f3/f2` drain gate was assigned before `rt.all_strategies` was
   populated, silently draining every peer wave. The gate is now initialized
   directly on each runtime strategy and covered by a regression test.

The final 27 traces pass all hard validator invariants and warnings. The full
test suite passes 94 tests, and Ruff reports no lint errors.

## Limits

- One seed demonstrates mechanisms but does not provide confidence intervals.
- Finished jobs remain available as donors; interpret this as retained
  checkpoint-daemon/SSD capacity rather than guaranteed node teardown behavior.
- Cohorts preserve logical ranks, bytes, and capacity but correlate failures
  within each represented group.
- The generated policies use the legacy v1 solver, which currently forces
  `k >= 1`; it should eventually expose an explicit infeasible result.
