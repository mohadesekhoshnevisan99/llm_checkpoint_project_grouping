# SysName-targeted 500-node pilot

This is a controlled architectural counterexample, not a general stress test.
It asks whether checkpointing methods that may use in-job replicas or a direct
object-store path can outperform SysName when almost the entire fleet belongs
to one protected job.

## Isolation controls

| Control | Setting | Why |
|---|---|---|
| Fleet | `dominant0`: 480 nodes; 20 one-node helper jobs | Only 20 nodes exist outside the dominant job. |
| Granularity | 500 cohorts for 500 ranks | Node loss deterministically removes the failed rank's local copy. |
| Failure | `dominant0`, cohort 479, after iteration 36 | One fixed progress-aligned node failure, consumed once. |
| Random failures | Disabled | Every non-ideal strategy receives the same event. |
| Whole-job eviction | None | In-job replication remains a valid competing mechanism. |
| Store | 1,000 GB/s aggregate | Direct-store strategies are not punished by an unrelated store bottleneck. |
| Warm-up | 36 completed iterations before failure | Gemini replicas, MegaScale's 300-second upload, and SysName waves can land first. |

The failure occurs before iteration 36's checkpoint is dispatched. SysName's
latest earlier local fallback for cohort 479 is destroyed by the node loss; the
cohort has no outside-job peer shard or L3 donor-drained piece. Gemini can use a
surviving in-job replica, JIT can drain surviving data-parallel state, and
MegaScale can use its earlier object-store checkpoint.

The selected cohort is the placement-tail adversary. A later experiment should
sweep or uniformly sample failed ranks to estimate how often the counterexample
occurs; this pilot only verifies that the architectural failure mode exists.

## Measured pilot result (seed 7)

All five traces passed strict validation with zero warnings, and every
non-ideal arm received the same single scheduled node failure. No job was
censored. The most informative completion metric is the dominant job: the
long-running helper jobs intentionally hide most recovery differences in the
aggregate makespan.

| Strategy | Fleet makespan | Dominant job | Dominant overhead vs failure-free ideal | Recovery source | Lost iterations | Failures / restores | Rank-weighted flushes |
|---|---:|---:|---:|---|---:|---:|---:|
| SysName (`ours_gp_rpo`) | 985.1 s | 985.1 s | +382.7 s (+63.53%) | Initial state | 36 | 1 / 1 | 20,860 |
| JIT | 700.0 s | 632.5 s | +30.1 s (+5.00%) | Surviving peer DRAM | 1 | 1 / 1 | 0 |
| Gemini | 700.2 s | 644.0 s | +41.6 s (+6.91%) | In-job replica | 2 | 1 / 1 | 28,320 |
| MegaScale | 700.0 s | 682.7 s | +80.3 s (+13.33%) | Object store | 6 | 1 / 1 | 520 |
| Ideal | 700.0 s | 602.4 s | reference | None (failure suppressed) | - | 0 / 0 | 0 |

This supports the intended claim only as an **existence proof**. Relative to
SysName, dominant-job completion was 35.79% faster with JIT, 34.63% faster
with Gemini, and 30.70% faster with MegaScale. It does not estimate how often
this placement-tail failure occurs in a representative workload.

The trace explains the SysName result directly. Before the failure, rank 479
had 17 local DRAM captures and 14 local-SSD fallback copies, but zero peer or
store/L3 copies. Its latest local capture (iteration 34) was erased by the node
loss at 361.902 s. Recovery therefore used iteration-zero initial state after
the 20-second restart. Across all ranks, SysName recorded 1,510 peer flushes
and 19,350 local fallbacks (7.24% durable by the current metric), plus 28 L3
drains tracked separately.

Interpret the cross-policy metrics carefully:

- The ideal arm suppresses the scheduled failure, so its reference gap includes
  failure/restart cost as well as checkpoint overhead.
- JIT's simulator model explicitly grants a one-iteration survivor-DRAM
  recovery for this single-node fault.
- Gemini's displayed `durable_frac=0` is a classification artifact: its
  `gemini_replica` path is not named `peer` or `store`, although that in-job
  copy successfully survived this node failure.
- MegaScale is deliberately evaluated with an uncongested 1,000 GB/s store to
  isolate SysName's outside-job donor scarcity.
- Flush counts are rank-weighted operations, not checkpoint-wave counts.

Open the [cross-policy comparison dashboard](../results/sysname_targeted_pilot_500/index.html)
or any linked per-strategy aggregate dashboard for the operation timelines,
goodput, checkpoint traffic, recovery sources, and per-job detail.

### Local DRAM-to-SSD rerun

The dashboard's **Owner-local checkpoint writes** graph counts completed
`checkpoint_dram_to_local_ssd_chunk` bytes per logical job. In the seed-7
rerun, SysName wrote 19,330.2 GB locally: 19,328.0 GB from `dominant0` and
2.2 GB from helper jobs. Every recorded SysName local write was a
`local_fallback` after peer placement was unavailable. JIT, Gemini,
MegaScale, and ideal recorded zero owner-local DRAM-to-SSD writes in this
scenario.

These values are cumulative physical write traffic, not end-of-run SSD
occupancy. Rewrites after rollback count again. Cross-job peer-SSD writes,
object-store traffic, and recovery reads are deliberately excluded. Open the
[rerun comparison](../results/sysname_targeted_pilot_500/local_ssd_rerun/index.html)
and select the SysName dashboard to inspect the graph.

## Reproduce

```powershell
$py = "C:\Users\rrath\AppData\Local\Programs\Python\Python313\python.exe"
$scenario = "scenarios/sysname_targeted_pilot_500.yaml"
$policy = "scenarios/sysname_targeted_pilot_500_policy.json"
$out = "results/sysname_targeted_pilot_500"

& $py gp_policy.py --scenario $scenario --out $policy --explain
& $py run_scenario.py --scenario $scenario --policy $policy --seed 7 `
  --out "$out/ours_results.json" --trace-dir "$out/traces"
& $py run_scenario.py --scenario $scenario --seed 7 --workers 2 `
  --out "$out/named_results.json" --trace-dir "$out/traces"
```
