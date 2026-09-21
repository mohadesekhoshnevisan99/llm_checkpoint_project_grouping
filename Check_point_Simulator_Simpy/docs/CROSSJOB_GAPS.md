# Cross-job peer-sharing: what the simulator still needs (gap list)

Context: the real system (`checkpointing/daemon/peerd.py` in the main project) implements
cross-JOB checkpoint donation — a flushing node spreads shards to OTHER jobs' idle nodes.
Measured on the 8-node fleet: k-donor scaling 107→411 MB/s (k=1..6), node-kill recovery in
11.8 s. The simulator's role is to show the COORDINATION GAP at scale (it does not appear at
8 nodes — reactive rerouting absorbs collisions while donor slack is thick).

Already in the sim (nice!): multi-job TOML runs, per-job checkpoint_every/strategy/failure_rate,
tiered strategy with intra-job rank-pairing, object-store strategy with upload contention.

## Gaps (in priority order)

1. **Cross-job donor pool** — `location_rank` pairing is intra-job. Need: a donor registry
   spanning jobs; a flushing rank reserves k donors from OTHER jobs' idle nodes.
2. **Endogenous availability** — a node must refuse hosting while its own job is flushing
   (and have a hosted-capacity cap). This is what makes the pool a shared, contended resource.
3. **Per-job PHASE (start offset)** — jobs currently all start at t=0. Need a phase field so
   coordinated-vs-uncoordinated schedules can be compared (the paper's core experiment).
4. **k-way sharding** — split a rank's checkpoint across k donors in parallel (bounded by
   NIC/SSD; equal shards fine for v1, speed-weighted later).
5. **Policy input** — accept the controller's policy JSON (interval_s, phase_s, kpeers per
   job/node) so the real controller drives the sim (same file peerd consumes).
6. **Contention metrics out** — per-run: peer-rate, fallback fraction, refusal reasons,
   flush makespans, store utilization over time (for the agreement table vs real fleet
   and vs the analytic model).

Planned implementation (by Sam's Claude agent, in NEW modules to avoid conflicts):
`checkpointing/crossjob.py` (strategy + donor registry) + `run_crossjob.py` (scenario driver)
+ `tests/test_crossjob.py`. Only integration touch: registering the strategy name.
Questions/objections welcome — nothing in your existing modules will be rewritten.

## Update 2026-07-03 — gaps 1-6 done; one new gap from cross-validation

Items 1-6 are implemented (`checkpointing/crossjob.py`, `run_crossjob.py`,
`tests/test_crossjob.py`; see results in the main repo's
`checkpointing/multilevel/results/m3d_agreement_20260703.md`). Cross-validating the sim
against the real 8-node fleet surfaced one modeling gap that matters:

7. **Per-node bandwidth classes** — the cluster model has ONE local_ssd/network rate for
   all nodes, but the real fleet mixes fast (GPU-node, ~290 MB/s disk) and slow (CPU-node,
   ~110 MB/s) donors. Homogeneous-slow makes every flush long -> overstates donor busy
   fraction -> sim peer-rate lands ~30 points below the fleet at high duty (a pessimistic
   envelope). Approximating heterogeneity via shard sizes recovers half the gap; the real
   fix is per-node (or per-node-class) rates in ClusterConfig + strategies reading the
   donor's rate. Also useful for your object-store scenarios (mixed uplinks).

## Update 2026-07-05 — three gaps from the 32-node hardware anchor (M5c)

8. **Sender-bound k-transfers**: with per-job class rates, eff bw = min(nic, k*donor_disk)
   caps correctly, but the fair-share network penalty keeps growing with k, so simulated
   flushes DEGRADE with k in the sender-bound regime while hardware improves (measured:
   sim 2.6->3.2 s vs real 2.4->1.9 s for k=4..8). The watched-donor set shouldn't add
   contention when the sender ceiling binds.
9. **Slot-wait semantics at saturation**: crossjob's async persist waits for the slot
   AFTER capture, per-rank serialized — a backlog can add a full period. The real daemon
   waits BEFORE flushing inside its interval loop. Below saturation they agree; at the
   critical band they diverge by ~20 pts. Align to pre-flush wait, or cap the added wait.
10. **Store fair-share**: object_store gives each concurrency-slot holder full bandwidth;
    hardware shows per-stream degradation before slots exhaust (out-of-sample store
    predictions optimistic 25-35%). A processor-sharing store rate would close it.

## Update 2026-07-06 — gaps 8 and 10 RESOLVED via the capacity model (new module)

`checkpointing/capacity.py`: every node gets duplex NIC (independent in/out) + disk-write
capacities; active flows get max-min fair allocations (progressive filling), recomputed
whenever the flow set changes. crossjob transfers route through it in capacity mode
(`--capacity` on run_crossjob.py): rates become emergent allocations instead of
min()+count-based slowdown; per-stream demand caps model client-side ceilings (e.g. the
store's measured 34 MB/s single-stream PUT). Validation vs the 32-node hardware anchors:
k-curve k=1 now 5.78 s vs 5.8 measured (was +35%), k>=4 saturates at the sender ceiling
instead of degrading (#8 fixed); store sweep out-of-sample 16/32 clients now 42/84 s vs
44/79 measured (was 29/59) (#10 fixed). Remaining known residual: sim slightly optimistic
at k=2-4 (real TCP/thread overheads grow with k; capacity model has a hard knee).
Gap #7 (per-node classes in ClusterConfig proper) remains open for YOUR strategies;
crossjob covers it via class_rates + capacity nodes.
