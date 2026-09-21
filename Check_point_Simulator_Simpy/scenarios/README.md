# Scenario files — how to define a simulation

One YAML file = one experiment. Everything the simulator assumes is in the
file; nothing about the workload is hard-coded. Copy the closest existing
file and edit (`handcheck2_imbalanced.yaml` = smallest template, 3 jobs;
`realwidth73.yaml` = the full 2026 workload, 74 jobs / 197,376 GPUs).

[Back to the repository overview](../README.md)

This is the input directory for `run_scenario.py`, not the general configured
simulator. Files in [`../configs/`](../configs/README.md) enumerate concrete
jobs and use the `jobs.config` schema. Files here define workload classes,
multiple policy arms, seed sweeps, controller inputs, and optional preemption;
the scenario driver expands them into concrete jobs at run time.

## Directory organization

| Family | Purpose |
| --- | --- |
| `handcheck*.yaml` | Small, intentionally inspectable cases for checkpoint timing, tier behavior, cadence, failure, eviction, and baseline sanity checks. |
| `miniC*.yaml` | Fast iteration harness and focused sensitivity variants for the cross-job experiment. |
| `realnode26*.yaml` | 2026 workload expressed in multi-GPU-node units, plus ablations, baselines, policy sweeps, and controller variants. |
| `adversarial_5k/` | Exactly 5,000-node failure, donor-scarcity, and bandwidth stress scenarios with hypotheses and reproduction commands. |
| `sysname_targeted_pilot_500.yaml` | Controlled 500-node placement-tail counterexample: one fixed dominant-job node loss, no random failures, and focused JIT/Gemini/MegaScale comparisons. |
| `realwidth73*.yaml` | Full-width workload and horizon/sizing/controller sensitivity variants. |
| `profile_c150.yaml` | Profiling-oriented scenario rather than a paper-comparison matrix. |
| `*_policy.json` | Solved or fixed controller decisions consumed with `run_scenario.py --policy`; these are not standalone scenarios. |

Suffixes such as `_ablation`, `_balanced`, `_freqsweep`, `_slice`, `_h7200`,
and `_kbudget` identify the experimental dimension changed from the base
scenario. Read the comments at the top of a file before copying it: many
variants deliberately accelerate failures or shorten horizons and must not be
reported as full-scale results.

## Anatomy

```yaml
simulation:
  seeds: [7, 11, 13]        # each seed = one independent failure history

cluster:                    # per-NODE hardware (defaults for all classes)
  node_count: 100           # validation headroom only; jobs claim what they use
  gpu_cpu_bandwidth_gbps: 12.5   # GPU->DRAM capture rate
  network_bandwidth_gbps: 1.85   # NIC, duplex (in and out independently)
  local_ssd_bandwidth_gbps: 0.29 # sustained disk write
  object_store_bandwidth_gbps: 8.0   # legacy field, initial warm pull only
  background_nic_gbps: 0.0   # Route C: per-node persistent BACKGROUND front-end
                             # traffic (data-loading + other-tenant storage I/O)
                             # sharing the front-end NIC with our checkpoint flows.
                             # Default 0.0 = off (every existing scenario unchanged).
                             # When > 0, every node carries a standing INGRESS
                             # demand of this magnitude (data loading is a READ into
                             # the node -> consumes nic_in); on a full-duplex NIC it
                             # contends with checkpoint RECEPTION (hosted peer shards,
                             # restores, store writes / other tenants), not with the
                             # node's own outgoing checkpoint writes. Implemented as a
                             # standing max-min demand in the capacity engine, not
                             # discrete events (adds no events; sim wall unchanged).
                             # CLI override: run_scenario.py --background-nic-gbps.
                             # Sourced data-loading ~0.005 GB/s (~5 MB/s/node; see
                             # control-plane/design/workload_sources_dossier.md
                             # "Dual-fabric evidence" §D). Contention becomes visible
                             # only at 10x-1000x that rate (5 GB/s = 10% of a 50 GB/s
                             # front-end NIC): results/miniC_background/.

classes:                    # job classes; jobs are named <class><index>
  myjob:
    count: 2                # how many identical jobs of this class
    ranks: 512              # REAL width: nodes in the job (1 GPU/node in Setup A)
    cohorts: 16             # simulation granularity: 16 workers x 32 nodes each
                            #   (cohorts == ranks -> exact per-node simulation)
    checkpoint_gb_per_rank: 1.0    # per-NODE checkpoint shard (measured quantity)
    iteration_seconds: 15.0        # compute time per training iteration
    iterations: 480                # horizon: 480 x ~15s ≈ 2h simulated
    checkpoint_every: 2            # cadence in iterations (overridden by --policy)
    kpeers: 2                      # donor fan-out k (overridden by --policy)
    rpo_s: 120.0                   # max acceptable expected loss (controller input)
    rates: {nic: 1.25, disk: 0.107}   # optional per-class hardware override
    weights_gb_per_rank: 4.0       # gradient size -> all-reduce time (optional)

failures:
  per_node_per_second: 1.0e-7     # organic failure rate PER NODE
  weights: {process: 0.4, node: 0.2, spot: 0.4}   # failure-type mix
  restart_seconds: {process: 5.0, node: 20.0, spot: 30.0}

preemption:                 # whole-job spot eviction (omit block to disable)
  class_prefix: be          # victim class: jobs named '<prefix>*'
  every_s: 120.0            # class-aggregate eviction period (local state wiped)
  model: per_job_hazard     # per_job_hazard (default) | one_of_class
  seed: 3

store:                      # the shared store node (store arm + backstop)
  in_gbps: 500.0
  out_gbps: 500.0
  disk_gbps: 500.0
  stream_gbps: 1.0          # per-client stream ceiling

controller:
  slot_period_s: 60.0       # flush-slot period for coordinated arms

arms:                       # which policies this scenario compares
  ours_full:   {kpeers: true, slots: true, controller_epoch_s: 120.0}
  peer_react:  {kpeers: true, random_phases: true}
  local_only:  {kpeers: false}
  store:       {kpeers: true, store_mode: true}
  gemini:      {baseline: gemini}   # named Tier-1 baseline (see below)
```

## Dynamic job arrivals

Without a `job_arrivals` block, scenarios retain their existing behavior: each
class expands `count` jobs, normally at time zero. The legacy class-level
`arrival_s` delay remains supported. New scenarios can instead declare exact or
weighted-random arrivals. In these modes, `classes.<name>.count` is the lifetime
cap and therefore the highest range of stable IDs the class may use.

Exact simulated-time batches use `mode: scheduled`:

```yaml
job_arrivals:
  mode: scheduled
  initial: {standard: 3}
  events:
    - at_s: 120
      add: {standard: 2, frontier: 1}
    - at_s: 600
      add: {mega: 1}
```

`at_s` is absolute simulated time. Jobs at one event are admitted as one batch,
and each class receives its next stable ID (`standard0`, `standard1`, ...).
Initial plus scheduled jobs may not exceed the class cap.

Weighted random arrivals draw exact times uniformly inside a window, sort them,
then select an eligible class at each time in proportion to its weight:

```yaml
job_arrivals:
  mode: weighted_random
  initial: {standard: 3}
  random:
    start_s: 120
    end_s: 1200
    total: 8                 # default: sum of pool limits
    seed: 41
    pool:
      frontier: {weight: 1, limit: 3}
      fxl:      {weight: 2, limit: 3}
      mega:     {weight: 5, limit: 2}
```

Equal weights give equal selection probability among classes that have not hit
their limits. Higher weights make earlier/more frequent selection more likely;
they do not impose a strict priority. The effective RNG combines the simulation
seed with `random.seed` but never the arm name, so all policy arms receive the
same manifest for one seed. Pending jobs cannot train, fail, be preempted, or
act as checkpoint donors. Every arrival creates a new fixed-width job using the
class's immutable `ranks`; a running job is never resized. After training
completes, those same nodes stop training but remain available as idle
checkpoint-only donors for jobs that are still active.

`job_arrivals` and class-level `arrival_s` cannot be combined. See
`dynamic_arrivals_small.yaml` for a runnable, visualization-sized example.

### Eviction model (`preemption.model`)

Preemptible capacity is modelled as an **independent per-job reclaim hazard**
(`per_job_hazard`, the default): each job of the victim class draws its next
eviction from `Exp(1 / (every_s * count))`, i.e. a per-job mean inter-eviction
time of `every_s * count` (default scenario: 120 s x 28 be jobs = 3,360 s/job).
At the start of a run the class-aggregate rate is therefore the intended
`1/every_s`, but — unlike a fixed cluster rate — it does **not concentrate on
the survivors** as jobs complete. A job is only evictable once it has been
healthy for >= half its own mean interval (grace, anti-livelock); a draw that
lands inside the grace window or while the job is already down is dropped, not
queued. Each victim's hazard RNG is seeded from `(seed, preemption.seed,
job_id)`, so runs are reproducible and a job's eviction history is independent
of its siblings'.

`model: one_of_class` restores the legacy behaviour — one healthy victim
evicted every `every_s`, cluster-rate fixed and therefore **redistributed** onto
whoever is left (with one job remaining it is evicted every `every_s`). This
permanently traps stragglers and manufactures "never finishes" outcomes; it is
kept only for the sensitivity note.

Note the grace window makes the *realized* long-run rate lower than the nominal
`1/every_s` (a job cycles through ~`grace + mean` between evictions rather than
`mean`); `every_s` sets the nominal hazard, not a measured inter-eviction time.
The GP cost model (`gp_policy.py`) already prices this class's preemption as the
per-job hazard `1/(every_s * count)`, so `per_job_hazard` is the model the
controller's own policy is solved against.

### Named Tier-1 baselines

`{baseline: <name>}` dispatches to a published system's policy implemented in
`checkpointing/baselines.py` (each with its paper's advantage — no strawmen):
`gemini` (SOSP'23, all THREE of its tiers: local host DRAM + in-job peer-DRAM
replication every iteration as the fast path, plus a periodic **remote
persistent store** as its durability backstop — see below), `checkfreq`
(FAST'21 local SSD at the profiled <=3.5%-overhead cadence), `checknrun`
(NSDI'22 store, fixed 30 min, differential+quantized 0.1x bytes), `megascale`
(NSDI'24 status quo: async full-size store flush every 300 s), `jit`
(EuroSys'24: no periodic checkpoints; survivor-DRAM drain, 1 iteration lost,
eviction = total loss). Fixed-cadence baselines accept a `cadence_s` override
(used by the small hand-check scenarios `handcheck_gemini.yaml`,
`handcheck_gemini_evict.yaml`, `handcheck_checkfreq.yaml`,
`handcheck_checknrun.yaml`, `handcheck_megascale.yaml`, `handcheck_jit.yaml`,
`handcheck_jit_evict.yaml`). All five run against ours in
`realnode26_baselines.yaml`.

**GEMINI's store tier.** GEMINI's in-memory tiers are the fast path and its
remote persistent store is the correctness fallback, so modelling only the
in-memory tiers would strawman it: peer replicas die with their holder on node
loss, so without the store a whole-job eviction would always mean total loss.
The arm uploads to the store node every `gemini_store_every_s` seconds
(default 10,800, or three hours). The cadence remains an explicit scenario
knob and should be treated as a sensitivity axis when comparing policies.
Recovery order is newest-surviving-copy with ties broken toward the faster
tier (own DRAM -> surviving in-job replica -> store -> initial_state), exactly
as `crossjob.py` does. Note `store_mode: true` is a *different* thing: it
routes EVERY persist to the store (what `checknrun`/`megascale` do), whereas
GEMINI's store rides *beside* the in-memory fast path.

### Ten-times-smaller policy matrix

`realnode26_x10_small.yaml` and `realnode26_balanced_x10_small.yaml` are the
reproducible scaled versions of the 24,672-node mega-dominant fleet and the
56,576-node balanced fleet. They keep every job and divide each job's ranks and
simulation cohorts by about ten (rounded, minimum one). This preserves the job
mix, same-class cross-job routes, and donor-feasibility limits while reducing
both physical width and event work. Per-rank state, hardware rates, iteration
horizons, failure/preemption hazards, and RPOs are unchanged. Shared-store
capacity is divided by ten so store pressure scales with the fleet.

Each YAML includes JIT, CheckFreq, Gemini, MegaScale, Check-N-Run, a 1000-second
all-tier policy, the tiered `10 / 100 s` policy (L1 every 10 s; L2 and L3 every
100 s), and a failure-free/checkpoint-free ideal comparator. SysName/ours is the
separate `*_policy.json` generated by `gp_policy.py`; it must not be replaced by
the generic `ours_full` arm. The checked-in scaled experiments use seed 7.

```powershell
python gp_policy.py --scenario scenarios/realnode26_x10_small.yaml `
  --out scenarios/realnode26_x10_small_policy.json --explain
python run_scenario.py --scenario scenarios/realnode26_x10_small.yaml `
  --seed 7 --workers 4 --out results/named.json --trace-dir results/traces
python run_scenario.py --scenario scenarios/realnode26_x10_small.yaml `
  --policy scenarios/realnode26_x10_small_policy.json --seed 7 `
  --out results/ours.json --trace-dir results/traces
```

## Running

```bash
# one arm, one seed (fast sanity):
python3 run_scenario.py --scenario scenarios/my.yaml --arm ours_full --seed 7 \
    --out results.json --trace-dir traces/

# every arm x every seed in the YAML:
python3 run_scenario.py --scenario scenarios/my.yaml --out results.json

# the controller-decided arm: solve the cost model first, then run with it
python3 gp_policy.py --scenario scenarios/my.yaml --out my_policy.json --explain
python3 run_scenario.py --scenario scenarios/my.yaml --policy my_policy.json \
    --seed 7 --out results.json
```

For a wide scenario, run one arm/seed and build the aggregate trace dashboard
without per-node lanes:

```bash
make visualise-scenario-aggregate SCENARIO=scenarios/realnode26.yaml ARM=ours_full SEED=7
```

To visualize an existing compressed trace directly:

```bash
make visualise-aggregate \
    AGGREGATE_LOG=results/traces/realnode26_ours_full_s7.jsonl.gz \
    AGGREGATE_HTML=results/realnode26_ours_full_s7_aggregate.html
```

The policy JSON (from `gp_policy.py`) carries the controller's solved
frequency, donor budget k, and slot per job — it overrides the YAML's
hand-set `checkpoint_every`/`kpeers`. `--explain` prints the per-class
decisions plus what failures cost at the optimum.

## Inspecting a run

```bash
python3 trace_report.py    --log traces/<name>.jsonl.gz --report out.html  # one-page visual
python3 trace_validator.py traces/<name>.jsonl.gz --scenario scenarios/my.yaml  # 10 physics checks
```

Rules of thumb: keep `checkpoint_gb_per_rank`/rates at measured values and
scale only widths; pick `cohorts` so each worker stands for ≤ ~2,048 nodes
(sensitivity: results must not change when you double it); horizon long
enough that every job checkpoints ≥ 5-10 times under the solved intervals.
