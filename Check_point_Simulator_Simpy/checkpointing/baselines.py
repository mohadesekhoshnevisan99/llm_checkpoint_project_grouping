"""Named Tier-1 baseline checkpoint policies, as arms against our system.

Each baseline is a faithful, no-strawman model of a published system's mechanism
(see the survey rows). A baseline is expressed as a `Baseline` record: an optional
strategy subclass, a set of arm flags (kpeers/store_mode/slots reused from the
scenario driver), class attributes forced on CrossJobPeerStrategy (e.g. store
byte scales), a per-job checkpoint-frequency rule, and an optional driver mode
(jit). run_scenario.run_arm resolves the record and wires it; existing (non-
baseline) arms are untouched.

Fairness note: every baseline gets its paper's advantages —
  gemini     in-job peer-DRAM replication, NO all-reduce coupling (interleaving);
  checkfreq  profiled cadence bounding visible overhead <= 3.5%;
  checknrun  differential+quantized store writes (10x smaller), 2x on restore;
  megascale  async DRAM-stage + fixed 300 s store flush (production status quo);
  jit        recovery from surviving DP replicas, exactly one iteration lost;
             where ZeRO sharding leaves no intact replica, the paper's OWN
             degraded mode: periodic checkpointing at c* = sqrt(N*f/(2*o))
             (Sec. 5.2 eq. 3) to the store — jit_pc_fallback, ON by default.

New module only; the engine, the student runtime, and existing arms are untouched.
Author: Sam's agent.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from . import parallelism
from .capacity import RES_IN, RES_OUT
from .crossjob import CrossJobPeerStrategy

GEMINI_STORE_CADENCE_S = 10800.0  # 3 h. CORRECTED 2026-07-15 from 600 s after
                                # reading the paper (PDF audit): GEMINI anchors
                                # its remote-store tier to real practice --- "In
                                # common practice, existing solutions checkpoint
                                # model states at a low frequency, e.g., every
                                # three hours in BLOOM training [3]" (Sec. 2) ---
                                # and that tier is USER-managed for transfer
                                # learning / debugging, NOT the recovery path.
                                # Our 600 s was invented and made GEMINI 18x more
                                # durable than the paper describes. Modeling it as
                                # a recovery fallback at all remains GENEROUS to
                                # GEMINI. Overridable via `gemini_store_every_s`.
CHECKNRUN_CADENCE_S = 1800.0    # Check-N-Run fixed 30-minute cadence (paper)
CHECKNRUN_WRITE_SCALE = 0.1     # differential+quantized: 10x byte reduction
CHECKNRUN_READ_SCALE = 0.2      # restore reads baseline+incrementals (2x compressed)
MEGASCALE_CADENCE_S = 300.0     # MegaScale fixed async store-flush cadence


# --------------------------------------------------------------------------- #
# per-job frequency rules: (job_spec, cluster_dict) -> checkpoint_every (iters)
# --------------------------------------------------------------------------- #
def iteration_wall_seconds(spec: Mapping, cluster: Mapping) -> float:
    """Wall clock of one training iteration = compute + ring all-reduce, exactly
    as CohortJobRuntime builds it (gradient_gb_per_rank default 4.0, class nic)."""
    it_s = float(spec["iteration_seconds"])
    ranks = int(spec["ranks"])
    nic = float(spec.get("rates", {}).get("nic", cluster["network_bandwidth_gbps"]))
    grad = float(spec.get("gradient_gb_per_rank", 4.0))
    ar = (2.0 * (ranks - 1) / ranks * grad / nic) if ranks > 1 else 0.0
    return it_s + ar


def capture_seconds(spec: Mapping, cluster: Mapping) -> float:
    """GPU->DRAM snapshot time for one per-rank shard (pipeline_stages == 1)."""
    return float(spec["checkpoint_gb_per_rank"]) / float(cluster["gpu_cpu_bandwidth_gbps"])


def checkfreq_every(spec: Mapping, cluster: Mapping, bound: float = 0.035) -> int:
    """CheckFreq (FAST'21) adaptive cadence: pick the interval so the visible
    (blocking snapshot) overhead amortized over the interval stays <= `bound`.
    checkpoint_every = ceil(capture_seconds / (bound * iteration_wall_seconds))."""
    cap = capture_seconds(spec, cluster)
    wall = iteration_wall_seconds(spec, cluster)
    return max(1, math.ceil(cap / (bound * wall)))


def fixed_time_every(seconds: float) -> Callable[[Mapping, Mapping], int]:
    """A fixed wall-clock cadence expressed in iterations for a given job."""
    def rule(spec: Mapping, cluster: Mapping) -> int:
        return max(1, round(seconds / iteration_wall_seconds(spec, cluster)))
    return rule


def _every_one(spec: Mapping, cluster: Mapping) -> int:
    return 1


def _never(spec: Mapping, cluster: Mapping) -> int:
    return 10 ** 9        # effectively no periodic checkpoints (>> any horizon)


# --------------------------------------------------------------------------- #
# jit_pc_fallback: JIT's OWN degraded mode (Gupta et al., "Just-In-Time
# Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures",
# EuroSys'24, Sec. 5.2) — periodic checkpointing where transparent JIT cannot
# reconstruct state, at the paper's optimal frequency (their eq. 3):
#
#     c* = sqrt(N * f / (2 * o))          [checkpoints per second]
#
#   N = job GPU count. Here spec["ranks"], the SIMULATED job width — the width
#       that actually drives this sim's failure process (CohortJobRuntime:
#       fail_rate = ranks * per_node_fail_rate), so N*f is exactly the job-level
#       failure rate the arm experiences.
#   f = per-GPU failure rate per second. The scenario's ACCELERATED
#       failures.per_node_per_second (e.g. mini-C 3.0e-06/s ~ 0.26/node-day) —
#       the same acceleration every arm in the scenario sees, passed in by
#       run_scenario.run_arm and documented in the run output row.
#   o = per-checkpoint stall cost in seconds. We use the sim's PREDICTED visible
#       stall = capture_seconds(spec, cluster): the blocking GPU->DRAM snapshot
#       of one per-rank shard. The store flush itself is async / off the
#       training path, so the capture is the whole first-order stall — the same
#       visible-cost convention checkfreq_every already uses.
# --------------------------------------------------------------------------- #
def jit_pc_cstar(spec: Mapping, cluster: Mapping,
                 per_gpu_fail_rate: float) -> dict:
    """The c* arithmetic for one class (JIT EuroSys'24 Sec. 5.2 eq. 3), spelled
    out so run_arm can document it verbatim in the results row. Requires
    per_gpu_fail_rate > 0 and a positive capture cost."""
    n_gpus = int(spec["ranks"])
    o = capture_seconds(spec, cluster)          # predicted GPU->DRAM stall (s)
    cstar = math.sqrt(n_gpus * per_gpu_fail_rate / (2.0 * o))   # ckpts/second
    interval_s = 1.0 / cstar
    iter_wall = iteration_wall_seconds(spec, cluster)
    every = max(1, round(interval_s / iter_wall))
    return {
        "N_gpus": n_gpus,
        "f_per_gpu_per_s": per_gpu_fail_rate,
        "o_stall_s": round(o, 6),
        "cstar_per_s": round(cstar, 8),
        "interval_s": round(interval_s, 3),
        "iteration_wall_s": round(iter_wall, 4),
        "checkpoint_every": every,
    }


def jit_pc_fallback_rule(per_gpu_fail_rate: float
                         ) -> Callable[[Mapping, Mapping], int]:
    """every_rule for the jit arm with jit_pc_fallback ON (the default).

    Per class: when the class's zero_stage leaves NO replicated state for JIT's
    survivor-drain mechanism (parallelism.jit_dram_recoverable False — the same
    condition that previously routed the driver to scratch), the class instead
    runs PERIODIC checkpointing to the L3 store at the paper's c* cadence
    (jit_pc_cstar above; recovery = newest store copy via the existing path).
    Replicated classes (dp>1, zero_stage 0) and legacy classes (no model:)
    return _never: zero periodic checkpoints, the replica path byte-untouched.
    For replicated classes facing ALL-replica-loss events (whole-job eviction)
    the paper's concrete prescription is an ADDITIONAL 1/day periodic
    checkpoint — which inside these sub-hour horizons never fires, so scratch
    remains there: that is the paper's accepted cost, not a strawman.
    per_gpu_fail_rate <= 0 (no organic failures configured) also yields _never."""
    def rule(spec: Mapping, cluster: Mapping) -> int:
        model = (parallelism.parse_model(spec["model"])
                 if spec.get("model") else None)
        if parallelism.jit_dram_recoverable(model) or per_gpu_fail_rate <= 0:
            return _never(spec, cluster)
        return int(jit_pc_cstar(spec, cluster,
                                per_gpu_fail_rate)["checkpoint_every"])
    return rule


def jit_pc_engaged(spec: Mapping, per_gpu_fail_rate: float) -> bool:
    """True iff jit_pc_fallback actually engages for this class (mirrors
    jit_pc_fallback_rule's gate): a model-configured class with no intact DP
    replica for JIT to drain, under a positive organic failure rate."""
    if per_gpu_fail_rate <= 0 or not spec.get("model"):
        return False
    return not parallelism.jit_dram_recoverable(
        parallelism.parse_model(spec["model"]))


# --------------------------------------------------------------------------- #
# gemini: in-job peer-DRAM replication (SOSP'23)
# --------------------------------------------------------------------------- #
class GeminiStrategy(CrossJobPeerStrategy):
    """GEMINI (SOSP'23) — THREE tiers, all of them:

      1. local host DRAM   the base capture (every iteration);
      2. in-job peer DRAM  the full shard REPLICATED (not striped) to m peers of
                           its OWN job over the NIC — the fast path;
      3. REMOTE PERSISTENT STORE  the durability BACKSTOP: a periodic async
                           upload to STORE_NODE (backstop_upload, cadence
                           GEMINI_STORE_CADENCE_S / `gemini_store_every_s`).

    The in-memory tiers are the fast path; the store is the correctness fallback.
    Modelling only tiers 1-2 would strawman the paper: replicas live in peer host
    DRAM, so they survive a holder's process crash but die with the holder on
    node loss / eviction (drop_local_copies wipes every copy the holder held, and
    whole-job eviction wipes every node) — WITHOUT tier 3 an eviction would always
    mean total loss, which is not what GEMINI does.

    The replication occupies NIC capacity flows (contends in CapacityRegistry)
    but is NOT wired to the all-reduce slowdown counters (network=False) — GEMINI
    interleaves it with training communication for near-zero overhead. The store
    upload rides the same background-QoS path our own f3 backstop uses.

    Recovery (newest-surviving-copy, tie-break toward the faster tier, exactly
    like CrossJobPeerStrategy.recover): own DRAM -> surviving in-job replica ->
    store -> initial_state. m (replication degree) reuses kpeers (paper: 2).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.strategy_name = "gemini"

    def _pick_injob_peers(self, worker, count):
        others = sorted((w for w in self.workers.values() if w.rank != worker.rank),
                        key=lambda w: w.rank)
        if not others or count <= 0:
            return []
        n = min(int(count), len(others))
        return [others[(worker.rank + j) % len(others)] for j in range(n)]

    def _persist_flow(self, worker, job, *, iteration, generation, shard_gb,
                      checkpoint_group):
        m = getattr(worker, "represents", 1)
        total_gb = m * shard_gb
        peers = self._pick_injob_peers(
            worker, self.kpeers_by_job.get(job.job_id, self.kpeers_default))
        if not peers:
            # degenerate (single-worker job): only the own DRAM copy exists.
            return True
        common = {
            "checkpoint_strategy": self.strategy_name,
            "checkpoint_group": checkpoint_group,
            "checkpoint_owner_rank": worker.rank,
            "shard": worker.pipeline_stage,
            "shard_count": job.pipeline_stages,
            "kpeers": len(peers),
            "represents": m,
        }
        # one flow per replica, each carrying the FULL shard (equal weights over
        # len(peers) copies => each stream = total_gb): sender NIC-out shared
        # across the copies, each bounded by the holder's NIC-in. A holder dying
        # mid-transfer aborts the wave (watched includes the holders), so a
        # replica is never recorded on a node that failed while receiving it.
        streams = [[(worker.name, RES_OUT), (p.name, RES_IN)] for p in peers]
        caps = self._node_caps(job.job_id)
        watched = (worker, *peers)
        gens = (generation, *[p.failure_generation for p in peers])
        ok = yield from self._capacity_transfer(
            worker, watched, gens, size_gb=len(peers) * total_gb,
            streams=streams, iteration=iteration,
            operation="checkpoint_dram_to_peer_dram_replica",
            source=f"{worker.name}/dram",
            destination="+".join(f"{p.name}/dram" for p in peers),
            details={**common, "path": "gemini_replica",
                     "replica_holders": [p.name for p in peers]},
            nominal_gbps=self.cluster.network_bandwidth_gbps,
            demand_per_stream=[m * min(caps.nic_out, caps.nic_in)] * len(peers),
            stream_weights=[float(m)] * len(peers),
            network=False)                # NIC capacity occupied, NO AR coupling
        if not ok:
            self.stats["gemini_replica_aborted"] += m
            return False
        for p in peers:
            self._put_copy(owner_rank=worker.rank, location_rank=p.rank,
                           tier="dram", iteration=iteration, size_gb=total_gb)
        self.stats["gemini_replicas"] += m * len(peers)
        return ok

    # -- recovery: own DRAM -> surviving replica -> store -> initial state ------
    def _tiers(self, worker, job):
        """(newest in-memory copy or None, newest store iteration or -1)."""
        copy = self._latest_available_copy(worker)
        store = self.store_copies.get((job.job_id, worker.rank))
        if (
            store is not None
            and store["iteration"] > getattr(worker, "current_iteration", store["iteration"])
        ):
            store = None
        return copy, (store["iteration"] if store is not None else -1)

    def peek_restore(self, worker, job, current_it):
        """Driver accounting hook: which checkpoint recover() will pick."""
        copy, store_it = self._tiers(worker, job)
        local_it = copy.iteration if copy is not None else -1
        if store_it > local_it:            # in-memory tiers win ties (fast path)
            return store_it, "store"
        if copy is None:
            return -1, "initial_state"
        if copy.location_rank == worker.rank:
            return copy.iteration, copy.tier
        return copy.iteration, "gemini_replica"

    def recover(self, worker, job, size_scale: float = 1.0):
        generation = worker.failure_generation
        shard_gb = job.checkpoint_gb / job.pipeline_stages
        copy, store_it = self._tiers(worker, job)
        local_it = copy.iteration if copy is not None else -1
        if store_it > local_it:
            # tier 3: both in-memory tiers gone (node loss / whole-job eviction)
            # or stale -> the remote persistent store is the durability backstop
            ok = yield from self._restore_from_store(
                worker, job, generation=generation, store_it=store_it,
                fetch_gb=shard_gb * size_scale, shard_gb=shard_gb)
            return ok
        if copy is None:
            # nothing anywhere (no store upload has landed yet) -> initial_state
            ok = yield from super().recover(worker, job, size_scale=size_scale)
            return ok
        if copy.location_rank == worker.rank and copy.tier == "dram":
            ok = yield from self._finish_restore(
                worker, generation, shard_gb, copy.iteration, "dram")
            return ok
        # surviving replica: holder DRAM -> own DRAM over the NIC (capacity-fair)
        holder = self.workers[copy.location_rank]
        caps = self._node_caps(job.job_id)
        loaded = yield from self._capacity_transfer(
            worker, (worker,), (generation,), size_gb=shard_gb * size_scale,
            streams=[[(holder.name, RES_OUT), (worker.name, RES_IN)]],
            iteration=copy.iteration,
            operation="checkpoint_replica_dram_to_dram_recovery_chunk",
            source=f"{holder.name}/dram", destination=f"{worker.name}/dram",
            details={"checkpoint_strategy": self.strategy_name,
                     "checkpoint_source_tier": "gemini_replica",
                     "checkpoint_source_rank": copy.location_rank},
            nominal_gbps=min(caps.nic_in, caps.nic_out), category="Recovery")
        if not loaded or worker.failure_generation != generation:
            return False
        ok = yield from self._finish_restore(
            worker, generation, shard_gb, copy.iteration, "gemini_replica")
        return ok


# --------------------------------------------------------------------------- #
# checkfreq: local-SSD two-phase checkpointing at the profiled cadence (FAST'21)
# --------------------------------------------------------------------------- #
class CheckFreqStrategy(CrossJobPeerStrategy):
    """CheckFreq: capture (GPU->DRAM snapshot, the only VISIBLE cost) + async
    persist to the node's OWN SSD. With kpeers=0 every persist takes the parent's
    local-SSD fallback path — no donors, no store. Their advantage is the
    profiled cadence (checkfreq_every): checkpoint_every is solved so the
    amortized snapshot overhead stays <= 3.5% of iteration wall time. Recovery:
    own DRAM/SSD (process crash) — node loss and eviction have no off-node copy
    and honestly fall to initial_state."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.strategy_name = "checkfreq"


# --------------------------------------------------------------------------- #
# checknrun: differential+quantized store checkpoints, 30-min cadence (NSDI'22)
# --------------------------------------------------------------------------- #
class CheckNRunStrategy(CrossJobPeerStrategy):
    """Check-N-Run: store-only checkpointing at a fixed wall-clock cadence
    (paper: 30 min), with THEIR advantage — differential + quantized
    checkpoints. Store writes carry write_scale=0.1x the raw bytes (10x
    reduction, the middle of their reported 6-17x). A restore has to read the
    quantized baseline plus accumulated incrementals: the store fetch is
    charged read_scale=0.2x the raw bytes (2x the compressed size). The
    dequantized full state still crosses DRAM->GPU at full size on restore.
    kpeers=0 and store_mode=True route every persist to the shared store."""

    write_scale = CHECKNRUN_WRITE_SCALE
    read_scale = CHECKNRUN_READ_SCALE

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.strategy_name = "checknrun"

    def _persist_flow(self, worker, job, *, iteration, generation, shard_gb,
                      checkpoint_group):
        # differential+quantized: the bytes that travel and land are 0.1x
        # (cohort dispatch inside the parent scales m * write_scale * shard)
        ok = yield from super()._persist_flow(
            worker, job, iteration=iteration, generation=generation,
            shard_gb=shard_gb * self.write_scale,
            checkpoint_group=checkpoint_group)
        return ok

    def recover(self, worker, job, size_scale: float = 1.0):
        # restore reads baseline + incrementals: 2x the compressed bytes
        # (= 0.2x raw). The parent's local-DRAM path ignores size_scale, and
        # its _finish_restore charges the FULL per-rank state to the GPU.
        ok = yield from super().recover(
            worker, job, size_scale=size_scale * self.read_scale)
        return ok


# --------------------------------------------------------------------------- #
# megascale: production status quo — async full-size store flush @300 s (NSDI'24)
# --------------------------------------------------------------------------- #
class MegaScaleStrategy(CrossJobPeerStrategy):
    """MegaScale-style production practice: every 300 s (fixed, cluster-wide
    convention — not per-job tuned) each node stages GPU->DRAM and an async
    background flush uploads the FULL shard to the shared store; recovery
    fetches from the store. Mechanically this is the parent's store path
    (async_persist + store_mode) pinned to the fixed cadence — the subclass
    exists so traces and metrics identify the arm."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.strategy_name = "megascale"


# --------------------------------------------------------------------------- #
# jit: Just-In-Time checkpointing — no periodic checkpoints (EuroSys'24)
# --------------------------------------------------------------------------- #
class JITStrategy(CrossJobPeerStrategy):
    """Just-In-Time checkpointing: NO periodic checkpoints (every_rule=_never;
    zero training-time overhead — their advantage). On a SINGLE node/process
    failure the surviving DP replicas' state is intact: the driver charges
    restart + one DRAM-snapshot drain over the NIC (jit_drain below) and the
    job loses exactly ONE iteration. Whole-job eviction leaves no survivors:
    recovery honestly falls to initial_state (total loss). The policy lives in
    the driver (run_scenario._single_node_failure); this class only supplies
    the drain transfer and the arm's trace identity."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.strategy_name = "jit"

    def jit_drain(self, worker, job, *, iteration: int):
        """Drain one per-rank DRAM snapshot from a surviving DP replica over
        the NIC (capacity flow survivor:out -> worker:in), then restore it."""
        generation = worker.failure_generation
        shard_gb = job.checkpoint_gb / job.pipeline_stages
        donor = next((w for w in self.workers.values()
                      if w.rank != worker.rank and not w.failed), None)
        if donor is None and getattr(worker, "represents", 1) > 1:
            donor = worker      # survivors live inside this cohort's other nodes
        if donor is None:
            return False        # no surviving replica anywhere
        caps = self._node_caps(job.job_id)
        loaded = yield from self._capacity_transfer(
            worker, (worker,), (generation,), size_gb=shard_gb,
            streams=[[(donor.name, RES_OUT), (worker.name, RES_IN)]],
            iteration=iteration,
            operation="jit_survivor_dram_drain_recovery_chunk",
            source=f"{donor.name}/dram", destination=f"{worker.name}/dram",
            details={"checkpoint_strategy": self.strategy_name,
                     "checkpoint_source_tier": "jit_peer_dram",
                     "checkpoint_source_rank": donor.rank},
            nominal_gbps=min(caps.nic_in, caps.nic_out), category="Recovery")
        if not loaded or worker.failure_generation != generation:
            return False
        ok = yield from self._finish_restore(worker, generation, shard_gb,
                                             iteration, "jit_peer_dram")
        return ok


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Baseline:
    name: str
    strategy_cls: type = CrossJobPeerStrategy
    arm_flags: Mapping = field(default_factory=dict)      # merged into arm_cfg
    class_attrs: Mapping = field(default_factory=dict)    # forced on the strategy
    every_rule: Optional[Callable[[Mapping, Mapping], int]] = None
    driver: Optional[str] = None                          # special driver mode
    # periodic store-backstop cadence in ITERATIONS (drives CohortJobRuntime
    # .store_every -> backstop_upload), for baselines whose paper has a remote
    # persistent store BESIDE its fast path (gemini). None = no store backstop.
    store_every_rule: Optional[Callable[[Mapping, Mapping], int]] = None


BASELINES: dict[str, Baseline] = {
    "gemini": Baseline(
        name="gemini",
        strategy_cls=GeminiStrategy,
        # in-job replication + the paper's remote persistent store as the
        # durability backstop; no cross-job donors, no controller slots.
        # store_mode stays FALSE: that flag routes EVERY persist to the store
        # (the checknrun/megascale arms), whereas GEMINI's store is a periodic
        # backstop BESIDE the in-memory fast path (store_every_rule below).
        arm_flags={"kpeers": True, "store_mode": False, "slots": False,
                   "random_phases": False, "replication_degree": 2},
        every_rule=_every_one,          # replicate every iteration
        store_every_rule=fixed_time_every(GEMINI_STORE_CADENCE_S),
    ),
    "checkfreq": Baseline(
        name="checkfreq",
        strategy_cls=CheckFreqStrategy,
        # local-SSD-only: kpeers=False -> k=0 -> the base persist takes the
        # local-SSD fallback path (no donors, no store). 2-phase async: the
        # visible cost is the GPU->DRAM snapshot; the SSD write is background.
        arm_flags={"kpeers": False, "store_mode": False, "slots": False,
                   "random_phases": False},
        every_rule=checkfreq_every,     # profiled cadence, overhead <= 3.5%
    ),
    "checknrun": Baseline(
        name="checknrun",
        strategy_cls=CheckNRunStrategy,
        # store-only: no donors; store_mode routes persists to the store node
        arm_flags={"kpeers": False, "store_mode": True, "slots": False,
                   "random_phases": False},
        every_rule=fixed_time_every(CHECKNRUN_CADENCE_S),   # fixed 30 min
    ),
    "megascale": Baseline(
        name="megascale",
        strategy_cls=MegaScaleStrategy,
        # full-size async store flushes; no donors, no controller coordination
        arm_flags={"kpeers": False, "store_mode": True, "slots": False,
                   "random_phases": False},
        every_rule=fixed_time_every(MEGASCALE_CADENCE_S),   # fixed 300 s
    ),
    "jit": Baseline(
        name="jit",
        strategy_cls=JITStrategy,
        arm_flags={"kpeers": False, "store_mode": False, "slots": False,
                   "random_phases": False},
        # every_rule here is the ARCHIVED default (no periodic checkpoints).
        # run_scenario.run_arm swaps in jit_pc_fallback_rule(f) unless the arm
        # sets `jit_pc_fallback: false` (fairness fix 2026-07-28): the paper's
        # own degraded mode is periodic checkpointing at c* (Sec. 5.2 eq. 3)
        # for state JIT cannot reconstruct, NOT from-scratch recovery.
        every_rule=_never,
        driver="jit",                   # single failures: survivor drain, -1 iter
    ),
}

# strategy names that appear in trace details (run_arm's path accounting)
STRATEGY_NAMES = frozenset({"crossjob_peer", *BASELINES})


def resolve(arm_cfg: Mapping) -> Optional[Baseline]:
    """Return the Baseline for an arm whose config carries `baseline: <name>`.
    Fixed-cadence baselines accept a `cadence_s` arm override (hand-check
    scenarios shrink the 30-min/300-s paper cadences to fit a small horizon);
    gemini accepts `gemini_store_every_s` for its remote-persistent-store
    backstop cadence (default GEMINI_STORE_CADENCE_S)."""
    name = arm_cfg.get("baseline") if isinstance(arm_cfg, Mapping) else None
    if not name:
        return None
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    bl = BASELINES[name]
    cadence = arm_cfg.get("cadence_s")
    if cadence and name in ("checknrun", "megascale"):
        bl = dataclasses.replace(bl, every_rule=fixed_time_every(float(cadence)))
    store_cadence = arm_cfg.get("gemini_store_every_s")
    if store_cadence and name == "gemini":
        bl = dataclasses.replace(
            bl, store_every_rule=fixed_time_every(float(store_cadence)))
    return bl
