"""Cost-model bridge: scenario YAML -> GP solve -> controller policy JSON.

This is the Level-1 controller's OPEN-LOOP half, run before the simulation:
the finalized cost model (checkpointing/multilevel/joint_async_optimizer.py,
unified_cost_model.md #2/#8) decides per job
  - FREQUENCY  f2 (durable peer-tier checkpoint rate)  -> checkpoint_every,
  - DONOR BUDGET k  (capacity-feasibility rule, #8.1's per-wave constraint),
  - PHASE slot (greedy longest-first packing by predicted flush duration,
    the operational form of #8.1's stagger objective; the in-sim closed loop
    then re-packs from MEASURED durations every epoch).
Nothing in the emitted policy is hand-set; edit the scenario YAML, re-solve.

Mapping (documented so the solve is auditable, see --explain):
  s_gb   = ranks x per-rank shard (job-aggregate, bandwidth-coupled objective)
  nu     = (lambda_process, lambda_node + lambda_spot, ~0) per job
           [process death -> DRAM survives; node/spot -> off-node copy needed;
            correlated multi-node loss ~0 at these horizons]
  A      = process restart; R = (process, node, spot) restart + fetch estimates
  B      = (sum capture bw, sum donor disk write bw, store aggregate)
  tau    = 1.0 -> k_steps are intervals in SECONDS; per job
           checkpoint_every = max(1, round(interval_s / iteration_seconds)).
  f1>=f2 hierarchy: the sim has ONE knob (capture+persist per checkpoint()),
           so the durable tier's f2 drives it; f1 headroom is unused.

Usage:
  python3 gp_policy.py --scenario scenarios/realwidth73.yaml \
      --out scenarios/realwidth73_gp_policy.json [--explain]
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import inspect
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from checkpointing import parallelism    # parallelism-aware shard calculator (D2)

_REPO_PARENT = Path(__file__).resolve().parents[1]
_MODEL_ROOTS = (
    _REPO_PARENT / "checkpointing",
    _REPO_PARENT / "checkpointing_repo",
)
_model_root = next(
    (root for root in _MODEL_ROOTS if (root / "multilevel").is_dir()),
    None,
)
if _model_root is None:
    searched = ", ".join(str(root) for root in _MODEL_ROOTS)
    raise ModuleNotFoundError(
        f"checkpoint cost-model package 'multilevel' was not found; searched {searched}"
    )
sys.path.insert(0, str(_model_root))
from multilevel.joint_async_optimizer import (  # noqa: E402
    AsyncModelSpec, evaluate_async_policy, optimize_joint_async_gp)
try:  # v2 (+ priced placement) live only in newer cost-model revisions.
    from multilevel.joint_async_optimizer import (  # noqa: E402
        PLACEMENTS, JobPhysical, V2Params, evaluate_async_policy_v2,
        optimize_joint_async_gp_v2, optimize_placement_v2)
except ImportError:  # keep the default v1 solver usable with older checkouts.
    JobPhysical = V2Params = None
    evaluate_async_policy_v2 = optimize_joint_async_gp_v2 = None
    PLACEMENTS = None
    optimize_placement_v2 = None
try:  # L1.5 peerdram option lives only in newer cost-model revisions.
    from multilevel.joint_async_optimizer import PLACEMENTS_L15  # noqa: E402
except ImportError:
    PLACEMENTS_L15 = None
try:  # rack failure + injob_ssd (rack_failure_spec.md) — newer revisions only.
    from multilevel.joint_async_optimizer import (  # noqa: E402
        PLACEMENTS_ALL, PLACEMENTS_RACK)
except ImportError:
    PLACEMENTS_RACK = PLACEMENTS_ALL = None
from multilevel.joint_optimizer import JobSpec  # noqa: E402

DEFAULT_WEIGHTS = {"mega": 5.0, "fxl": 5.0, "frontier": 3.0,
                   "standard_g": 2.0, "standard_c": 2.0, "be": 1.0}
HOSTED_CAP = 2          # shards a node hosts concurrently (crossjob.py)
# priced placement (2026-07-21): in-job DRAM replication degree, matching
# run_scenario.INJOB_DEGREE (a replica on ONE other node of the job survives the
# common single-node loss). local -> k=0 (own SSD + L3); crossjob -> solved k.
INJOB_DEGREE = 2
WAVE_UTIL = 0.9         # per-wave donor-capacity utilization target (#8.1)
# crossjob-first placement mode (Sam's placement ruling, 2026-07-27): the
# minimum stripe width for crossjob to count as FEASIBLE. A width-1 "stripe"
# is a single unprotected remote copy: no XOR-parity / k-1 stitch tolerance to
# donor loss, strictly less redundancy than injob's INJOB_DEGREE=2 replica,
# while still paying donor duty — a donor pool that can only grant k=1 under
# the per-wave capacity rule (#8.1) is DONOR-SCARCE and crossjob falls back.
CF_MIN_STRIPE = 2
# epsilon-margin refinement of crossjob-first (Sam's rule formalized,
# 2026-07-27): with --cf-epsilon, a FEASIBLE class chooses crossjob UNLESS an
# alternative option's priced objective beats crossjob's by MORE than epsilon
# (RELATIVE), evaluated at the placement descent's current assignment (exit
# gate — once a class has cleared the margin it shops by pure price; see
# optimize_placement_v2's prefer_by_class). The hard feasibility gate stays on
# top unchanged: INFEASIBLE classes go to the fallback menu regardless of any
# margin. Bare --cf-epsilon means this default; omitting the flag keeps the
# plain (pin-when-feasible) crossjob-first rule byte-identical.
CF_EPSILON_DEFAULT = 0.01
# v2 (cost_model_v2_design.md). gamma is PROVISIONAL: the simulator does not yet
# charge donor-side hosting slowdown (blocked on the engine merge), so this
# hosting-tax coefficient is uncalibrated. --explain marks it loudly. Default 0
# keeps the v2 head-to-head resting on the CALIBRATED terms (tail + duty + store).
PROVISIONAL_GAMMA = 0.0
# correlated / rack-scale failure mass assigned to tier 3 (design §4): per node
# per second. Placeholder until the trace dossier lands. Makes f3 non-degenerate
# so the store budget is ALLOCATED rather than floor-defaulted.
NU3_CORRELATED_PER_NODE = 1.0e-9
# L1.5 DRAM price (l15_peer_dram_spec.md; Mark's memory-cost column). Units:
# fraction of ONE NODE-second per hosted GB-second. Derivation (documented,
# defensible; every input is a knob via model.v2.gamma_dram):
#   server DDR5 ~ $4/GB (2026 street), amortized over 4 years
#     -> 4 / (4 * 3.156e7 s) ~= 3.2e-8 $/GB-s of resident DRAM;
#   H100-class GPU node ~ $10/h rental-equivalent -> 2.78e-3 $/node-s;
#   gamma_dram = 3.2e-8 / 2.78e-3 ~= 1.2e-5 node-s per GB-s.
# The experiment reports shopping at gamma_dram = 0 AND at this value.
GAMMA_DRAM_DOC = 1.2e-5


@dataclass
class ModelInputs:
    """Everything the cost model needs, derived from a scenario dict.

    Built by :func:`build_model_inputs` so the scenario -> cost-model mapping has
    exactly ONE definition, shared by two consumers:
      - :func:`solve` (below), which OPTIMIZES over these inputs, and
      - checkpointing/multilevel/run_model_validation.py, which EVALUATES other
        arms' enacted policies against the same model (experiment X8).
    `jobs[j]` and `meta[j]` are parallel: same job, same index.
    """
    sc: dict
    jobs: list
    meta: list
    async_model: AsyncModelSpec
    B: tuple                 # (B1, B2, B3) tier budgets
    f_max: list
    r_max: list
    wave_util: float
    hosted_cap: int


def build_model_inputs(sc: dict, dp_dedup: bool = False) -> ModelInputs:
    """Scenario dict -> cost-model inputs (see :class:`ModelInputs`).

    D2 (2026-07-23): classes carrying a `model:` parallelism block get their
    per-rank shard from the calculator (checkpointing/parallelism.py), not from a
    hand-set number. `dp_dedup=True` prices the DP-aware effective shard
    (unique + replicated/dp) — the shard_gb the coupled solve optimizes against,
    so reduced bytes shorten cadences. Classes WITHOUT `model:` keep their literal
    `checkpoint_gb_per_rank` and `dp_dedup` is a no-op for them (legacy exact)."""
    parallelism.apply_model_shards(sc)      # inject no-dedup shard for model: classes
    cl = sc["cluster"]
    f = sc["failures"]
    # every cost-model input is a scenario knob (model: block, all optional)
    mdl = sc.get("model", {})
    weights_map = {**DEFAULT_WEIGHTS, **(mdl.get("weights") or {})}
    price_preemption = bool(mdl.get("price_preemption_hazard", True))
    wave_util = float(mdl.get("wave_util", WAVE_UTIL))
    hosted_cap = int(mdl.get("hosted_cap", HOSTED_CAP))
    spec_kwargs = {k: tuple(mdl[y]) for k, y in
                   (("fg_fraction", "fg_fraction"),
                    ("overlap_alpha", "overlap_alpha"),
                    ("delay_scale", "delay_scale")) if y in mdl}
    async_model = AsyncModelSpec(**spec_kwargs)
    p_node = float(f["per_node_per_second"])
    w = f["weights"]
    rs = f["restart_seconds"]
    # rack failure slice (rack_failure_spec.md + post-spec decisions 2026-07-27).
    # Scenario knobs (exact names, riding the failures: block like the other
    # failure physics): rack_size (default 4), rack_weight (ABSENT/0 => the rack
    # slice is OFF and everything below reduces to the committed numbers,
    # bit-for-bit — the regression gate), rack_p_destroy (default 0.01),
    # rack_outage_s (default [3600, 21600] uniform). rack_id = node_index //
    # rack_size in the sim; here only the hazard geometry matters.
    rack_weight = float(f.get("rack_weight", 0.0) or 0.0)
    rack_size = max(int(f.get("rack_size", 4)), 1)
    rack_outage = f.get("rack_outage_s", [3600.0, 21600.0])
    rack_outage_mean = (float(sum(rack_outage)) / len(rack_outage)
                        if isinstance(rack_outage, (list, tuple))
                        else float(rack_outage))
    # per-RACK event rate: cluster-wide rack-event rate = rack_weight x the
    # cluster's node-level ORGANIC event mass (sum over nodes of their organic
    # lambda = node_count * p_node; the preemption hazard is NOT organic),
    # spread uniformly over node_count/rack_size racks. Independent of
    # rack_size at cluster level (bigger racks = bigger blast, same count).
    mu_rack = rack_weight * p_node * rack_size
    backstop_repair_s = float((sc.get("store") or {}).get("backstop_s", 600.0))

    jobs, meta = [], []
    for cname, spec in sc["classes"].items():
        ranks = int(spec["ranks"])
        # per-rank shard: the DP-aware effective shard when dp_dedup is on and the
        # class is parallelism-configured (unique + replicated/dp); otherwise the
        # no-dedup shard already resolved into checkpoint_gb_per_rank (calculator
        # for model: classes, literal for legacy classes).
        if dp_dedup and "model" in spec:
            shard = parallelism.effective_shard_gb(
                parallelism.parse_model(spec["model"]), dedup=True)
        else:
            shard = float(spec["checkpoint_gb_per_rank"])
        rates = spec.get("rates", {})
        nic = float(rates.get("nic", cl["network_bandwidth_gbps"]))
        disk = float(rates.get("disk", cl["local_ssd_bandwidth_gbps"]))
        # L1.5 duplex NIC (per-direction overrides; default = symmetric `nic`)
        nic_in = float(rates.get("nic_in", nic))
        nic_out = float(rates.get("nic_out", nic))
        lam_total = ranks * p_node
        # preemption hazard: the scenario's whole-job spot eviction targets one
        # healthy job of the victim class every `every_s` — a per-job hazard of
        # 1/(every_s * class_count), usually dominating organic failures for
        # that class (be: ~3e-4/s vs organic 6.4e-6/s). The corrected-engine GP
        # arm lost 5x more work than hand-set cadences until this was priced.
        pre = sc.get("preemption") or {}
        if price_preemption and \
                cname.startswith(str(pre.get("class_prefix", "\x00"))):
            lam_total += 1.0 / (float(pre["every_s"]) * int(spec["count"]))
        gpu_cpu = float(cl["gpu_cpu_bandwidth_gbps"])
        # tier mapping v3: the sim now ENACTS f1 (DRAM snapshots) and f3
        # (store backstop), so failure mass splits by what each class needs:
        # process death -> host DRAM survives, freshness = f1 (nu1);
        # node/spot -> off-node copy, freshness = f2 (nu2). A=0 because each
        # class's full restart time is inside its restore term (fix #22).
        # (v2 had lumped everything on tier 2 — correct only while f1 was
        # unenacted; the enactment removes the phantom-tier objection.)
        # reboot (2026-07-21, reboot_failure_type_spec.md): optional 4th organic
        # class. Its restart time (source-backed ~180 s) folds into the off-node
        # tier-2 restore average so the coupled solve sees its true cost. Absent
        # => w_reboot 0, so v1 and every pre-reboot scenario stay byte-identical.
        w_reboot = float(w.get("reboot", 0.0))
        rs_reboot = float(rs.get("reboot", rs["node"]))
        r1 = float(rs["process"]) + shard / gpu_cpu
        r_mix = (float(w["node"]) * (float(rs["node"]) + shard / nic
                                     + shard / gpu_cpu)
                 + float(w["spot"]) * (float(rs["spot"]) + shard / nic
                                       + shard / gpu_cpu)
                 + w_reboot * (rs_reboot + shard / nic
                               + shard / gpu_cpu)) /                 max(float(w["node"]) + float(w["spot"]) + w_reboot, 1e-9)
        # AUDIT FIX (2026-07-25): per-placement restore terms. `local`'s tier-2
        # mass is REBOOT only (its f2 copy is the reboot-surviving own SSD), so
        # its tier-2 restore prices the reboot restart + an OWN-SSD read at
        # disk rate (not the node/spot NIC mix — was ~1.3% undercharged); its
        # tier-3 mass (node/spot/evict) prices the node/spot restart mix.
        # crossjob/injob/peerdram keep the historical (r1, r_mix, r_mix).
        r_reboot_ssd = rs_reboot + shard / max(disk, 1e-9) + shard / gpu_cpu
        r_mix_nse = ((float(w["node"]) * (float(rs["node"]) + shard / nic
                                          + shard / gpu_cpu)
                      + float(w["spot"]) * (float(rs["spot"]) + shard / nic
                                            + shard / gpu_cpu))
                     / max(float(w["node"]) + float(w["spot"]), 1e-9))
        R_default = (r1, r_mix, r_mix)
        R_by_placement = {
            "crossjob": R_default,
            "injob": R_default,
            # injob_ssd (rack spec): tier-2 mass is the same node/spot/reboot
            # mix as injob and the replica travels the same in-job NIC, so the
            # committed r_mix shape applies (the donor-side SSD read is left
            # unpriced exactly as it is for crossjob donor reads — one
            # convention across placements).
            "injob_ssd": R_default,
            "peerdram": R_default,
            "local": (r1, r_reboot_ssd, r_mix_nse),
        }
        # audit fix #15: the simulator's iteration WALL time includes the ring
        # all-reduce (mirrors run_scenario.CohortJobRuntime/build_config)
        grad = float(spec.get("weights_gb_per_rank", 4.0))
        ar = 2.0 * (ranks - 1) / max(ranks, 1) * grad / nic
        iter_wall = float(spec["iteration_seconds"]) + ar
        # priced-placement failure decomposition (2026-07-21): split the off-node
        # mass into (a) single-node/spot organic loss lam_ns and (b) whole-job
        # eviction hazard lam_evict. crossjob nu2 = lam_ns + lam_evict is exactly
        # today's lam_off (weights sum to 1), so the crossjob mapping is unchanged;
        # injob/local remap differently (eviction kills in-job/local copies).
        lam_ns = ranks * p_node * (float(w["node"]) + float(w["spot"]))
        # reboot organic mass: its own tier bucket in v2's per-placement nu
        # (crossjob/injob -> peer age f2; local -> own SSD age f2). v1 folds it
        # into lam_off (tier 2) via the weights summing to 1 — see _nu_by_placement.
        lam_reboot = ranks * p_node * float(w.get("reboot", 0.0))
        lam_evict = (lam_total - ranks * p_node) if (
            price_preemption and cname.startswith(
                str(pre.get("class_prefix", "\x00")))) else 0.0
        # L1.5 piece-availability factor a = (1 - p_host_event_in_demotion_
        # window)^k (l15_peer_dram_spec.md): a dram-tier piece dies if ITS host
        # suffers a reboot/node/spot event before the piece demotes to SSD
        # (window T = (shard/k)/disk at background disk rate; host whole-job
        # eviction hazard omitted — donors are overwhelmingly non-victim
        # classes at these mixes). a ~= 1 for every planned sweep; computed,
        # not assumed, so donorloss-grade regimes price it honestly.
        k_req_cls = max(int(spec.get("kpeers", 2)) or 1, 1)
        host_event_rate = p_node * (float(w.get("reboot", 0.0))
                                    + float(w["node"]) + float(w["spot"]))
        t_demote = (shard / k_req_cls) / max(disk, 1e-9)
        p_win = min(host_event_rate * t_demote, 1.0)
        dram_avail = (1.0 - p_win) ** k_req_cls
        # ---- rack slice geometry (rack_weight=0 => 0.0 / 1.0, inert) --------
        # OWNER-rack hazard: contiguous packing spans ceil(ranks/rack_size)
        # racks, each failing at mu_rack -> lam_rack = mu_rack * racks_spanned
        # (~ rack_weight * job organic node mass, rounded up to rack grain).
        lam_rack = mu_rack * math.ceil(ranks / rack_size)
        # DONOR-side exposure -> availability discount on the L2 term for
        # rack-anti-affine stripes (crossjob/peerdram), FIRST-ORDER:
        #   a piece's host rack goes dark at rate mu_rack; under the no-return
        #   rule it stays dark until the NEXT peer wave re-places it on a live
        #   donor, i.e. one refresh interval 1/f2 <= backstop_s (the durable-
        #   peer maintenance floor), or until the outage return timer if that
        #   is somehow sooner -> T_dark = min(backstop_s, E[outage]);
        #   p_dark = mu_rack * T_dark per piece, independent across pieces
        #   (anti-affinity: k distinct racks, none the owner's).
        #   The stripe is recoverable iff >= k-1 data pieces are available
        #   (the XOR-parity / k-1 stitch recoverability the sim already grants
        #   donor loss) -> a_rack = P(#dark <= 1) =
        #   (1-p)^k + k*p*(1-p)^(k-1),  evaluated at the class's requested k
        #   (a GP cannot carry a k-variable exponent; same convention as
        #   dram_avail above). This one factor prices BOTH sides at once:
        #   every stripe's owner sees the aggregate darkness caused by OTHER
        #   jobs' racks failing (donor-side exposure), symmetrically.
        if rack_weight > 0.0:
            t_dark = min(backstop_repair_s, rack_outage_mean)
            p_dark = min(mu_rack * t_dark, 1.0)
            rack_l2_avail = ((1.0 - p_dark) ** k_req_cls
                             + k_req_cls * p_dark
                             * (1.0 - p_dark) ** (k_req_cls - 1))
        else:
            rack_l2_avail = 1.0
        for i in range(int(spec["count"])):
            lam_proc = ranks * p_node * float(w["process"])
            lam_off = lam_total - ranks * p_node * float(w["process"])
            jobs.append(JobSpec(
                name=f"{cname}{i}", s_gb=ranks * shard,
                nu=(lam_proc, lam_off, 1e-12),
                weight=float(spec.get("weight", weights_map.get(cname, 1.0))),
                A=0.0, R=(r1, r_mix, r_mix)))
            meta.append({"job_id": f"{cname}{i}", "class": cname, "ranks": ranks,
                         "shard_gb": shard, "nic": nic, "disk": disk,
                         "nic_in": nic_in, "nic_out": nic_out,
                         "dram_avail": dram_avail,
                         "gpu_cpu": gpu_cpu,
                         "iteration_seconds": float(spec["iteration_seconds"]),
                         "iter_wall": iter_wall,
                         "lam_proc": lam_proc, "lam_ns": lam_ns,
                         "lam_reboot": lam_reboot,
                         "lam_evict": lam_evict,
                         "lam_rack": lam_rack,
                         "rack_l2_avail": rack_l2_avail,
                         "R_by_placement": R_by_placement})

    B1 = sum(m["ranks"] for m in meta) * float(cl["gpu_cpu_bandwidth_gbps"])
    B2 = sum(m["ranks"] * m["disk"] for m in meta)          # donor disk aggregate
    B3 = float(sc.get("store", {}).get("in_gbps", 500.0))
    gpu_cpu = float(cl["gpu_cpu_bandwidth_gbps"])
    # audit fix #18/#19: per-job caps — a job cannot checkpoint more often than
    # it iterates, and cannot exceed its own capture bw / NIC egress / the donor
    # disk pool EXCLUDING its own nodes / the store ingest
    f_max = [1.0 / m["iter_wall"] for m in meta]
    r_max = [[m["ranks"] * gpu_cpu,
              min(m["ranks"] * m["nic"],
                  sum(o["ranks"] * o["disk"] for o in meta
                      if o["job_id"] != m["job_id"])),
              B3]
             for m in meta]
    return ModelInputs(sc=sc, jobs=jobs, meta=meta, async_model=async_model,
                       B=(B1, B2, B3), f_max=f_max, r_max=r_max,
                       wave_util=wave_util, hosted_cap=hosted_cap)


def _crossjob_first_feasibility(sc: dict, meta: list, total_ranks: int,
                                hosted_cap: int, wave_util: float) -> dict:
    """Per-class crossjob FEASIBILITY for the crossjob-first placement mode
    (Sam's placement ruling, 2026-07-27). Crossjob is feasible for a class iff

      (a) DUTY-CAPACITY HEADROOM: the per-wave donor-capacity rule (#8.1 —
          the same k_wave = int(wave_util * hosted_cap * other_ranks / ranks)
          bound that caps enactment and the v2 solve, see _v2_k_cap) grants a
          GENUINE stripe: k_wave >= CF_MIN_STRIPE. A pool that can only grant
          width 1 (donor scarcity, e.g. miniC_tight's mega at 60% of cluster
          ranks) fails: a single remote piece has no parity/stitch tolerance
          to donor loss and less redundancy than injob's INJOB_DEGREE=2.

      (b) RACK ANTI-AFFINITY: k distinct NON-OWNER racks of donors exist for
          the stripe that would be enacted (k = min(k_req, k_wave)). Donors
          are other jobs' ranks; under contiguous packing they occupy
          (total_ranks - ranks) nodes -> donor_racks = that // rack_size (a
          conservative floor: boundary racks shared with the owner are not
          counted). Vacuous (always true) when the scenario does not price
          rack events (failures.rack_weight absent/0) — the sim then runs no
          rack-anti-affine matcher at all.

    Returns {class: {k_req, k_wave_cap, k_enact, duty_ok, donor_racks,
    rack_ok, feasible}} in scenario declaration order."""
    f = sc.get("failures") or {}
    rack_on = float(f.get("rack_weight", 0.0) or 0.0) > 0.0
    rack_size = max(int(f.get("rack_size", 4)), 1)
    feas: dict = {}
    for m in meta:
        c = m["class"]
        if c in feas:
            continue
        spec = sc["classes"][c]
        k_req = max(int(spec.get("kpeers", 2)) or 1, 1)
        others = hosted_cap * (total_ranks - m["ranks"])
        k_wave = int(wave_util * others / m["ranks"])
        k_enact = max(0, min(k_req, k_wave))
        duty_ok = k_wave >= CF_MIN_STRIPE
        if rack_on:
            donor_racks = (total_ranks - m["ranks"]) // rack_size
            rack_ok = donor_racks >= max(k_enact, 1)
        else:
            donor_racks = None
            rack_ok = True
        feas[c] = {"k_req": k_req, "k_wave_cap": k_wave, "k_enact": k_enact,
                   "duty_ok": duty_ok, "donor_racks": donor_racks,
                   "rack_ok": rack_ok, "feasible": bool(duty_ok and rack_ok)}
    return feas


def _print_crossjob_first_sidecar(feas: dict, placement_by_class: dict,
                                  shopping: dict, fallback_menu: tuple,
                                  cf_epsilon: float | None = None) -> None:
    """The crossjob-first sidecar: per-class feasibility verdict, chosen
    placement, and — when the fallback fired — the priced fallback comparison.
    Printed on EVERY crossjob-first solve (captured into the shopping .txt
    sidecars next to the policy JSONs). With the epsilon refinement
    (cf_epsilon not None) FEASIBLE classes also get their margin comparison:
    the final shopping row plus each alternative's relative margin vs
    crossjob (positive = the alternative beats crossjob by that fraction)."""
    if cf_epsilon is None:
        print(f"[crossjob-first] rule: crossjob wherever FEASIBLE (duty: "
              f"per-wave donor cap k_wave >= {CF_MIN_STRIPE}; rack: k "
              f"distinct non-owner donor racks); infeasible -> cheapest of "
              f"{list(fallback_menu)} by the priced objective. Crossjob is "
              f"never price-compared to the fallback menu when feasible.")
    else:
        print(f"[crossjob-first] rule: FEASIBLE (duty: per-wave donor cap "
              f"k_wave >= {CF_MIN_STRIPE}; rack: k distinct non-owner donor "
              f"racks) classes choose crossjob UNLESS an alternative beats "
              f"crossjob's priced objective by more than epsilon="
              f"{cf_epsilon:g} RELATIVE at the descent's current assignment "
              f"(exit gate); infeasible -> cheapest of {list(fallback_menu)} "
              f"by the priced objective regardless of margin.")
    for c, fe in feas.items():
        verdict = "FEASIBLE " if fe["feasible"] else "INFEASIBLE"
        rack_part = (f"donor_racks={fe['donor_racks']} rack_ok={fe['rack_ok']}"
                     if fe["donor_racks"] is not None
                     else "rack_ok=vacuous (no rack pricing)")
        print(f"  {c:12s} {verdict} | duty: k_req={fe['k_req']} "
              f"k_wave_cap={fe['k_wave_cap']} k_enact={fe['k_enact']} "
              f"duty_ok={fe['duty_ok']} | {rack_part} "
              f"-> chosen={placement_by_class.get(c)}")
        if not fe["feasible"]:
            row = shopping.get(c) or {}
            cmp_s = "  ".join(
                f"{'*' if p == placement_by_class.get(c) else ' '}{p}={v:.6f}"
                for p, v in row.items())
            print(f"  {'':12s} fallback comparison: {cmp_s}")
        elif cf_epsilon is not None:
            row = shopping.get(c) or {}
            xj = row.get("crossjob")
            parts = []
            for p, v in row.items():
                star = "*" if p == placement_by_class.get(c) else " "
                if xj and p != "crossjob":
                    parts.append(f"{star}{p}={v:.6f} "
                                 f"(margin {100.0 * (xj - v) / xj:+.4f}%)")
                else:
                    parts.append(f"{star}{p}={v:.6f}")
            print(f"  {'':12s} epsilon comparison (final assignment): "
                  + "  ".join(parts))


def build_v2_params(sc: dict) -> V2Params:
    """cost_model_v2_design.md tunables from the scenario `model.v2` block (all
    optional). Calibrated alpha/beta default to the mini-C k-sweep fit.
    L1.5: gamma_dram / dram_host_slots ride the same block (dram_host_slots
    falls back to the sim-shared model.dram_host_slots knob)."""
    v2 = dict((sc.get("model") or {}).get("v2") or {})
    known = ("alpha", "beta", "eta", "hosted_cap", "gamma", "f3_floor",
             "store_binding", "u_cap", "fixed_point_iters", "fixed_point_tol",
             "link_wave_speed", "drain_bytes_rate", "peer_maintain_floor",
             "gamma_dram", "dram_host_slots", "price_dram_rent")
    kw = {k: v2[k] for k in known if k in v2}
    kw.setdefault("gamma", PROVISIONAL_GAMMA)
    kw.setdefault("hosted_cap", int((sc.get("model") or {}).get(
        "hosted_cap", HOSTED_CAP)))
    kw.setdefault("dram_host_slots", int((sc.get("model") or {}).get(
        "dram_host_slots", 1)))
    # AUDIT FIX (2026-07-25): ONE consistent store-cadence floor. The operator
    # backstop request (store.backstop_s) used to be applied at ENACTMENT as an
    # up-clamp OVER the solved f3 — the solve honored B3 while the enacted
    # cadence demanded up to 3x B3 (miniC_tight: 15.2 GB/s vs 5). The request
    # now enters the SOLVE as the f3 floor, capped at the budget-feasible
    # uniform cadence 0.9*B3/total_state (so the floor can never make the B3
    # constraint infeasible); the enactment then honors the solved f3
    # unclamped. An explicit model.v2.f3_floor still wins.
    if "f3_floor" not in kw:
        backstop_s = float((sc.get("store") or {}).get("backstop_s", 600.0))
        b3 = float((sc.get("store") or {}).get("in_gbps", 500.0))
        total_state = sum(
            int(s2["count"]) * int(s2["ranks"])
            * float(s2["checkpoint_gb_per_rank"])
            for s2 in sc["classes"].values()) or 1.0
        kw["f3_floor"] = min(1.0 / max(backstop_s, 1e-9),
                             0.9 * b3 / total_state)
    # older cost-model checkouts have no gamma_dram/dram_host_slots fields
    import dataclasses as _dc
    fields = {f.name for f in _dc.fields(V2Params)}
    kw = {k: v for k, v in kw.items() if k in fields}
    return V2Params(**kw)


def _v2_k_cap(meta: list, hosted_cap: int, j: int,
              wave_util: float = WAVE_UTIL) -> int:
    """The v1 post-hoc donor-feasibility cap, kept in v2 as an INTEGER upper
    bound on the solved k and a sanity assertion that v2 never over-stripes.
    AUDIT FIX (2026-07-25): takes the SCENARIO wave_util (model.wave_util) so
    the solver bound and the enactment assertion share one source — a non-
    default wave_util previously crashed the assertion (solver capped at the
    module constant 0.9 while the assertion capped at the scenario value)."""
    total_ranks = sum(m["ranks"] for m in meta)
    m = meta[j]
    others = hosted_cap * (total_ranks - m["ranks"])
    return max(1, int(wave_util * others / m["ranks"]))


def solve(scenario: Path, backend: str = "cvxpy",
          model_version: str = "v1",
          measured_drain_gbps: float | None = None,
          placement: str = "solve",
          solve_mode: str = "joint",
          dp_dedup: bool = False,
          peerdram: bool = False,
          gamma_dram: float | None = None,
          placement_mode: str = "price-all",
          cf_epsilon: float | None = None,
          price_dram_rent: bool = False) -> dict:
    """backend: "cvxpy" (default, disciplined GP via CLARABEL — scales to the
    385-job scenarios) or "scipy" (the original SLSQP solve). Both solve the
    same geometric program; see checkpointing/multilevel/test_solver_equivalence.py.

    model_version: "v1" (default, UNCHANGED behavior) or "v2" (the globally
    coupled solve of cost_model_v2_design.md — solver-emitted k, donor-duty
    budget, tail-stretch flush, binding store, provisional hosting tax).

    measured_drain_gbps: a sim-measured donor-drain bytes-rate (GB/s) plumbed
    into the v2 donor hosting tax input (design §2); overrides model.v2's
    drain_bytes_rate. Provisional (the tax stays OFF unless gamma>0).

    placement (v2 ONLY; ignored for v1 which is always crossjob): "solve"
    (default) prices per-class placement in {crossjob,injob,local} by coordinate
    descent — the menu grows to {crossjob,injob,injob_ssd,local} whenever the
    scenario prices rack events (failures.rack_weight > 0; rack_failure_spec.md)
    and gains peerdram with the peerdram flag; a fixed placement name forces ALL
    classes to it (the all-crossjob A/B comparator + injob/injob_ssd/local
    ablations).

    solve_mode: "joint" (default, UNCHANGED — the cluster-wide coupled controller)
    or "perjob" (the SELFISH per-job baseline: each class solved ALONE against the
    real cluster geometry but with zero contention from other jobs; see
    :func:`_solve_perjob`).

    dp_dedup: DP-aware checkpointing (D2, OURS ONLY). When True, parallelism-
    configured classes are priced at the effective shard (unique + replicated/dp);
    the emitted policy carries flags.dp_dedup so the sim reduces the same bytes.

    peerdram (L1.5, v2 only): add the opt-in `peerdram` placement option to the
    coordinate-descent shopping (or force it via placement='peerdram'). OFF by
    default: existing solves search the committed three placements unchanged.

    gamma_dram (L1.5): override the scenario's model.v2.gamma_dram DRAM price
    (see GAMMA_DRAM_DOC) without editing the file — the two-gamma report knob.

    placement_mode (Sam's placement ruling, 2026-07-27): "price-all" (default,
    UNCHANGED — the committed coordinate-descent price shopping) or
    "crossjob-first" (production rule: crossjob for every class where it is
    FEASIBLE — see _crossjob_first_feasibility — and only infeasible classes
    shop the fallback menu {injob, injob_ssd, local} by price; crossjob never
    loses to injob on price alone when feasible).

    cf_epsilon (the epsilon-margin refinement of crossjob-first, Sam's rule
    formalized 2026-07-27; crossjob-first only): None (default) keeps the
    plain pin-when-feasible rule byte-identical; a value makes each FEASIBLE
    class choose crossjob UNLESS an alternative beats crossjob's priced
    objective by more than cf_epsilon RELATIVE (exit gate in the placement
    descent; infeasible classes still take the fallback menu regardless).

    price_dram_rent (resource bill, 2026-07-28; v2 only, REPORT-ONLY flag):
    charge the `injob` placement its in-job DRAM replica rent, gamma_dram x
    resident replica GB (one full shard per rank in the model's k=1 injob
    geometry). Needs gamma_dram > 0 (set model.v2.gamma_dram or --gamma-dram,
    documented value GAMMA_DRAM_DOC). OFF by default — placement pricing and
    every emitted policy stay byte-identical."""
    sc = yaml.safe_load(scenario.read_text())
    return solve_dict(sc, name=str(scenario), backend=backend,
                      model_version=model_version,
                      measured_drain_gbps=measured_drain_gbps,
                      placement=placement, solve_mode=solve_mode,
                      dp_dedup=dp_dedup, peerdram=peerdram,
                      gamma_dram=gamma_dram, placement_mode=placement_mode,
                      cf_epsilon=cf_epsilon,
                      price_dram_rent=price_dram_rent)


def _nu_by_placement(m: dict, nu3_corr: float) -> dict:
    """Failure-mass -> (nu1,nu2,nu3) tier remapping per placement (2026-07-21).

    process death always survives to DRAM (nu1). node/spot organic loss survives
    to the peer copy (nu2) under crossjob/injob but only to L3 (nu3) under local.
    Whole-job EVICTION survives to the peer copy (nu2) ONLY under crossjob; injob
    replicas and local copies die with the job, so its mass rolls to L3 (nu3).

    reboot (2026-07-21, reboot_failure_type_spec.md): the host restarts on the
    SAME node — DRAM is lost but the LOCAL DISK survives. So reboot mass rolls
    back to peer age f2 under crossjob/injob (the DRAM replica on the OTHER node
    survives; crossjob pieces survive trivially) AND to LOCAL SSD age f2 under
    `local` — the f2 knob for a local-placed job IS its own-SSD write cadence,
    which the host restart leaves intact. This is where `local` finally earns a
    coverage niche: before reboot existed its nu2 was ~0, so f2 insured nothing
    and local was strictly dominated.

    peerdram (L1.5, l15_peer_dram_spec.md): pieces on OTHER hosts survive every
    OWNER event — node/spot/reboot AND whole-job eviction (the sim's ground
    truth: eviction only wipes pieces the evicted job HOSTS, not its pieces on
    donors; demoted pieces are ordinary L2 by recovery time anyway). Freshness
    for that whole mass = 1/(2 f1.5), where f1.5 rides the f2 slot. HOST-side
    volatility (a dram piece dying with its host before demotion) is priced by
    the availability factor a = m['dram_avail'] ~= 1: the un-covered sliver
    (1-a) of the off-node mass falls to L3 age.

    injob_ssd (rack_failure_spec.md): k=1 full replica on a same-job ring
    neighbor's SSD. Without rack events its coverage EQUALS injob's — the
    off-node replica survives owner process death (nu1 anyway), single
    node/spot loss AND host reboot (non-volatile — reboot wipes DRAM only),
    and dies with whole-job eviction (in-job copies go with the job) -> same
    (proc, ns + reb, corr + evict) split. The SSD medium buys nothing in this
    first-order single-failure pricing; it exists for the rack axis below
    (and, in the sim, for compound events the solver does not price).

    RACK SLICE (rack_failure_spec.md + post-spec decisions 2026-07-27). A new
    owner-rack failure mass lam_rack = m['lam_rack'] (= rack_weight x the
    job's organic node mass, at rack granularity — build_model_inputs) is
    remapped per placement by blast-radius geometry; there is NO waiting
    branch anywhere (the rack's nodes+copies are gone for the rest of the
    run; jobs restart on replacements from the best OFF-RACK copy):

      crossjob   pieces are rack-ANTI-AFFINE (k distinct racks, none the
                 owner's), so an owner-rack event leaves the whole stripe
                 available -> the lam_rack mass STAYS at L2 freshness. The
                 entire L2 mass (organic + eviction + rack) is then degraded
                 by the donor-side availability factor a_rack =
                 m['rack_l2_avail'] = P(>= k-1 of k pieces not dark)
                 (XOR-parity / k-1 stitch recoverability, the same tolerance
                 the sim grants donor loss; derivation in
                 build_model_inputs):
                   nu2 = a_rack * (ns + evict + reb + lam_rack)
                   nu3 = corr + (1 - a_rack) * (ns + evict + reb + lam_rack)
      injob      the DRAM replica sits in the owner's rack (contiguous
                 packing -> ring neighbor is usually SAME rack) and is
                 DESTROYED outright (volatile medium):    nu3 += lam_rack
      injob_ssd  the SSD replica sits in the owner's rack too. It is NOT
                 destroyed (non-volatile; destroyed only w.p. rack_p_destroy
                 per node) — but it is UNAVAILABLE for the rest of the run
                 (return timer U(rack_outage_s) typically fires beyond
                 horizon), so recovery remaps to L3 freshness all the same:
                   nu3 += lam_rack   (same remap as injob; the volatile/
                 non-volatile distinction is documented, not priced — it only
                 matters if the rack returns within horizon, which the
                 decisions rule out first-order)
      local      own SSD is in the failed rack:            nu3 += lam_rack
      peerdram   pieces rack-anti-affine like crossjob -> the rack mass stays
                 at L1.5 freshness, discounted by BOTH availability factors:
                   nu2 = a * a_rack * (off + lam_rack)
                   nu3 = corr + (1 - a * a_rack) * (off + lam_rack)

    Scratch corner: with p_destroy = rack_p_destroy, a destroyed same-rack
    copy AND no L3 copy would mean restart-from-scratch. The solve always
    prices a live L3 (f3 >= f3_floor > 0 via the operator backstop), so the
    p_destroy * P(no L3) corner carries ZERO first-order mass here; the sim
    charges it when it actually happens.

    Zero-nu guard: lam_rack == 0 (rack_weight = 0 or knobs absent) SHORT-
    CIRCUITS to the committed mapping — the same float expressions, no rack
    arithmetic, no division — so pre-rack scenarios stay byte-identical."""
    proc, ns, evict = m["lam_proc"], m["lam_ns"], m["lam_evict"]
    reb = m.get("lam_reboot", 0.0)
    corr = m["ranks"] * nu3_corr
    a = float(m.get("dram_avail", 1.0))
    off = ns + evict + reb
    lam_rack = float(m.get("lam_rack", 0.0))
    if lam_rack <= 0.0:
        return {
            "crossjob": (proc, ns + evict + reb, corr),    # peer stripe survives reboot
            "injob":    (proc, ns + reb, corr + evict),    # off-node DRAM replica survives
            "injob_ssd": (proc, ns + reb, corr + evict),   # == injob w/o rack events
            "local":    (proc, reb, corr + ns + evict),    # own SSD survives reboot (f2)
            "peerdram": (proc, a * off, corr + (1.0 - a) * off),   # L1.5 (a ~= 1)
        }
    ar = float(m.get("rack_l2_avail", 1.0))
    l2_cross = ns + evict + reb + lam_rack      # anti-affine stripe rides out racks
    off_r = off + lam_rack
    return {
        "crossjob": (proc, ar * l2_cross, corr + (1.0 - ar) * l2_cross),
        "injob":    (proc, ns + reb, corr + evict + lam_rack),   # DRAM destroyed
        "injob_ssd": (proc, ns + reb, corr + evict + lam_rack),  # SSD dark > horizon
        "local":    (proc, reb, corr + ns + evict + lam_rack),   # own rack dark
        "peerdram": (proc, a * ar * off_r, corr + (1.0 - a * ar) * off_r),
    }


# A cross-job checkpoint needs at least one donor.  When capacity planning
# yields zero, enact the already-supported owner-local SSD + store backstop
# placement instead of issuing a zero-width cross-job stripe.
# L1.5: a zero-width peerdram stripe is equally meaningless -> local.
def _enact_placement_for_k(placement: str, k: int) -> str:
    return ("local" if placement in ("crossjob", "peerdram") and k == 0
            else placement)


# tiny positive stand-ins that ZERO a class's DEMAND while keeping its ranks (so
# the shard/weight terms vanish but the class still counts as donor CAPACITY and
# node geometry). Must stay > the optimizer's EPS (1e-12) so the GP stays DGP.
_PERJOB_EPS_SHARD = 1e-9
_PERJOB_EPS_WEIGHT = 1e-9


def _solve_perjob(sc: dict, name: str, backend: str, model_version: str,
                  measured_drain_gbps: float | None, placement: str,
                  dp_dedup: bool = False) -> dict:
    """SELFISH per-job baseline — the controller-vs-daemon discriminator.

    Each class is solved ALONE with the SAME v2 cost model + placement search,
    but assuming ZERO contention from other jobs. Faithfulness (be smart but
    selfish, not a strawman):

      * KNOWS static cluster geometry. For each subject class C we solve on a view
        where every OTHER class is retained as pure idle-donor CAPACITY: its ranks
        / rates / count are kept (so C's total-rank count, per-wave donor pool,
        donor-duty capacity eta*H*N and store budget B3 are the REAL cluster's) but
        its DEMAND is zeroed (shard,weight -> eps). Hence C's geometric k-cap still
        reads 0.9*cap*others/own against the whole cluster — a lone mega still caps
        at k=3, not 0.
      * Does NOT know other jobs exist. Only C's own demand loads the shared
        resources in C's solve: the donor-duty LHS is C's own duty, the tail-
        stretch S(u,n) sees only C's utilization, and the store budget B3 is priced
        as if entirely C's. Each class therefore over-books the shared store/donor
        pool exactly as a selfish scheduler would; the collision only materialises
        when the assembled per-class policies are SIMULATED together.

    The per-class results are stitched into one policy (each class keeps its own
    selfishly-solved kpeers / cadences / placement). Slots are RANDOM (no phase
    coordination) and the enacted flags run the SAME peer-striping daemon with
    random phases instead of the coordinated closed loop — there is no central
    controller in this baseline. This does NOT fork the optimizer: it reuses the
    joint solve path once per class."""
    classes = list(sc["classes"])
    assembled: dict = {}
    per_class: dict = {}
    for subject in classes:
        sub = copy.deepcopy(sc)
        for cname, spec in sub["classes"].items():
            if cname == subject:
                continue
            # idle donor: keep ranks/count/rates (geometry) -> zero demand. Drop
            # any parallelism block so apply_model_shards does not re-inject a
            # real shard over the eps (model: classes must zero like legacy ones).
            spec.pop("model", None)
            spec.pop("_model_shard", None)
            spec["checkpoint_gb_per_rank"] = _PERJOB_EPS_SHARD
            spec["weight"] = _PERJOB_EPS_WEIGHT
            spec["rpo_s"] = None
        pol = solve_dict(sub, name=f"{name}#perjob:{subject}", backend=backend,
                         model_version=model_version,
                         measured_drain_gbps=measured_drain_gbps,
                         placement=placement, solve_mode="joint",
                         dp_dedup=dp_dedup)
        for jid, pj in pol["jobs"].items():
            if pj["class"] == subject:
                assembled[jid] = pj
        sv = pol.get("solver", {})
        pv = sv.get("placement", {}) or {}
        v2 = sv.get("v2", {}) or {}
        per_class[subject] = {
            "placement": (pv.get("by_class") or {}).get(subject),
            "shopping": (pv.get("shopping") or {}).get(subject),
            "donor_duty_util": v2.get("donor_duty_util"),
            "store_demand_gbps": v2.get("store_demand_gbps"),
            "B3_store_gbps": v2.get("B3_store_gbps"),
            "objective": sv.get("objective")}
    # RANDOM slot assignment — the selfish baseline does NOT coordinate phases
    # (the enacted flags use random_phases so the sim decorrelates per seed; the
    # slot_s here is a reproducible stand-in kept for policy-shape parity).
    period = float(sc["controller"]["slot_period_s"])
    rng = random.Random("perjob-slots")
    for jid in sorted(assembled):
        assembled[jid]["slot_s"] = round(rng.uniform(0.0, period), 2)
    flags = {"kpeers": True, "random_phases": True}
    if dp_dedup:
        flags["dp_dedup"] = True
    return {"name": "perjob" + ("_dedup" if dp_dedup else ""),
            "scenario": name,
            "solver": {"model_version": model_version, "solve_mode": "perjob",
                       "placement_mode": placement, "per_class": per_class},
            "flags": flags,
            "jobs": assembled}


def solve_dict(sc: dict, name: str = "<dict>", backend: str = "cvxpy",
               model_version: str = "v1",
               measured_drain_gbps: float | None = None,
               placement: str = "solve", solve_mode: str = "joint",
               dp_dedup: bool = False, peerdram: bool = False,
               gamma_dram: float | None = None,
               placement_mode: str = "price-all",
               cf_epsilon: float | None = None,
               price_dram_rent: bool = False) -> dict:
    """Same solve, but over a pre-loaded scenario DICT — the entry point the
    Stage-3 churn controller calls in-process to RE-SOLVE for the current active
    cluster composition (a sub-scenario of the arrived classes) on every
    arrival/departure. `name` only labels the emitted policy.

    dp_dedup (D2): price parallelism-configured classes at the DP-aware effective
    shard; recorded in flags.dp_dedup so the sim reduces the same bytes.

    placement_mode (Sam's placement ruling, 2026-07-27): "price-all" (default)
    keeps the committed coordinate-descent shopping BYTE-IDENTICAL (regression
    gate: re-solving miniC_ablation_full / rack_miniC_ablation must equal the
    committed policy JSONs exactly). "crossjob-first" is the PRODUCTION rule:
    per class, choose crossjob whenever FEASIBLE (feasibility = the per-wave
    donor duty-capacity headroom grants a genuine stripe AND k distinct
    non-owner donor racks exist under rack-anti-affinity; see
    _crossjob_first_feasibility); infeasible classes fall back to the cheapest
    of {injob, injob_ssd, local} by the normal priced objective (injob_ssd only
    when the scenario prices rack events, matching the search lattice rule).
    Crossjob is deliberately NEVER price-compared against the fallback menu
    when feasible — that is the point of the rule. v2 + placement='solve' +
    joint mode only.

    cf_epsilon (Sam's rule formalized, 2026-07-27; crossjob-first only): the
    epsilon-margin refinement. None (default) = the plain rule above,
    byte-identical. A value epsilon makes each FEASIBLE class choose crossjob
    UNLESS an alternative option's priced objective beats crossjob's by MORE
    than epsilon RELATIVE, evaluated at the placement descent's current
    assignment (an EXIT GATE: once a class clears the margin it shops by pure
    price — see optimize_placement_v2 prefer_by_class for why the gate is not
    re-tested every pass). The hard feasibility gate stays on top unchanged:
    INFEASIBLE classes take the fallback menu regardless of any margin."""
    if solve_mode not in ("joint", "perjob"):
        raise ValueError(f"solve_mode must be 'joint' or 'perjob', got {solve_mode!r}")
    if placement_mode not in ("price-all", "crossjob-first"):
        raise ValueError(f"placement_mode must be 'price-all' or "
                         f"'crossjob-first', got {placement_mode!r}")
    if placement_mode == "crossjob-first":
        if model_version != "v2":
            raise ValueError("crossjob-first requires --model-version v2 "
                             "(placement machinery is v2-only)")
        if placement != "solve":
            raise ValueError("crossjob-first conflicts with a forced "
                             "--placement; use --placement solve")
        if solve_mode != "joint":
            raise ValueError("crossjob-first is the joint controller's "
                             "production rule; perjob keeps price-all")
        if peerdram:
            raise ValueError("crossjob-first defines its own fallback menu "
                             "{injob, injob_ssd, local}; --peerdram is "
                             "incompatible")
        if cf_epsilon is not None and not 0.0 <= float(cf_epsilon) < 1.0:
            raise ValueError(f"cf_epsilon must be in [0, 1), got {cf_epsilon!r}")
    elif cf_epsilon is not None:
        raise ValueError("cf_epsilon is the epsilon-margin refinement of "
                         "crossjob-first; it requires --placement-mode "
                         "crossjob-first")
    if solve_mode == "perjob":
        return _solve_perjob(sc, name=name, backend=backend,
                             model_version=model_version,
                             measured_drain_gbps=measured_drain_gbps,
                             placement=placement, dp_dedup=dp_dedup)
    if model_version not in ("v1", "v2"):
        raise ValueError(f"model_version must be 'v1' or 'v2', got {model_version!r}")
    if model_version == "v2" and JobPhysical is None:
        raise RuntimeError(
            "model-version v2 requires a newer checkpointing/multilevel checkout; "
            "the installed cost-model dependency only provides v1"
        )
    # priced placement is a v2-only feature; validate it only when the newer
    # cost-model (which exports PLACEMENTS) is installed. On older checkouts
    # PLACEMENTS is None and any v2 request already tripped the RuntimeError above.
    # L1.5: `peerdram` is a legal FORCED placement (and a search option when
    # peerdram=True) only on checkouts exporting PLACEMENTS_L15. Rack:
    # `injob_ssd` likewise requires PLACEMENTS_RACK/PLACEMENTS_ALL.
    _all_placements = tuple(PLACEMENTS_ALL or PLACEMENTS_L15 or PLACEMENTS or ())
    if PLACEMENTS is not None and placement not in ("solve",) + _all_placements:
        raise ValueError(f"placement must be 'solve' or one of {_all_placements}, "
                         f"got {placement!r}")
    if (peerdram or placement == "peerdram") and PLACEMENTS_L15 is None:
        raise RuntimeError("the peerdram option requires a newer "
                           "checkpointing/multilevel checkout (PLACEMENTS_L15)")
    # rack pricing needs the injob_ssd-aware cost model whenever the scenario
    # prices rack events OR injob_ssd is forced; refuse loudly, never silently.
    _rack_on = float((sc.get("failures") or {}).get("rack_weight", 0.0)
                     or 0.0) > 0.0
    if (placement == "injob_ssd" or (_rack_on and model_version == "v2")) \
            and PLACEMENTS_RACK is None:
        raise RuntimeError("rack pricing / the injob_ssd option requires a newer "
                           "checkpointing/multilevel checkout (PLACEMENTS_RACK)")
    if _rack_on and model_version == "v1":
        print("WARNING: failures.rack_weight > 0 but model_version=v1 — the v1 "
              "solver has no placement machinery, so the rack slice is NOT "
              "priced (use --model-version v2)", file=sys.stderr)
    mi = build_model_inputs(sc, dp_dedup=dp_dedup)
    jobs, meta, async_model = mi.jobs, mi.meta, mi.async_model
    wave_util, hosted_cap = mi.wave_util, mi.hosted_cap
    B1, B2, B3 = mi.B
    total_ranks = sum(m["ranks"] for m in meta)

    v2p = None
    phys = None
    s_const = 1.0
    legacy_symmetry_classes = 0
    placement_by_class: dict = {}
    placement_info: dict = {}
    cf_feas: dict | None = None          # crossjob-first feasibility verdicts
    cf_fallback_menu: tuple = ()
    if model_version == "v2":
        v2p = build_v2_params(sc)
        if measured_drain_gbps is not None:     # plumb the sim-measured rate in
            v2p = dataclasses.replace(v2p, drain_bytes_rate=float(measured_drain_gbps))
        if gamma_dram is not None:              # L1.5 two-gamma report knob
            v2p = dataclasses.replace(v2p, gamma_dram=float(gamma_dram))
        if price_dram_rent and hasattr(v2p, "price_dram_rent"):
            # resource bill (2026-07-28): charge injob its in-job DRAM replica
            # rent (gamma_dram x shard GB). hasattr-guarded like gamma_dram for
            # older cost-model checkouts. OFF (default) = byte-identical solves.
            v2p = dataclasses.replace(v2p, price_dram_rent=True)
        # priced placement: an off-node peer/in-job copy must be REFRESHED at
        # least at the store-backstop cadence to stay durable (crossjob/injob pay
        # this maintenance NIC; local rides the shared L3 backstop and is exempt).
        # Only for the coordinate-descent search -- forced single-placement solves
        # (the all-crossjob A/B comparator) keep today's un-floored v2 pricing.
        if placement == "solve" and v2p.peer_maintain_floor <= 0.0:
            backstop_s = float((sc.get("store") or {}).get("backstop_s", 600.0))
            v2p = dataclasses.replace(v2p, peer_maintain_floor=1.0 / max(backstop_s, 1e-9))
        # real correlated/rack failure mass on tier 3 so f3 is ALLOCATED (§4)
        nu3_corr = float((sc.get("store") or {}).get(
            "nu3_correlated_per_node", NU3_CORRELATED_PER_NODE))
        jobs = [dataclasses.replace(jb, nu=(jb.nu[0], jb.nu[1],
                                            m["ranks"] * nu3_corr))
                for jb, m in zip(jobs, meta)]
        _jp_fields = {f.name for f in dataclasses.fields(JobPhysical)}
        _duplex = {"nic_in", "nic_out"} <= _jp_fields   # newer checkouts only
        phys = [JobPhysical(ranks=m["ranks"], shard_gb=m["shard_gb"],
                            nic=m["nic"], disk=m["disk"],
                            k_max=float(_v2_k_cap(meta, hosted_cap, j,
                                                  wave_util=wave_util)),
                            k_req=float(sc["classes"][m["class"]].get("kpeers", 2)),
                            **({"nic_in": m["nic_in"], "nic_out": m["nic_out"]}
                               if _duplex else {}))
                for j, m in enumerate(meta)]
        # PRICED PLACEMENT (Option B). Build the per-job per-placement nu, then
        # either coordinate-descend to the cheapest per-class placement ("solve")
        # or force a single placement for every class (A/B comparator / ablation).
        class_of = [m["class"] for m in meta]
        nubp = [_nu_by_placement(m, nu3_corr) for m in meta]
        # AUDIT FIX (2026-07-25): per-placement restore terms ride along with
        # the per-placement nu (local prices reboot restart + own-SSD read).
        rbp = [m["R_by_placement"] for m in meta]
        class_order = list(sc["classes"])         # scenario declaration order
        _opv2_params = inspect.signature(optimize_placement_v2).parameters
        if placement == "solve":
            _kw = {}
            # crossjob-first (Sam's placement ruling, 2026-07-27): pin every
            # FEASIBLE class to crossjob (rule-chosen, never price-compared)
            # and restrict infeasible classes to the fallback menu
            # {injob, injob_ssd, local} (injob_ssd only when the scenario
            # prices rack events — the same lattice rule as price-all). The
            # price-all path below is untouched (byte-identity regression).
            # epsilon refinement (cf_epsilon not None): FEASIBLE classes shop
            # the FULL lattice instead of being pinned, but the descent may
            # move them off crossjob only past the epsilon exit gate
            # (prefer_by_class/prefer_margin); infeasible classes keep the
            # fallback menu regardless — the hard gate is unchanged.
            if placement_mode == "crossjob-first":
                if "options_by_class" not in _opv2_params:
                    raise RuntimeError(
                        "crossjob-first requires a newer checkpointing/"
                        "multilevel checkout (optimize_placement_v2 "
                        "options_by_class)")
                cf_feas = _crossjob_first_feasibility(
                    sc, meta, total_ranks, hosted_cap, wave_util)
                _base = PLACEMENTS_RACK if _rack_on else PLACEMENTS
                cf_fallback_menu = tuple(p for p in _base if p != "crossjob")
                if cf_epsilon is None:
                    _kw["options_by_class"] = {
                        c: (("crossjob",) if cf_feas[c]["feasible"]
                            else cf_fallback_menu)
                        for c in class_order}
                else:
                    if "prefer_by_class" not in _opv2_params:
                        raise RuntimeError(
                            "cf-epsilon requires a newer checkpointing/"
                            "multilevel checkout (optimize_placement_v2 "
                            "prefer_by_class)")
                    _kw["options_by_class"] = {
                        c: (tuple(_base) if cf_feas[c]["feasible"]
                            else cf_fallback_menu)
                        for c in class_order}
                    _kw["prefer_by_class"] = {
                        c: "crossjob" for c in class_order
                        if cf_feas[c]["feasible"]}
                    _kw["prefer_margin"] = float(cf_epsilon)
            # shopping lattice: the committed three; + injob_ssd exactly when
            # the scenario prices rack events (rack_weight > 0 — keeps every
            # pre-rack policy JSON byte-identical, the regression gate);
            # + peerdram on the L1.5 opt-in flag. Omitting `options` entirely
            # when nothing is added preserves the committed call signature.
            if _rack_on and peerdram:
                _kw["options"] = PLACEMENTS_ALL
            elif _rack_on:      # rack: the full 2x2 + local
                _kw["options"] = PLACEMENTS_RACK
            elif peerdram:      # L1.5: opt the 4th option into the shopping
                _kw["options"] = PLACEMENTS_L15
            if "R_by_placement" in _opv2_params:
                _kw["R_by_placement"] = rbp
            res, placement_info = optimize_placement_v2(
                jobs, phys, B=mi.B, class_of=class_of, nu_by_placement=nubp,
                tau=1.0, model=async_model, v2=v2p,
                f_max=mi.f_max, r_max=mi.r_max, class_order=class_order, **_kw)
            placement_by_class = dict(placement_info["placement_by_class"])
            if cf_feas is not None:      # the crossjob-first sidecar printout
                _print_crossjob_first_sidecar(
                    cf_feas, placement_by_class,
                    placement_info.get("shopping", {}), cf_fallback_menu,
                    cf_epsilon=cf_epsilon)
        else:
            placement_by_class = {c: placement for c in class_order}
            forced = [placement] * len(jobs)
            jobs = [dataclasses.replace(jb, nu=tuple(nubp[j][placement]),
                                        R=tuple(rbp[j][placement]))
                    for j, jb in enumerate(jobs)]
            res = optimize_joint_async_gp_v2(
                jobs, phys, B=mi.B, tau=1.0, model=async_model, v2=v2p,
                f_max=mi.f_max, r_max=mi.r_max, placements=forced)
            res.placements = forced
        # re-derive jobs at the FINAL placement so the enacted-policy scoring
        # below uses the same nu/R remapping the solve was priced under
        placements_list = [placement_by_class[c] for c in class_of]
        jobs = [dataclasses.replace(jb, nu=tuple(nubp[j][placements_list[j]]),
                                    R=tuple(rbp[j][placements_list[j]]))
                for j, jb in enumerate(jobs)]
        s_const = (1.0 - min(res.v2_util or 0.0, v2p.u_cap)) ** (-v2p.beta)
    else:
        solver_parameters = inspect.signature(optimize_joint_async_gp).parameters
        solver_jobs = jobs
        expansion: list[tuple[int, int]] | None = None
        if "f_max" not in solver_parameters:
            # Older cost-model checkouts expose only the SciPy solver. Compress
            # identical jobs into one exact class-symmetric representative so
            # the 385-job balanced fleet does not become a 2,310-variable SLSQP
            # problem. For n identical jobs, use aggregate state n*s, aggregate
            # bandwidth R=n*r, and objective weight n*w; s/r and every failure
            # term are then unchanged, while the cluster capacity still sums R.
            groups: dict[str, list[int]] = {}
            for index, job_meta in enumerate(meta):
                groups.setdefault(job_meta["class"], []).append(index)
            if len(groups) < len(jobs):
                solver_jobs = []
                by_original_index: dict[int, tuple[int, int]] = {}
                for group_index, (class_name, indices) in enumerate(groups.items()):
                    count = len(indices)
                    representative = jobs[indices[0]]
                    solver_jobs.append(dataclasses.replace(
                        representative,
                        name=f"{class_name}[{count}]",
                        s_gb=representative.s_gb * count,
                        weight=representative.weight * count,
                    ))
                    for original_index in indices:
                        by_original_index[original_index] = (group_index, count)
                expansion = [by_original_index[index] for index in range(len(jobs))]
                legacy_symmetry_classes = len(solver_jobs)
        solver_kwargs = {
            "B": mi.B,
            "tau": 1.0,
            "max_iter": 20000,
            "tol": 1e-7,
            "model": async_model,
        }
        if "f_max" in solver_parameters:
            solver_kwargs["f_max"] = mi.f_max
        if "r_max" in solver_parameters:
            solver_kwargs["r_max"] = mi.r_max
        if "backend" in solver_parameters:
            solver_kwargs["backend"] = backend
        res = optimize_joint_async_gp(solver_jobs, **solver_kwargs)
        if expansion is not None:
            full_f = [list(res.f[group_index]) for group_index, _count in expansion]
            full_r = [
                [value / count for value in res.r[group_index]]
                for group_index, count in expansion
            ]
            evaluated = evaluate_async_policy(
                jobs, full_f, full_r, tau=1.0, model=async_model,
                method=res.method, status=res.status,
            )
            res = dataclasses.replace(
                evaluated,
                iterations=res.iterations,
                method=res.method,
                status=res.status,
            )
    if not res.converged:
        raise RuntimeError(f"GP did not converge: {res.status}")

    # donor BUDGET k: largest k <= requested s.t. one wave of k x ranks shards
    # fits WAVE_UTIL of what all OTHER jobs can host (#8.1 per-wave constraint).
    # v2: k comes from the SOLVE (rounded here); the v1 rule survives only as an
    # assertion that the coupled solve never over-stripes past donor capacity.
    policy_jobs = {}
    dram_slots_knob = int((sc.get("model") or {}).get("dram_host_slots", 1))
    for j, m in enumerate(meta):
        spec = sc["classes"][m["class"]]
        plc = placement_by_class.get(m["class"], "crossjob")
        k_req = int(spec.get("kpeers", 2))
        others = hosted_cap * (total_ranks - m["ranks"])
        k_cap = max(0, min(k_req, int(wave_util * others / m["ranks"])))
        # L1.5: the peerdram stripe is bounded by the per-wave DRAM slot pool
        # (dram_host_slots per donor rank), the dram analogue of k_cap.
        k_cap_dram = max(0, min(k_req, int(
            wave_util * dram_slots_knob * (total_ranks - m["ranks"])
            / m["ranks"])))
        if model_version == "v2":
            if plc == "peerdram":
                # solver kub is the float eta-bound; enact the integer cap
                k = (0 if k_cap_dram == 0
                     else min(max(1, round(res.k[j])), k_cap_dram))
            else:
                k = 0 if plc == "crossjob" and k_cap == 0 else max(1, round(res.k[j]))
            if plc == "crossjob":
                assert k <= k_cap, (
                    f"v2 over-striped {m['job_id']}: solved k={res.k[j]:.3f} "
                    f"-> {k} exceeds donor feasibility cap {k_cap}")
        else:
            k = k_cap
        # A zero-width cross-job stripe is not a checkpoint path. Switch to the
        # existing local + owner-push-store placement instead of attempting an
        # impossible peer reservation for every rank.
        plc = _enact_placement_for_k(plc, k)
        # priced-placement enacted stripe width: crossjob keeps the solved k;
        # injob replicates to INJOB_DEGREE in-job peers; injob_ssd is BY
        # DEFINITION the k=1 ring-neighbor SSD replica (the sim's recovery
        # source label for it is `injob_ssd_replica`); local writes no peers.
        kpeers = (0 if plc == "local"
                  else min(k, INJOB_DEGREE) if plc == "injob"
                  else 1 if plc == "injob_ssd" else k)
        # the DURABLE cadence the sim enacts: crossjob/injob = the peer tier f2;
        # local = the L3 store tier f3 (its own SSD copy is not node-loss durable,
        # so the store backstop IS its durable copy -> drive checkpoint_every from
        # f3 so the owner-push backstop rides every persist wave).
        durable_f = res.f[j][2] if plc == "local" else res.f[j][1]
        interval_s = 1.0 / max(durable_f, 1e-12)
        # frequency-RPO (#8.3), audit fix #20: enforce on the ENACTED cadence
        # (after rounding to iterations), not the pre-rounding interval
        rpo = spec.get("rpo_s")
        every = max(1, round(interval_s / m["iter_wall"]))
        clamped = False
        if rpo is not None:
            cap_every = max(1, int((2 * float(rpo)) // m["iter_wall"]))
            if every > cap_every:
                every, clamped = cap_every, True
        enacted_interval = every * m["iter_wall"]
        # f1 enactment: snapshot cadence from the solved DRAM-tier frequency
        # (hierarchy guarantees f1 >= f2 -> capture_every <= checkpoint_every)
        cap_every = max(1, round((1.0 / max(res.f[j][0], 1e-12)) / m["iter_wall"]))
        cap_every = min(cap_every, every)
        # f3 enactment: store backstop. nu3 ~ 0 makes the solved f3 degenerate,
        # so an operator floor bounds it (like RPO): backstop at least every
        # `backstop_s` (scenario store.backstop_s, default 600 s).
        # LOCAL placement: the store IS the durable copy, so the backstop rides
        # every persist wave (store_every == checkpoint_every).
        backstop_s = float(sc.get("store", {}).get("backstop_s", 600.0))
        if plc == "local":
            st_every = every
        else:
            st_every = max(1, round((1.0 / max(res.f[j][2], 1e-12)) / m["iter_wall"]))
            if model_version == "v2":
                # AUDIT FIX (2026-07-25): honor the SOLVED f3. The operator
                # backstop floor now lives INSIDE the v2 solve (budget-aware;
                # see build_v2_params), so clamping the enacted cadence faster
                # than solved would re-break the B3 budget the solve honored
                # (miniC_tight: 3.0x overshoot). Riding persist waves (below)
                # only ever SLOWS the cadence, which is allowed.
                pass
            else:
                # v1: the solved f3 is degenerate by design (nu3 ~ 0), so the
                # operator backstop_s clamp remains the floor here — unchanged.
                st_every = min(st_every, max(every, int(backstop_s // m["iter_wall"])))
            st_every = max(st_every, every)     # backstop rides persist waves
        # predicted flush duration (slot packing input): crossjob stripes over
        # k donors (k*disk); peerdram stripes to donor DRAM (owner NIC_out vs
        # k*donor NIC_in — homogeneous-pool proxy: own class's nic_in); injob
        # replicates over own NIC; injob_ssd crosses the in-job NIC AND lands
        # on a same-job node's SSD (bottleneck of the two, matching the solver's
        # two wave-speed caps); local writes own SSD.
        if plc == "crossjob":
            flush_bw = min(m["nic"], max(k, 1) * m["disk"])
        elif plc == "peerdram":
            flush_bw = min(m["nic_out"], max(k, 1) * m["nic_in"])
        elif plc == "injob":
            flush_bw = m["nic"]
        elif plc == "injob_ssd":
            flush_bw = min(m["nic"], m["disk"])
        else:
            flush_bw = m["disk"]
        if plc == "peerdram":
            # capacity sanity: a piece bigger than the per-node DRAM byte cap
            # would be refused at reservation — surface it at solve time.
            _default_gb = 2.0 * max(
                float(s2["checkpoint_gb_per_rank"])
                / max(int(s2.get("kpeers", 2)) or 1, 1)
                for s2 in sc["classes"].values())
            _gb_cap = float((sc.get("model") or {}).get(
                "dram_host_gb", _default_gb))
            if m["shard_gb"] / max(k, 1) > _gb_cap + 1e-9:
                print(f"WARNING: {m['job_id']} peerdram piece "
                      f"{m['shard_gb'] / max(k, 1):.2f} GB exceeds "
                      f"dram_host_gb {_gb_cap:.2f} — waves will be refused",
                      file=sys.stderr)
        policy_jobs[m["job_id"]] = {
            "class": m["class"], "kpeers": kpeers,
            "placement": plc,
            **({"dram_piece_gb": round(m["shard_gb"] / max(k, 1), 3)}
               if plc == "peerdram" else {}),
            "checkpoint_every": every,
            "capture_every": cap_every,
            "store_every": st_every,
            "solved_interval_s": round(interval_s, 2),
            "enacted_interval_s": round(enacted_interval, 2),
            "f2_hz": round(1.0 / enacted_interval, 6),
            "f2_hz_solved": round(durable_f, 6),
            # predicted flush duration (per-rank wave; slot packing input)
            "pred_flush_s": round(m["shard_gb"] / max(flush_bw, 1e-9), 2),
            "rpo_s": spec.get("rpo_s"), "rpo_clamped": clamped,
        }

    # PHASE (#8.1 stagger), audit fix #21: least-loaded CIRCULAR placement on a
    # 1 s load grid, weighted by each job's duty (pred_flush x period/interval —
    # a job flushing every 4th period only occupies 1/4 of the windows), instead
    # of a cursor that silently wraps whole jobs onto slot 0 under overload.
    period = float(sc["controller"]["slot_period_s"])
    grid = [0.0] * max(int(period), 1)
    true_duty = 0.0
    for job_id in sorted(policy_jobs, key=lambda x: -policy_jobs[x]["pred_flush_s"]):
        p = policy_jobs[job_id]
        span = max(1, min(len(grid), int(round(p["pred_flush_s"]))))
        occ = min(1.0, period / max(p["enacted_interval_s"], period))
        true_duty += p["pred_flush_s"] * occ / period
        best_off, best_load = 0, float("inf")
        for off in range(len(grid)):
            load = sum(grid[(off + t) % len(grid)] for t in range(span))
            if load < best_load - 1e-12:
                best_off, best_load = off, load
        for t in range(span):
            grid[(best_off + t) % len(grid)] += occ
        p["slot_s"] = float(best_off)
    # enacted per-tier frequency: f1 (DRAM) = solved; f2 = enacted durable cadence
    # (in the tier-2 slot for crossjob/injob); f3 = store cadence. For LOCAL the
    # durable copy IS the store (f3), and the tier-2 slot is inert (nu2~0), so
    # keep the solved f2 there and put the enacted store cadence in f3.
    enacted_f = []
    for j, m in enumerate(meta):
        pj = policy_jobs[m["job_id"]]
        plc = pj.get("placement", "crossjob")
        if plc == "local":
            f2_slot = res.f[j][1]
            f3_slot = 1.0 / max(pj["store_every"] * m["iter_wall"], 1e-9)
        else:
            f2_slot = 1.0 / pj["enacted_interval_s"]
            f3_slot = res.f[j][2]
        enacted_f.append([res.f[j][0], f2_slot, f3_slot])
    placements_list = ([policy_jobs[m["job_id"]]["placement"] for m in meta]
                       if model_version == "v2" else None)
    jobs_nofail = [dataclasses.replace(jb, nu=(1e-12, 1e-12, 1e-12)) for jb in jobs]
    if model_version == "v2":
        enacted_k = [policy_jobs[m["job_id"]]["kpeers"] for m in meta]
        enacted = evaluate_async_policy_v2(jobs, phys, enacted_f, res.r,
                                           enacted_k, async_model, v2p, s_const,
                                           placements=placements_list)
        overhead_only = evaluate_async_policy_v2(jobs_nofail, phys, enacted_f,
                                                 res.r, enacted_k, async_model,
                                                 v2p, s_const,
                                                 placements=placements_list)
    else:
        enacted = evaluate_async_policy(jobs, enacted_f, res.r, tau=1.0,
                                        model=async_model)
        overhead_only = evaluate_async_policy(jobs_nofail, enacted_f, res.r,
                                              tau=1.0, model=async_model)
    for j, m in enumerate(meta):
        pj = policy_jobs[m["job_id"]]
        pj["per_job_objective"] = round(enacted.per_job_objective[j], 4)
        fail_cost = max(0.0, enacted.per_job_objective[j]
                        - overhead_only.per_job_objective[j])
        pj["overhead_frac"] = round(overhead_only.per_job_objective[j], 4)
        # seconds of training lost to failures PER SECOND, at the optimal policy;
        # halving lambda is worth ~half of this — the operator's break-even
        # budget for healthier hardware / fewer evictions (lambda as a decision
        # variable: checkpoint frequency and reliability spend are substitutes)
        pj["failure_cost_s_per_s"] = round(fail_cost, 4)
        pj["value_of_halving_lambda_gpu_s_per_s"] = round(
            fail_cost / 2 * m["ranks"], 2)
    solver_meta = {"method": res.method, "iterations": res.iterations,
                   "objective": res.objective, "B": [B1, B2, B3],
                   "overload_x": round(true_duty, 2),
                   "model_version": model_version,
                   "dependency_api": (
                       "bounded" if "f_max" in inspect.signature(
                           optimize_joint_async_gp).parameters else "legacy"
                   ),
                   "symmetry_classes": legacy_symmetry_classes}
    if model_version == "v2":
        solver_meta["v2"] = {
            "donor_duty_util": round(res.v2_util or 0.0, 4),
            "fixed_point_drift": res.v2_util_drift,
            "alpha": v2p.alpha, "beta": v2p.beta, "eta": v2p.eta,
            "gamma": v2p.gamma, "gamma_provisional": True,
            # L1.5 knobs (only meaningful when the peerdram option is searched)
            "peerdram_option": bool(peerdram),
            "gamma_dram": float(getattr(v2p, "gamma_dram", 0.0)),
            "gamma_dram_doc": GAMMA_DRAM_DOC,
            "dram_host_slots": int(getattr(v2p, "dram_host_slots", 1)),
            "drain_bytes_rate": v2p.drain_bytes_rate,
            "store_binding": v2p.store_binding,
            "duty_capacity_slots": round(v2p.eta * v2p.hosted_cap * total_ranks, 1),
            "store_demand_gbps": round(
                sum(jobs[j].s_gb * res.f[j][2] for j in range(len(jobs))), 3),
            "B3_store_gbps": B3}
        # in-job DRAM rent flag: recorded ONLY when ON, so every committed
        # flag-OFF policy JSON stays byte-identical (the regression gate).
        if getattr(v2p, "price_dram_rent", False):
            solver_meta["v2"]["price_dram_rent"] = True
        # rack slice (rack_failure_spec.md): knobs + the per-class hazard and
        # L2 availability actually priced, auditable in the emitted policy.
        # GATED on rack_weight > 0 so every pre-rack policy JSON stays
        # byte-identical (the regression gate).
        if _rack_on:
            _f = sc["failures"]
            _seen_rack: dict = {}
            for m2 in meta:
                _seen_rack.setdefault(m2["class"], m2)
            solver_meta["rack"] = {
                "rack_weight": float(_f.get("rack_weight", 0.0)),
                "rack_size": int(_f.get("rack_size", 4)),
                "rack_p_destroy": float(_f.get("rack_p_destroy", 0.01)),
                "rack_outage_s": list(_f.get("rack_outage_s",
                                             [3600.0, 21600.0])),
                "lam_rack_by_class": {c: m2["lam_rack"]
                                      for c, m2 in _seen_rack.items()},
                "l2_avail_by_class": {c: m2["rack_l2_avail"]
                                      for c, m2 in _seen_rack.items()}}
        # priced placement (2026-07-21): the chosen per-class placement + the
        # per-class shopping table (each option's cluster objective) so the
        # decision is auditable in the emitted policy and in --explain.
        solver_meta["placement"] = {
            "mode": placement, "by_class": placement_by_class,
            "shopping": {c: {p: round(v, 6) for p, v in opts.items()}
                         for c, opts in placement_info.get("shopping", {}).items()},
            "search_passes": placement_info.get("passes"),
            "search_solves": placement_info.get("solves"),
            "search_converged": placement_info.get("converged_search")}
        # crossjob-first audit block — ONLY in crossjob-first mode, so every
        # price-all policy JSON stays byte-identical (the regression gate).
        if cf_feas is not None:
            solver_meta["placement"]["placement_mode"] = "crossjob-first"
            solver_meta["placement"]["min_stripe"] = CF_MIN_STRIPE
            solver_meta["placement"]["fallback_menu"] = list(cf_fallback_menu)
            solver_meta["placement"]["feasibility"] = cf_feas
            # epsilon refinement audit key — ONLY when epsilon mode is on, so
            # plain crossjob-first policy JSONs stay byte-identical (the
            # regression gate against the committed cf_* policies).
            if cf_epsilon is not None:
                solver_meta["placement"]["cf_epsilon"] = float(cf_epsilon)
    flags = {"kpeers": True, "slots": True, "controller_epoch_s": 120.0}
    if dp_dedup:
        flags["dp_dedup"] = True        # D2: sim reduces the same effective bytes
    return {"name": "ours_gp_rpo" + ("_v2" if model_version == "v2" else "")
            + ("_dedup" if dp_dedup else "") + ("_l15" if peerdram else "")
            + (("_cf" if cf_epsilon is None else "_cfe")
               if placement_mode == "crossjob-first" else ""),
            "scenario": name, "solver": solver_meta,
            "flags": flags, "jobs": policy_jobs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--backend", choices=("cvxpy", "scipy"), default="cvxpy",
                    help="GP solver backend (default cvxpy/CLARABEL)")
    ap.add_argument("--model-version", choices=("v1", "v2"), default="v1",
                    help="cost model: v1 (default, unchanged) or v2 "
                         "(globally coupled; cost_model_v2_design.md)")
    ap.add_argument("--measured-drain-gbps", type=float, default=None,
                    help="v2 only: sim-measured donor-drain bytes-rate (GB/s) fed "
                         "into the provisional donor hosting tax input (design §2)")
    ap.add_argument("--placement",
                    choices=("solve",) + tuple(PLACEMENTS_ALL or PLACEMENTS_L15
                                               or PLACEMENTS or ()),
                    default="solve",
                    help="v2 only: 'solve' (default) prices per-class placement in "
                         "{crossjob,injob,local} by coordinate descent (+injob_ssd "
                         "automatically when the scenario prices rack events, "
                         "failures.rack_weight > 0); a fixed "
                         "name forces every class to it (all-crossjob A/B, etc.). "
                         "peerdram (L1.5) forces the DRAM tier everywhere; to add "
                         "it to the SEARCH instead, pass --peerdram.")
    ap.add_argument("--placement-mode", choices=("price-all", "crossjob-first"),
                    default="price-all",
                    help="Sam's placement ruling (2026-07-27): 'price-all' "
                         "(default, unchanged) shops every option by price; "
                         "'crossjob-first' (PRODUCTION rule, v2 + placement "
                         "solve only) pins each class to crossjob whenever "
                         "FEASIBLE (per-wave donor duty-capacity headroom "
                         f"grants k >= {CF_MIN_STRIPE} AND k distinct "
                         "non-owner donor racks exist under rack-anti-"
                         "affinity) and lets only infeasible classes shop the "
                         "fallback menu {injob, injob_ssd, local} by price.")
    ap.add_argument("--cf-epsilon", type=float, nargs="?",
                    const=CF_EPSILON_DEFAULT, default=None,
                    help="epsilon-margin refinement of crossjob-first (Sam's "
                         "rule formalized, 2026-07-27; requires "
                         "--placement-mode crossjob-first): a FEASIBLE class "
                         "chooses crossjob UNLESS an alternative beats "
                         "crossjob's priced objective by more than epsilon "
                         "RELATIVE at the placement descent's current "
                         "assignment (exit gate; INFEASIBLE classes still "
                         "take the fallback menu regardless of margin). Bare "
                         f"flag = {CF_EPSILON_DEFAULT}; omit the flag for the "
                         "plain pin-when-feasible rule (committed behavior, "
                         "byte-identical).")
    ap.add_argument("--peerdram", action="store_true",
                    help="L1.5: add the opt-in `peerdram` placement option to the "
                         "coordinate-descent shopping (l15_peer_dram_spec.md). "
                         "Off = the committed 3-option search, unchanged.")
    ap.add_argument("--gamma-dram", type=float, default=None,
                    help="L1.5: override model.v2.gamma_dram (DRAM price, node-s "
                         f"per GB-s; documented value {GAMMA_DRAM_DOC}). Report "
                         "runs use 0 and the documented value.")
    ap.add_argument("--price-dram-rent", action="store_true",
                    help="Resource bill (REPORT-ONLY): charge the injob "
                         "placement its in-job DRAM replica rent, gamma_dram x "
                         "resident replica GB (one full shard per rank). Needs "
                         "gamma_dram > 0. OFF by default — solves are byte-"
                         "identical to the committed policies.")
    ap.add_argument("--solve-mode", choices=("joint", "perjob"), default="joint",
                    help="'joint' (default, unchanged): the cluster-wide coupled "
                         "controller. 'perjob': the SELFISH per-job baseline — each "
                         "class solved ALONE against real cluster geometry with zero "
                         "contention (controller-vs-daemon discriminator)")
    ap.add_argument("--dp-dedup", action="store_true",
                    help="D2 DP-aware checkpointing (OURS): price parallelism-"
                         "configured classes at the effective shard "
                         "(unique + replicated/dp); emits flags.dp_dedup so the "
                         "sim reduces the same bytes. No-op for classes without a "
                         "model: block.")
    args = ap.parse_args()
    pol = solve(args.scenario, backend=args.backend,
                model_version=args.model_version,
                measured_drain_gbps=args.measured_drain_gbps,
                placement=args.placement, solve_mode=args.solve_mode,
                dp_dedup=args.dp_dedup, peerdram=args.peerdram,
                gamma_dram=args.gamma_dram,
                placement_mode=args.placement_mode,
                cf_epsilon=args.cf_epsilon,
                price_dram_rent=args.price_dram_rent)
    args.out.write_text(json.dumps(pol, indent=1))
    print("wrote", args.out)
    if args.explain:
        seen = set()
        for jid, p in pol["jobs"].items():
            if p["class"] in seen:
                continue
            seen.add(p["class"])
            print(f'{p["class"]:12s} [{p.get("placement","crossjob"):8s}] '
                  f'k={p["kpeers"]} every={p["checkpoint_every"]} '
                  f'(solved {p["solved_interval_s"]}s, f2={p["f2_hz"]}Hz) '
                  f'store_every={p["store_every"]} '
                  f'slot={p["slot_s"]}s pred_flush={p["pred_flush_s"]}s '
                  f'| failure-cost {p["failure_cost_s_per_s"]:.4f} s/s '
                  f'-> halving lambda worth {p["value_of_halving_lambda_gpu_s_per_s"]} GPU-s/s')
        # priced-placement shopping table (the per-class comparison Sam asked for)
        plc = pol["solver"].get("placement")
        if plc and plc.get("shopping"):
            print(f'[placement] mode={plc["mode"]} '
                  f'(search: {plc.get("search_passes")} passes, '
                  f'{plc.get("search_solves")} solves, '
                  f'converged={plc.get("search_converged")})')
            for c, opts in plc["shopping"].items():
                chosen = plc["by_class"].get(c)
                shop = "  ".join(
                    f'{("*" if p == chosen else " ")}{p}={opts[p]:.5f}'
                    for p in opts)          # row keys = searched option lattice
                print(f'  {c:12s} -> {chosen:8s} | {shop}')
        print("solver:", json.dumps(pol["solver"]))
        v2 = pol["solver"].get("v2")
        if v2:
            print(f'[v2] donor-duty util u={v2["donor_duty_util"]} '
                  f'(cap {v2["duty_capacity_slots"]} slots, fixed-point drift '
                  f'{v2["fixed_point_drift"]}); store demand '
                  f'{v2["store_demand_gbps"]}/{v2["B3_store_gbps"]} GB/s '
                  f'(binding={v2["store_binding"]}); tail alpha={v2["alpha"]} '
                  f'beta={v2["beta"]}')
            dr = v2.get("drain_bytes_rate") or 0.0
            drain_note = (f' + measured donor-drain {dr} GB/s fed into the tax '
                          f'input' if dr else
                          ' (measured donor-drain rate 0 -> not fed; pass '
                          '--measured-drain-gbps)')
            if v2["gamma"]:
                print(f'*** PROVISIONAL donor hosting tax gamma={v2["gamma"]}'
                      f'{drain_note} — UNCALIBRATED: the drain rate is measured '
                      f'but gamma is not yet fit to a donor-side slowdown. ***')
            else:
                print(f'[v2] donor hosting tax OFF (gamma=0){drain_note}; term '
                      'implemented, PROVISIONAL until gamma is calibrated.')
        rack = pol["solver"].get("rack")
        if rack:
            lam = " ".join(f'{c}={v:.3e}'
                           for c, v in rack["lam_rack_by_class"].items())
            print(f'[rack] w={rack["rack_weight"]} size={rack["rack_size"]} '
                  f'p_destroy={rack["rack_p_destroy"]} '
                  f'outage={rack["rack_outage_s"]}s | owner-rack hazard/s: {lam}')


if __name__ == "__main__":
    main()
