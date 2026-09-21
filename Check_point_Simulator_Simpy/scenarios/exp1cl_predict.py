#!/usr/bin/env python3
"""exp1cl_predict — sealed M0/M1/M2 predictions for the exp1cl campaign.

Wednesday 2026-08-05 (CloudLab r7525 x2), plan:
meetings_summary/2026-08-03_wednesday_exact_plan.md §3 (arms 1-9 + optional
arm 10 dj_flip) + §4 (seal discipline).
Spec of record: checkpointing/realgpu_coefficient/exp1cl_scenario.yaml
(Sam's 08-04 2x2 redesign is folded in there: jobs A AND B are both 2-node
DDP; job C is gone).

    python3 scenarios/exp1cl_predict.py --constants <yaml> [--stamp]
    python3 scenarios/exp1cl_predict.py --selftest

Reads the on-site measured constants (written 10:50-11:20 MT), builds the
exp1cl arms as simulator scenarios, runs run_scenario.py under all THREE
coupling models x BOTH capture models, and emits
sealed_predictions_exp1cl.json:

  M0  count-based (shipping)          both coupling flags off
  M1  fabric-scoped (CL-014)          model.fabric_aware_coupling: true
  M2  intensity    (CL-016, S3)       model.intensity_coupling: true

plus, NEW, each model's CAPTURE VARIANT (CL-017):

  M0c/M1c/M2c = the same coupling flags + model.nonblocking_capture: true
  (the D2H capture takes no GPU lock). exp1 hardware evidence says capture
  OVERLAPS training (capture 413 ms vs 485 ms iteration, p99 stretch 1.002,
  "The capture does not stall training at all" — 2026-07-31_exp1_results.md
  §5 point 3; PAPER_NOTES.md 2026-08-01; protocol §7.4 CL-017). The blocking
  columns carry the per-cadence capture floor; the 'c' columns have NONE.
  Predictions are sealed BOTH ways so the hardware arbitrates: dj/off-
  adjacent alpha at the floor -> the blocking model wins and CL-017 is
  refuted; at ~0 -> CL-017 confirmed.

THE 2x2 DESIGN (Sam 08-04; replaces the single-GPU B and job C):
  Job A  2-node DDP, GPU0 of cl1+cl2, GPT-2 774M fp16, trains on the arm's
         train fabric; rank 0 streams the 12.4 GB state to cl2's NVMe spool.
  Job B  2-node DDP, GPU1 of cl1+cl2, SAME model shape, its OWN rendezvous
         (MASTER_PORT = A's per-cell port + 10). NEVER checkpoints; hosts A's
         stripes on cl2. Its slowdown is PURELY gamma. B's train fabric is
         FIXED PER BLOCK (jobs.B.train_fabric_by_block in the spec):
           25 Gb block  (off, sat25, sat25_tc10, sat25_tc5, sat25_d15;
                         A on path1=eth25)            -> B on path2 = eth100p0
           100 Gb block (off100, dj_t100_c25, sh100, sameCard, dj_flip)
                                                      -> B on path3 = eth100p1
         so gamma's baseline always shares B's fabric. gamma = B's mean dt
         ratio against its BLOCK's baseline (off / off100) under the SAME
         model. FOOTNOTE (accepted): arm sameCard's ckpt stream rides
         eth100p1 too — it shares path3 with B's collective in that arm.
         (dj_flip's A trains on path1, but its B/gamma block is the 100 Gb
         one; its ALPHA baseline is off, its GAMMA baseline is off100.)
  Job C  REMOVED (no probe job; nothing launches on a third GPU slot).

alpha/gamma are computed exactly as exp1_arms.py does (METRICS.md §7): mean
dt_s over the MEASURE phase, rank 0, pooled across seeds, ratio against the
arm's BLOCK baseline under the SAME model — alpha against the arm's `baseline`
(A's train-fabric block), gamma against GAMMA_BASELINE (B's fabric block).

=============================================================================
CONSTANTS — every model input, with provenance. NOTHING here is fitted.
=============================================================================
From the constants yaml (exp1cl_scenario.yaml `cluster.measured`, or a flat
`constants:` yaml). All measured on-site BEFORE the seal:
  nccl_allreduce_25g_gbps    busbw of A's collective on eth25   -> A rates.nic
  nccl_allreduce_100g_gbps   busbw on eth100p0                  -> A rates.nic
  b_busbw_eth100p0_gbps      busbw of B's collective on eth100p0 (the 25 Gb
                             block's B fabric), same on-site NCCL ladder;
                             ABSENT/null -> falls back to
                             nccl_allreduce_100g_gbps (same wire; recorded)
  b_busbw_eth100p1_gbps      busbw of B's collective on eth100p1 (the 100 Gb
                             block's B fabric; path3 is nameplate 100 Gb);
                             ABSENT/null -> falls back to b_busbw_eth100p0
                             (same card, other port; recorded)
  compute_ms_774m_fp16       single-GPU iteration median -> iteration_seconds
                             for BOTH classes. If absent, derived as
                             iter_ms_774m_fp16 - ring ar_base on iter_ms_fabric
                             (or from jobs[B].measured.iter_ms - B's ring
                             ar_base on eth100p0; derivation recorded).
  capture_d2h_gbps           ckpt_probe capture rate -> gpu_cpu_bandwidth_gbps
  peer_stream_25g_gbps       single TCP shard stream on eth25 -> stream demand
  peer_stream_100g_gbps      same on eth100; ABSENT -> falls back to the 25g
                             value (single-stream TCP is CPU-bound; recorded)
  spool_write_gbps           receiver write rate -> donor disk cap; ABSENT
                             -> 999 NON-BINDING (recorded; exp1's discipline:
                             an unmeasured constant must not invent a term)
Fixed by the hardware spec (not measured):
  line rates 25 / 100 Gbit -> /8 GB/s. CL-012.D S4: the intensity model's link
     capacity is the NAMEPLATE rate, min(rates.nic_in, rates.nic_out).
  state 12.4 GB (774M x 16 B/param), fp16 gradient bytes 1.548 GB/rank.
  cadence 60 s (sat25_d15: 15 s), 300 measured + 30 warmup iters, seeds {0,1}.

=============================================================================
HOW THE ARMS MAP ONTO THE SIMULATOR (deliberate choices, all disclosed)
=============================================================================
* Class A rides its arm's TRAIN fabric: rates.nic = measured busbw there,
  rates.nic_fabric = the fabric label, rates.nic_in/nic_out = that fabric's
  NAMEPLATE line rate (the M2 link capacity AND the solver's node caps).
* Class B is a REAL 2-rank training class now: rates.nic = the measured
  b_busbw of its block's fabric (-> its ar_base AND its M2 collective
  demand), rates.nic_fabric = that fabric, rates.nic_in/nic_out = the 100 Gb
  NAMEPLATE (B always rides a 100 Gb port; per CL-012.D S4 that is B's M2
  link capacity). As the donor-ingress cap it never binds — stream demand
  (<= ~2.5-3 GB/s) is far below any wire here — and in the ONE arm whose
  stripe shares B's fabric (sameCard) the ckpt wire IS eth100p1 with the
  same nameplate, so the two roles coincide exactly where it matters.
* The checkpoint stream cap is the DONOR demand. The sim computes donor
  demand as min(B.rates.nic, B.rates.disk); B.nic now carries b_busbw, so
  the demand knob moved to B.rates.disk/disk_w =
  min(peer_stream_<ckpt fabric>, tc shape/8, spool_write) — the identical
  effective cap the old mirror enforced (its ceiling already min'd stream,
  tc and spool), valid while b_busbw >= that stream demand (true for the
  dummies and any sane ladder; a violation is printed and recorded).
* STRIPE TOPOLOGY: hardware streams the FULL 12.4 GB from A rank 0 only.
  The mirror gives each of A's two ranks a 6.2 GB shard (cohorts: 2) landing
  on the two B workers, so TOTAL BYTES per wave match but two streams run
  concurrently at the per-stream rate: the mirror's wave lasts about HALF
  the hardware's single-stream wave. Same family of disclosed gap as exp1's
  mirror; per-arm stripe_dur_s is stated in the seal so the duty is explicit.
* NODES: 4 sim nodes = 4 cohort workers (cl1: A r0 + B r0, cl2: A r1 + B r1
  + spool). Each physical node's TWO workers are modelled as SEPARATE
  capacity nodes — intra-node PCIe/memory contention is NOT modelled; the
  hardware gamma includes it (disclosed as more representative of
  multi-tenant packing, per the plan §2).
* gamma is now EXPRESSIBLE (the old mirror's B had ranks=1 -> ar_base 0 ->
  gamma structurally 0; that gap is CLOSED by this redesign):
    M0  a stripe in flight on B's nodes stretches B's collective by
        (1 + tasks) whichever wire it rides — gamma > 0 in EVERY stripe arm,
        fabric-blind, duration-driven (tc ladder: gamma RISES as the cap
        falls).
    M1  the stripe counts only if it rides B's OWN wire — gamma = 0 in every
        arm except sameCard (ckpt eth100p1 = B's eth100p1), where it is the
        same count charge.
    M2  charges only if stripe demand + B's collective busbw exceed B's
        nameplate link. With the dummy constants no arm over-subscribes
        (sameCard offers ~8.0 + 2.5 = 10.5 of 12.5 GB/s), so gamma_M2 = 0
        everywhere — and B never captures, so B has NO capture floor: the
        zero is exact. Discriminating claim: hardware gamma > 0 in sameCard
        would refute M2's clean-path assumption.
* sameCard ALPHA: the sim carries eth100p0 / eth100p1 as DISTINCT fabrics
  with their own nameplates and no card-level sharing term, so M1/M2 predict
  alpha(sameCard) == alpha(dj) by construction. That IS the sim's
  prediction; the hardware prices what "fake disjoint" actually costs. Its
  GAMMA now differs between M1 (>0) and M2 (0) — a second discriminator.
* Every arm's ALPHA carries the capture floor (state/capture rate once per
  cadence) UNDER THE BLOCKING COLUMNS ONLY: the plan's "alpha ~ 0" shapes are
  about the COUPLING term above that floor. The seal states per-arm
  capture_dur_s so the floor is explicit. The 'c' columns (CL-017) take no
  GPU lock, so they carry NO floor — their "~0" is a literal ~0. gamma
  carries NO floor either way (B never captures).
* The plan's "identical alpha for sh100 and sat25" under M0/M1 is exact for
  the coupling FACTOR (1 + count, blind to load). End-to-end alpha also
  carries each block's ar_base/iteration-wall ratio and the stripe duration,
  which differ between blocks, so the mirror's M0/M1 alphas are same-order,
  not equal. The selftest asserts the load-blind SIGNATURE (unsaturated
  sh100 charged the same order as saturated sat25) and reports the literal
  difference with this reason.

--selftest runs everything with DUMMY constants (plausible 25/100 Gb numbers,
stream rates deliberately EQUAL across fabrics and b_busbw deliberately EQUAL
across B's two ports, so count-based cross-block comparisons are clean) and
asserts the discriminating shapes from plan §4 plus the NEW gamma shapes,
plus the CL-017 capture-column shapes (C.1-C.3): the blocking columns are
byte-consistent with the pre-CL-017 generator (the flag defaults OFF on the
identical code path), the 'c' columns lose exactly the capture floor and
keep every model's ordering signature.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
BUILD = HERE / "exp1cl_build"

sys.path.insert(0, str(HERE))
import exp1_emit  # noqa: E402

STATE_GB = 12.4               # 774M x 16 B/param (params + grads + adam m,v)
PARAMS_M = 774
GRADIENT_GB = round(2 * PARAMS_M * 1e6 / 1e9, 6)   # fp16 grads = 1.548 GB/rank
RANKS_A = 2
RANKS_B = 2                   # Sam 08-04: B is 2-node DDP too (GPU1 of cl1+cl2)
ITER_MEASURED = 300           # run.iterations
WARMUP = 30                   # run.warmup_iters
SEQ_LEN = 1024

MODELS = {
    # blocking capture (default): capture holds the GPU lock -> per-cadence
    # capture floor in every ckpt arm's alpha.
    "M0": {},                                     # count-based, shipping
    # 'c' = + model.nonblocking_capture (CL-017): capture takes no GPU lock,
    # floor gone; the capture DURATION still gates the persist pipeline.
    "M0c": {"nonblocking_capture": True},
    "M1": {"fabric_aware_coupling": True},        # CL-014
    "M1c": {"fabric_aware_coupling": True, "nonblocking_capture": True},
    "M2": {"intensity_coupling": True},           # CL-016 (implies fabric scope)
    "M2c": {"intensity_coupling": True, "nonblocking_capture": True},
}
BLOCKING = ("M0", "M1", "M2")     # the pre-CL-017 columns (byte-consistent)

# (id, train_fabric, ckpt_fabric, cadence_s, shape_gbit, alpha_baseline, note)
# exact run order from exp1cl_scenario.yaml `arms` / plan §3 (+ optional 10).
ARMS = [
    ("off",         "eth25",    None,       None, None, None,
     "baseline iteration time for the saturated (25 Gb) block"),
    ("off100",      "eth100p0", None,       None, None, None,
     "baseline for the unsaturated (100 Gb) block"),
    ("sat25",       "eth25",    "eth25",    60.0, None, "off",
     "THE arm: forced contention - all-reduce + stripe together exceed 25 Gb"),
    ("sat25_tc10",  "eth25",    "eth25",    60.0, 10,   "off",
     "contention ladder point"),
    ("sat25_tc5",   "eth25",    "eth25",    60.0, 5,    "off",
     "contention ladder point (longer stripes, ~33% duty)"),
    ("sat25_d15",   "eth25",    "eth25",    15.0, None, "off",
     "saturation x high duty: worst case for alpha and gamma"),
    ("dj_t100_c25", "eth100p0", "eth25",    60.0, None, "off100",
     "separate cards - the paper's premise at a realistic ckpt rate"),
    ("sh100",       "eth100p0", "eth100p0", 60.0, None, "off100",
     "shared-unsaturated: does exp1's zero replicate?"),
    ("sameCard",    "eth100p0", "eth100p1", 60.0, None, "off100",
     "two ports of ONE card - prices the AWS-EFA 'fake disjoint' config. "
     "FOOTNOTE (accepted): its ckpt shares path3 with B's collective"),
    ("dj_flip",     "eth25",    "eth100p0", 60.0, None, "off",
     "OPTIONAL arm 10 (Sam): disjoint FLIPPED - ckpt fabric FASTER than "
     "train; alpha vs off (A trains eth25), gamma vs off100 (B on path3)"),
]

# B's train fabric, FIXED PER BLOCK (spec jobs.B.train_fabric_by_block):
# the 25 Gb block's B rides path2 (eth100p0); the 100 Gb block's B rides
# path3 (eth100p1). Never A's train fabric; constant within a block so
# gamma's baseline shares B's fabric.
B_FABRIC = {
    "off": "eth100p0", "sat25": "eth100p0", "sat25_tc10": "eth100p0",
    "sat25_tc5": "eth100p0", "sat25_d15": "eth100p0",
    "off100": "eth100p1", "dj_t100_c25": "eth100p1", "sh100": "eth100p1",
    "sameCard": "eth100p1", "dj_flip": "eth100p1",
}
# gamma baseline = the off arm of B's fabric block. Equals the alpha baseline
# for every arm except dj_flip (alpha vs off, gamma vs off100).
GAMMA_BASELINE = {arm: (None if arm in ("off", "off100")
                        else "off" if fab == "eth100p0" else "off100")
                  for arm, fab in B_FABRIC.items()}

# plan §4, restated VERBATIM (the seal carries these so the discriminating
# outcomes are on record before any interference arm runs). The gamma shapes
# are the 08-04 redesign's addition (source: exp1cl_scenario.yaml jobs.B +
# derived: block), sealed alongside.
# ---- SINGLE-WIRE mode (2026-08-05 CloudLab reality: one 25 Gb link, vlan266).
# All flows share eth25. Hardware additionally runs offA/satA (B absent) as
# HARDWARE-ONLY exploratory cells (jobshare coefficient); they are NOT sealed
# because the mirror always builds both jobs. Sealed set = B-present arms,
# baseline offAB for alpha AND gamma. NOTE (sealed alongside): the sim carries
# no job-vs-job collective coupling term, so busbw constants are calibrated
# with BOTH jobs running (the offAB steady state); jobshare itself is a
# hardware-only observation, like gamma was in exp1.
ARMS_SW = [
    ("offAB",    "eth25", None,    None, None, None,
     "baseline: A + B share the wire, no ckpt"),
    ("sat",      "eth25", "eth25", 60.0, None, "offAB",
     "3-flow crash test: A collective + B collective + stripe on one 25 Gb link"),
    ("sat_tc10", "eth25", "eth25", 60.0, 10,   "offAB", "capped ladder point"),
    ("sat_tc5",  "eth25", "eth25", 60.0, 5,    "offAB", "capped ladder point, ~33% duty"),
    ("sat_d15",  "eth25", "eth25", 15.0, None, "offAB", "saturation x duty worst case"),
]
B_FABRIC_SW = {a[0]: "eth25" for a in ARMS_SW}
GAMMA_BASELINE_SW = {a[0]: (None if a[0] == "offAB" else "offAB") for a in ARMS_SW}

SHAPE_CLAIMS = {
    "source": "meetings_summary/2026-08-03_wednesday_exact_plan.md §4 (alpha, "
              "verbatim); exp1cl_scenario.yaml jobs.B / derived (gamma, Sam's "
              "08-04 2x2 redesign)",
    "sealed_before_campaign": True,
    "model_prediction_shapes": {
        "M0": "α = same nonzero value wherever a transfer is in flight on the "
              "job's fabric — **identical α for `sh100` and `sat25`**, blind "
              "to load",
        "M1": "α = 0 on `dj_t100_c25`; still count-based on shared arms — "
              "**still identical `sh100` vs `sat25`**",
        "M2": "α ≈ 0 on `sh100` (link unsaturated); α > 0 on `sat25*`, "
              "**rising** as stream cap ↑ and as cadence ↓; α = 0 on `dj`",
    },
    "gamma_prediction_shapes": {
        "M0": "γ > 0 in EVERY stripe arm — a stripe in flight on B's nodes "
              "stretches B's collective whichever wire it rides (fabric-"
              "blind); duration-driven, so the tc ladder's γ RISES as the "
              "cap falls (tc5 > tc10 > sat25)",
        "M1": "γ = 0 wherever the stripe rides a different wire than B's "
              "collective — every arm except `sameCard` (ckpt eth100p1 = "
              "B's eth100p1), where the count charge applies",
        "M2": "γ = 0 in every arm at these rates: the one shared-wire arm "
              "(`sameCard`) offers b_busbw + stream ≪ the 100 Gb nameplate, "
              "and an unsaturated link charges NOTHING. B never captures, so "
              "the zero is capture-floor-free and exact",
    },
    "capture_model_claims": {
        "source": "control-plane/design/sim_validation_protocol.md §7.4 "
                  "CL-017; meetings_summary/2026-07-31_exp1_results.md §5 "
                  "point 3; PAPER_NOTES.md 2026-08-01",
        "flag": "model.nonblocking_capture (DEFAULT OFF; the 'c' columns "
                "M0c/M1c/M2c set it; blocking columns M0/M1/M2 do not)",
        "evidence": "exp1 hardware: capture 413 ms vs 485 ms iteration, p99 "
                    "stretch 1.002 — 'The capture does not stall training at "
                    "all'",
        "claim": "exp1 evidence says capture overlaps; if hardware dj/"
                 "off-adjacent arms show alpha at the capture floor instead "
                 "of ~0, the blocking model wins and CL-017 is refuted",
    },
    "discriminating_outcomes": [
        "Hardware `sat25` α > 0 while `sh100` α ≈ 0 → **kills the count-based "
        "coupling** (M0/M1 on shared fabric) and supports intensity. This is "
        "the expected outcome.",
        "Hardware `sat25` α ≈ 0 too → contention doesn't materialize even at "
        "over-offered load; intensity model is falsified in this range and we "
        "say so; the sim keeps a documented limitation instead of a fix.",
        "`sameCard` α ≈ `dj` α → two-ports-one-card is harmless at these rates "
        "and our AWS disqualification needs softening; `sameCard` α > `dj` α → "
        "it's confirmed, with a number.",
        "GAMMA (new, 08-04 design): hardware γ ≈ 0 on the sat-block arms "
        "kills M0's receiver-side fabric-blindness (B rides path2 there, a "
        "different card than the stripe); hardware γ on `sameCard` separates "
        "M1 (count: γ > 0) from M2 (intensity: γ = 0) — and γ > 0 there "
        "would refute M2's clean-path assumption with a number.",
    ],
    "pass_bar": "Pass bar per metric: same as protocol τ (~5% band per Mark's "
                "ruling; ReCycle's own sim error is 5.98%). All comparisons "
                "are PREDICTION error — no post-hoc retuning; misses become "
                "change-log entries with pre-registered fixes or declared "
                "limitations.",
    "mirror_caveats": [
        "Every arm's simulated alpha includes the capture floor "
        "(state/capture rate once per cadence) in the BLOCKING columns "
        "(M0/M1/M2); the '~0' shapes are about the coupling term ABOVE that "
        "floor (per-arm capture_dur_s is stated). The 'c' columns "
        "(M0c/M1c/M2c, model.nonblocking_capture per CL-017) take no GPU "
        "lock and carry NO floor. gamma has NO capture floor under either "
        "capture model: B never captures.",
        "The M0/M1 'identical sh100 vs sat25' is exact for the coupling "
        "factor (count-based, load-blind); end-to-end alpha also carries each "
        "block's ar_base/iteration-wall ratio and stripe duration, so the "
        "mirror's M0/M1 alphas are same-order, not equal.",
        "B is a 2-rank DDP class with its own per-block fabric, so simulated "
        "gamma is now EXPRESSIBLE (the pre-08-04 mirror's structural zero is "
        "closed). sameCard FOOTNOTE (accepted): that arm's ckpt stream rides "
        "eth100p1, the same wire as B's collective — the one arm where the "
        "stripe and B share a path.",
        "Wave duration: the mirror's two concurrent 6.2 GB streams (one per "
        "A rank, one per B donor) complete in about HALF the hardware's "
        "single-stream 12.4 GB wave at the same per-stream rate; total bytes "
        "per wave match. Per-arm stripe_dur_s is stated in the seal.",
        "cl1/cl2 each host one A and one B worker; the sim models the four "
        "workers as four capacity nodes — intra-node PCIe/memory contention "
        "is NOT modelled (hardware gamma includes it; disclosed).",
        "sameCard == dj under M1/M2 for ALPHA by construction: the simulator "
        "has no card-level sharing term for two ports of one card. Their "
        "GAMMA differs under M1 (sameCard > 0) — a modelled, not card-level, "
        "effect: the stripe rides B's wire there.",
    ],
    "emergent_discriminator": (
        "Not in the plan's table but visible in the sealed numbers: the tc "
        "ladder's ORDERING inverts between the models. M0/M1 alpha FALLS as "
        "the shaped cap rises (a smaller cap makes the stripe LONGER, and the "
        "count-based coupling charges by duration: tc5 > tc10 > sat25). M2 "
        "alpha RISES with the cap (a faster stream pushes the shared wire "
        "further past capacity: tc5 < tc10 < sat25). The hardware ladder's "
        "monotonicity direction alone discriminates count from intensity, "
        "independent of the sh100-vs-sat25 comparison. The M0 GAMMA ladder "
        "shows the same duration signature on the RECEIVER side (tc5 > tc10 "
        "> sat25) while M1/M2 pin those arms' gamma at 0 — B rides a "
        "different card there."),
}

DUMMY_CONSTANTS = {
    # SELFTEST ONLY — plausible, NOT measured. Stream rates are deliberately
    # EQUAL across fabrics (single-stream TCP is CPU-bound) and b_busbw is
    # deliberately EQUAL across B's two ports, so count-based cross-block
    # comparisons are not confounded by stripe duration or B's wall time.
    "nccl_allreduce_25g_gbps": 2.7,     # busbw, 2-node, eth25 (21.6 Gbit)
    "nccl_allreduce_100g_gbps": 8.0,    # busbw, 2-node, eth100p0 (64 Gbit)
    "b_busbw_eth100p0_gbps": 8.0,       # B's ladder, eth100p0 (== A's wire)
    "b_busbw_eth100p1_gbps": 8.0,       # B's ladder, eth100p1 (== on purpose)
    "compute_ms_774m_fp16": 400.0,      # single-GPU 774M fp16 median
    "capture_d2h_gbps": 6.0,            # D2H capture rate
    "peer_stream_25g_gbps": 2.5,        # single TCP stream, eth25
    "peer_stream_100g_gbps": 2.5,       # single TCP stream, eth100 (== on purpose)
    "spool_write_gbps": 3.0,            # receiver spool write
    "batch_calibrated": 8,
    "line_rate_25_gbit": 25,
    "line_rate_100_gbit": 100,
}

REQUIRED = ("nccl_allreduce_25g_gbps", "nccl_allreduce_100g_gbps",
            "capture_d2h_gbps", "peer_stream_25g_gbps")


def load_constants(path: Path) -> tuple[dict, dict]:
    """Return (constants, provenance). Accepts either a flat constants yaml
    ({constants: {...}} or top-level keys) or exp1cl_scenario.yaml itself
    (reads cluster.measured / jobs / fabrics)."""
    doc = yaml.safe_load(path.read_text())
    prov: dict[str, str] = {}
    c: dict = {}
    b_iter_ms = None            # jobs[B].measured.iter_ms (2-node DDP median)
    if "cluster" in doc and "measured" in doc.get("cluster", {}):
        meas = doc["cluster"]["measured"] or {}
        for k, v in meas.items():
            if v is not None:
                c[k] = v
                prov[k] = f"{path.name}:cluster.measured (on-site)"
        for j in doc.get("jobs", []):
            if j.get("id") == "B" and (j.get("measured") or {}).get("iter_ms"):
                # 08-04 redesign: B is 2-node DDP now, so its iter_ms is NOT
                # pure compute anymore — held for derivation below.
                b_iter_ms = float(j["measured"]["iter_ms"])
            if j.get("id") == "A":
                b = (j.get("train") or {}).get("batch")
                if b:
                    c.setdefault("batch_calibrated", b)
        fabs = ((doc["cluster"].get("nodes") or [{}])[0].get("fabrics") or {})
        if "eth25" in fabs:
            c.setdefault("line_rate_25_gbit", fabs["eth25"].get("nominal_gbps", 25))
        if "eth100p0" in fabs:
            c.setdefault("line_rate_100_gbit",
                         fabs["eth100p0"].get("nominal_gbps", 100))
    else:
        flat = doc.get("constants", doc)
        for k, v in flat.items():
            if v is not None:
                c[k] = v
                prov[k] = f"{path.name} (on-site)"
    missing = [k for k in REQUIRED if k not in c]
    if missing:
        raise SystemExit(
            f"REFUSING TO PREDICT: unmeasured required constants {missing} in "
            f"{path} — fill them from the 10:50-11:20 calibration first "
            "(nothing here may be assumed).")
    # B's collective busbw (its ar_base + M2 demand). Fallback chain, recorded:
    # p0 <- the 100g ladder (SAME wire — nccl_allreduce_100g_gbps IS measured
    # on eth100p0; B's GPU1 pair assumed to match it), p1 <- p0 (same card,
    # other port, same nameplate). Measure b_busbw_* on-site to remove both.
    if "b_busbw_eth100p0_gbps" not in c:
        c["b_busbw_eth100p0_gbps"] = c["nccl_allreduce_100g_gbps"]
        prov["b_busbw_eth100p0_gbps"] = (
            "FALLBACK = nccl_allreduce_100g_gbps (same wire eth100p0; B's "
            "GPU1 ring assumed to match the ladder — measure to remove)")
    if "b_busbw_eth100p1_gbps" not in c:
        c["b_busbw_eth100p1_gbps"] = c["b_busbw_eth100p0_gbps"]
        prov["b_busbw_eth100p1_gbps"] = (
            "FALLBACK = b_busbw_eth100p0_gbps (same card, other 100 Gb port; "
            "measure the p1 ladder on-site to remove)")
    # compute time: direct measurement preferred; else derive from a 2-node
    # DDP median by subtracting the ring ar_base on the calibration fabric
    # (A's iter_ms_774m_fp16 first, else B's jobs[B].measured.iter_ms).
    if "compute_ms_774m_fp16" not in c:
        if "iter_ms_774m_fp16" in c:
            fab = c.get("iter_ms_fabric", "eth100p0")
            bus = (c["nccl_allreduce_25g_gbps"] if fab == "eth25"
                   else c["nccl_allreduce_100g_gbps"])
            ar = 2 * (RANKS_A - 1) / RANKS_A * GRADIENT_GB / float(bus)
            c["compute_ms_774m_fp16"] = float(c["iter_ms_774m_fp16"]) - ar * 1000.0
            prov["compute_ms_774m_fp16"] = (
                f"DERIVED: iter_ms_774m_fp16 ({c['iter_ms_774m_fp16']}) - ring "
                f"ar_base on {fab} ({ar*1000:.1f} ms); prefer a direct "
                "single-GPU measurement")
        elif b_iter_ms is not None:
            # B calibrates on eth100p0 (its 25 Gb-block fabric) per the spec.
            ar = (2 * (RANKS_B - 1) / RANKS_B * GRADIENT_GB
                  / float(c["b_busbw_eth100p0_gbps"]))
            c["compute_ms_774m_fp16"] = b_iter_ms - ar * 1000.0
            prov["compute_ms_774m_fp16"] = (
                f"DERIVED: jobs[B].measured.iter_ms ({b_iter_ms}) - B's ring "
                f"ar_base on eth100p0 ({ar*1000:.1f} ms); prefer a direct "
                "single-GPU measurement")
        else:
            raise SystemExit(
                "REFUSING TO PREDICT: need compute_ms_774m_fp16 (single-GPU "
                "median) or iter_ms_774m_fp16 / jobs[B].measured.iter_ms "
                "(2-node DDP medians) to derive it.")
    if "peer_stream_100g_gbps" not in c:
        c["peer_stream_100g_gbps"] = c["peer_stream_25g_gbps"]
        prov["peer_stream_100g_gbps"] = (
            "FALLBACK = peer_stream_25g_gbps (unmeasured; single-stream TCP "
            "assumed CPU-bound, not wire-bound)")
    if "spool_write_gbps" not in c:
        c["spool_write_gbps"] = 999.0
        prov["spool_write_gbps"] = (
            "UNMEASURED -> 999 NON-BINDING (no invented disk term; if the real "
            "spool binds below the stream rate the sim under-predicts stripe "
            "time — stated exposure, same discipline as exp1's NVMe)")
    c.setdefault("batch_calibrated", 8)
    c.setdefault("line_rate_25_gbit", 25)
    c.setdefault("line_rate_100_gbit", 100)
    return c, prov


def _fabric_tab(c: dict) -> dict:
    """fabric -> busbw GB/s for a collective there (A's ladder / B's ladder),
    nameplate GB/s, single-stream cap."""
    n25 = float(c["line_rate_25_gbit"]) / 8.0
    n100 = float(c["line_rate_100_gbit"]) / 8.0
    return {
        "eth25":    {"busbw": float(c["nccl_allreduce_25g_gbps"]),
                     # single-wire mode: B rides eth25 too; fall back to A's
                     # measured busbw (symmetric jobs, same wire, both TCP)
                     "b_busbw": float(c.get("b_busbw_eth25_gbps")
                                      or c["nccl_allreduce_25g_gbps"]),
                     "nameplate": n25,
                     "stream": float(c["peer_stream_25g_gbps"])},
        "eth100p0": {"busbw": float(c["nccl_allreduce_100g_gbps"]),
                     "b_busbw": float(c["b_busbw_eth100p0_gbps"]),
                     "nameplate": n100,
                     "stream": float(c["peer_stream_100g_gbps"])},
        # port 1 of the SAME card: same nameplate; A's collective never rides
        # it in these arms, busbw only as a defensive fallback.
        "eth100p1": {"busbw": float(c["nccl_allreduce_100g_gbps"]),
                     "b_busbw": float(c["b_busbw_eth100p1_gbps"]),
                     "nameplate": n100,
                     "stream": float(c["peer_stream_100g_gbps"])},
    }


def arm_derived(c: dict, arm_row) -> dict:
    """Per-arm derived numbers (pure arithmetic on declared constants)."""
    arm, train_f, ckpt_f, cadence, shape_gbit, baseline, note = arm_row
    tab = _fabric_tab(c)
    compute_s = float(c["compute_ms_774m_fp16"]) / 1000.0
    busbw = tab[train_f]["busbw"]
    ar_base = 2 * (RANKS_A - 1) / RANKS_A * GRADIENT_GB / busbw
    wall = compute_s + ar_base
    every = None
    if cadence is not None:
        every = max(1, round(cadence / wall))
    stream_cap = None
    if ckpt_f is not None:
        stream_cap = tab[ckpt_f]["stream"]
        if shape_gbit is not None:
            stream_cap = min(stream_cap, shape_gbit / 8.0)
    # B's side (08-04 redesign): its own fabric, busbw, ar_base, wall.
    b_fab = B_FABRIC[arm]
    b_busbw = tab[b_fab]["b_busbw"]
    b_ar = 2 * (RANKS_B - 1) / RANKS_B * GRADIENT_GB / b_busbw
    b_wall = compute_s + b_ar
    b_nameplate = tab[b_fab]["nameplate"]
    # M2 over-subscription test for B's link: only a stripe on B's OWN fabric
    # contributes demand there (only sameCard qualifies among these arms).
    b_oversub = bool(ckpt_f is not None and ckpt_f == b_fab
                     and stream_cap is not None
                     and b_busbw + stream_cap > b_nameplate + 1e-12)
    return {"compute_s": compute_s, "ar_base_s": ar_base, "iter_wall_s": wall,
            "checkpoint_every": every, "stream_cap_gbps": stream_cap,
            "train_nameplate_gbps": tab[train_f]["nameplate"],
            "ckpt_nameplate_gbps": (tab[ckpt_f]["nameplate"] if ckpt_f else None),
            "b_fabric": b_fab, "b_busbw_gbps": b_busbw,
            "b_ar_base_s": b_ar, "b_iter_wall_s": b_wall,
            "b_nameplate_gbps": b_nameplate,
            "m2_b_oversubscribed": b_oversub}


def build_scenario(c: dict, arm_row, model_flags: dict, seeds: list[int],
                   iterations: int) -> dict:
    arm, train_f, ckpt_f, cadence, shape_gbit, baseline, note = arm_row
    d = arm_derived(c, arm_row)
    shard_gb = STATE_GB / RANKS_A          # per-rank shard; total == hardware
    never = iterations + 1
    a_rates = {
        "nic": _fabric_tab(c)[train_f]["busbw"],   # MEASURED busbw -> ar_base
        "nic_fabric": train_f,                      # wire the collective rides
        # NAMEPLATE line rate of the train fabric: the M2 link capacity
        # (CL-012.D S4) AND this node's solver caps. Never binds a stream
        # below its own wire (stream demand <= ckpt nameplate in every arm).
        "nic_in": d["train_nameplate_gbps"],
        "nic_out": d["train_nameplate_gbps"],
        "disk": 999.0, "disk_w": 999.0,             # inert: no local persist
    }
    if ckpt_f is not None:
        a_rates["ckpt_fabric"] = ckpt_f             # wire the stripe rides
    # donor stream DEMAND (what the sim reads as min(B.nic, B.disk)): the
    # measured single-stream rate on the ckpt fabric, tc-shaped where the arm
    # shapes it, bounded by the spool write rate — the same effective cap the
    # pre-08-04 mirror enforced via its ceiling. Inert in off arms.
    stream_demand = (min(d["stream_cap_gbps"], float(c["spool_write_gbps"]))
                     if d["stream_cap_gbps"] is not None
                     else float(c["spool_write_gbps"]))
    if d["b_busbw_gbps"] < stream_demand - 1e-12:
        # min(B.nic, B.disk) would bind the stripe below its intended demand.
        # Cannot happen with sane constants (a 100 Gb ring busbw under a
        # single TCP stream rate); printed AND recorded rather than silently
        # mis-capping.
        print(f"WARNING {arm}: b_busbw {d['b_busbw_gbps']} < stream demand "
              f"{stream_demand} — the mirror under-caps the stripe (recorded)",
              file=sys.stderr)
    b_rates = {
        # B is a REAL 2-rank class now: nic = MEASURED b_busbw of its block's
        # fabric -> its ar_base AND its M2 collective demand.
        "nic": d["b_busbw_gbps"],
        "nic_fabric": d["b_fabric"],                # per-block (B_FABRIC map)
        # B's train-fabric NAMEPLATE (always a 100 Gb port): B's M2 link
        # capacity per CL-012.D S4. As the donor-ingress cap it never binds
        # (stream demand ~2.5-3 GB/s), and in the one arm whose stripe shares
        # B's fabric (sameCard) the ckpt wire IS eth100p1 = same nameplate.
        "nic_in": d["b_nameplate_gbps"],
        "nic_out": d["b_nameplate_gbps"],
        # the donor demand knob (see stream_demand above): disk/disk_w carry
        # min(stream, tc, spool); B itself never checkpoints so its own disk
        # use is inert.
        "disk": stream_demand,
        "disk_w": stream_demand,
    }
    sc = {
        "simulation": {"seeds": list(seeds), "max_sim_s": 36000.0,
                       "engine": "event", "emit_iterations": True},
        "cluster": {
            # 4 sim nodes = 4 cohort workers (cl1: A r0 + B r0; cl2: A r1 +
            # B r1 + spool). Each physical node's TWO workers are separate
            # capacity nodes; intra-node PCIe/memory sharing is not modelled
            # (disclosed — the hardware gamma includes it).
            "node_count": 4,
            "gpu_cpu_bandwidth_gbps": float(c["capture_d2h_gbps"]),
            "network_bandwidth_gbps": _fabric_tab(c)[train_f]["busbw"],
            "local_ssd_bandwidth_gbps": 999.0,
            "object_store_bandwidth_gbps": 999.0,
        },
        "classes": {
            "A": {
                "count": 1, "ranks": RANKS_A, "cohorts": RANKS_A,
                "iteration_seconds": d["compute_s"],
                "gradient_gb_per_rank": GRADIENT_GB,
                "weights_gb_per_rank": GRADIENT_GB,
                "checkpoint_gb_per_rank": shard_gb,
                "iterations": iterations,
                "checkpoint_every": d["checkpoint_every"] or never,
                "kpeers": 1, "rpo_s": 60.0,
                "rates": a_rates,
            },
            "B": {
                # 2-node DDP donor (own rendezvous on hardware: A's per-cell
                # port + 10). NEVER checkpoints; hosts A's stripes. Its own
                # iteration_seconds (same measured single-GPU compute; its
                # wall differs from A's via its own fabric's ar_base).
                "count": 1, "ranks": RANKS_B, "cohorts": RANKS_B,
                "iteration_seconds": d["compute_s"],
                "gradient_gb_per_rank": GRADIENT_GB,
                "weights_gb_per_rank": GRADIENT_GB,
                "checkpoint_gb_per_rank": 0.0,      # never flushed (spec: 0)
                "iterations": iterations,
                "checkpoint_every": 1000000,
                "kpeers": 0, "rpo_s": 60.0,
                "rates": b_rates,
            },
        },
        "failures": {"per_node_per_second": 0.0, "inject_random": False,
                     "weights": {"process": 0.4, "node": 0.2, "spot": 0.4},
                     "restart_seconds": {"process": 5.0, "node": 20.0,
                                         "spot": 30.0}},
        "store": {"in_gbps": 999.0, "out_gbps": 999.0, "disk_gbps": 999.0,
                  "stream_gbps": 999.0, "backstop_s": 300.0},
        "controller": {"slot_period_s": 60.0},
        "backstop": "owner_push",
        "arms": {arm: {"kpeers": True, "slots": False}},
        "_exp1cl_arm": {"arm": arm, "train_fabric": train_f,
                        "ckpt_fabric": ckpt_f, "cadence_s": cadence,
                        "shape_gbit": shape_gbit, "alpha_baseline": baseline,
                        "b_fabric": d["b_fabric"],
                        "gamma_baseline": GAMMA_BASELINE[arm],
                        "note": note, **{k: round(v, 9) for k, v in d.items()
                                         if isinstance(v, float)}},
    }
    if model_flags:
        sc["model"] = dict(model_flags)
    return sc


def build_policy(arm_row, every: int | None, iterations: int,
                 scenario_path: Path, iter_wall: float) -> dict:
    arm = arm_row[0]
    never = iterations + 1
    return {
        "name": arm,
        "scenario": str(scenario_path),
        "solver": {"method": "hand_set_mirror_of_exp1cl_scenario.yaml",
                   "note": "not a solved policy: enacts the HARDWARE's hand-set "
                           "config (A: cadence per arm, k=1, cross-job full "
                           "state to B's spool; B: never checkpoints)"},
        "flags": {"kpeers": True, "slots": False, "backstop": "owner_push"},
        "jobs": {
            "A0": {"class": "A", "kpeers": 1,
                   "checkpoint_every": every if every else never,
                   "capture_every": None, "store_every": None,
                   "placement": "crossjob",
                   "enacted_interval_s": (round(every * iter_wall, 3)
                                          if every else None)},
            "B0": {"class": "B", "kpeers": 0, "checkpoint_every": never,
                   "capture_every": None, "store_every": None,
                   "placement": "local"},
        },
    }


def run_model(model: str, c: dict, seeds: list[int], iterations: int,
              build_dir: Path, results_dir: Path) -> dict:
    """Build + run all arms under one coupling model. Returns per-arm stats."""
    flags = MODELS[model]
    bdir = build_dir / model
    rdir = results_dir / model
    traces = rdir / "traces"
    for p in (bdir, traces):
        p.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    for arm_row in ARMS:
        arm = arm_row[0]
        d = arm_derived(c, arm_row)
        sc = build_scenario(c, arm_row, flags, seeds, iterations)
        spath = bdir / f"exp1cl_{model}_{arm}.yaml"
        spath.write_text(yaml.safe_dump(sc, sort_keys=False))
        policy = build_policy(arm_row, d["checkpoint_every"], iterations,
                              spath, d["iter_wall_s"])
        ppath = bdir / f"exp1cl_{model}_{arm}_policy.json"
        ppath.write_text(json.dumps(policy, indent=1))
        cmd = [sys.executable, "run_scenario.py", "--scenario", str(spath),
               "--policy", str(ppath), "--trace-dir", str(traces),
               "--out", str(rdir / f"scenario_exp1cl_{model}_{arm}.json"),
               "--progress-interval", "60"]
        proc = subprocess.run(cmd, cwd=SIM, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:], file=sys.stderr)
            raise SystemExit(f"{model}/{arm} failed")
        meta = {"tokens_per_iter": int(c["batch_calibrated"]) * SEQ_LEN,
                "warmup_iters": WARMUP,
                "arms": {arm: {"fabric": arm_row[2] or "none"}}}
        mean_dt: dict[str, list[float]] = {}
        stripes: list[float] = []
        for seed in seeds:
            trace = traces / f"exp1cl_{model}_{arm}_{arm}_s{seed}.jsonl.gz"
            run_json = exp1_emit.emit(trace, rdir / arm / f"seed{seed}",
                                      arm=arm, seed=seed, meta=meta)
            for job, js in run_json["jobs"].items():
                mean_dt.setdefault(job, []).append(js["mean_dt_s"])
            ms = run_json["checkpoints"]["mean_stripe_s"]
            if ms:
                stripes.append(ms)
        out[arm] = {
            "mean_dt": {j: statistics.mean(v) for j, v in mean_dt.items()},
            "stripe_s": statistics.mean(stripes) if stripes else None,
            "derived": d,
        }
        print(f"  {model} {arm:<12} dtA={out[arm]['mean_dt']['A']:.6f} "
              f"dtB={out[arm]['mean_dt']['B']:.6f} "
              f"stripe={out[arm]['stripe_s']}", flush=True)
    return out


def alpha_gamma(res: dict) -> dict:
    """Per-arm alpha/gamma, each against ITS OWN block baseline, same model:
    alpha vs the arm's `baseline` (A's train-fabric block), gamma vs
    GAMMA_BASELINE (B's fabric block). They differ only for dj_flip."""
    rows = {}
    for arm_row in ARMS:
        arm, _tf, _cf, _cad, _sh, baseline, _note = arm_row
        a_base = res[baseline] if baseline else res[arm]
        g_name = GAMMA_BASELINE[arm]
        g_base = res[g_name] if g_name else res[arm]
        a = res[arm]["mean_dt"]["A"] / a_base["mean_dt"]["A"] - 1.0
        g = res[arm]["mean_dt"]["B"] / g_base["mean_dt"]["B"] - 1.0
        rows[arm] = {"alpha_pp": round(100 * a, 4), "gamma_pp": round(100 * g, 4),
                     "iteration_time_s[A]": round(res[arm]["mean_dt"]["A"], 6),
                     "iteration_time_s[B]": round(res[arm]["mean_dt"]["B"], 6),
                     "stripe_dur_s": (round(res[arm]["stripe_s"], 4)
                                      if res[arm]["stripe_s"] else None)}
    return rows


def git_info() -> dict:
    def g(*args):
        return subprocess.run(["git", *args], cwd=SIM, capture_output=True,
                              text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"),
            "commit_short": g("rev-parse", "--short", "HEAD"),
            "commit_subject": g("log", "-1", "--format=%s"),
            "dirty": bool(g("status", "--porcelain"))}


def emit_seal(c: dict, prov: dict, ag: dict, res: dict, seeds: list[int],
              iterations: int, out_path: Path, *, selftest: bool,
              stamp: bool) -> dict:
    capture_s = STATE_GB / RANKS_A / float(c["capture_d2h_gbps"])
    predictions = {}
    for arm_row in ARMS:
        arm, train_f, ckpt_f, cadence, shape_gbit, baseline, note = arm_row
        d = res["M0"][arm]["derived"]
        entry = {
            "set": "held-out (campaign not yet run; sealed pre-campaign)",
            "train_fabric": train_f, "ckpt_fabric": ckpt_f or "none",
            "b_fabric": d["b_fabric"],
            "cadence_s": cadence, "shape_gbit": shape_gbit,
            "stream_cap_gbps": d["stream_cap_gbps"],
            "checkpoint_every": d["checkpoint_every"],
            "iter_wall_s_analytic": round(d["iter_wall_s"], 6),
            "b_iter_wall_s_analytic": round(d["b_iter_wall_s"], 6),
            "b_busbw_gbps": d["b_busbw_gbps"],
            "alpha_baseline": baseline or "self",
            "gamma_baseline": GAMMA_BASELINE[arm] or "self",
            "capture_dur_s": (round(capture_s, 6) if ckpt_f else None),
            "m2_b_oversubscribed": d["m2_b_oversubscribed"],
            "note": note,
        }
        for m in MODELS:
            entry[m] = ag[m][arm]
        entry["delta_alpha_pp"] = {
            "M1_minus_M0": round(ag["M1"][arm]["alpha_pp"]
                                 - ag["M0"][arm]["alpha_pp"], 4),
            "M2_minus_M0": round(ag["M2"][arm]["alpha_pp"]
                                 - ag["M0"][arm]["alpha_pp"], 4),
            "M2_minus_M1": round(ag["M2"][arm]["alpha_pp"]
                                 - ag["M1"][arm]["alpha_pp"], 4),
        }
        entry["delta_gamma_pp"] = {
            "M1_minus_M0": round(ag["M1"][arm]["gamma_pp"]
                                 - ag["M0"][arm]["gamma_pp"], 4),
            "M2_minus_M0": round(ag["M2"][arm]["gamma_pp"]
                                 - ag["M0"][arm]["gamma_pp"], 4),
            "M2_minus_M1": round(ag["M2"][arm]["gamma_pp"]
                                 - ag["M1"][arm]["gamma_pp"], 4),
        }
        # CL-017: what each model's alpha/gamma loses when the capture takes
        # no GPU lock (should be ~ -capture floor for alpha in ckpt arms, ~0
        # for gamma — B never captures).
        entry["delta_capture_alpha_pp"] = {
            f"{m}c_minus_{m}": round(ag[m + "c"][arm]["alpha_pp"]
                                     - ag[m][arm]["alpha_pp"], 4)
            for m in BLOCKING}
        entry["delta_capture_gamma_pp"] = {
            f"{m}c_minus_{m}": round(ag[m + "c"][arm]["gamma_pp"]
                                     - ag[m][arm]["gamma_pp"], 4)
            for m in BLOCKING}
        predictions[arm] = entry
    gi = git_info()
    seal = {
        "schema": "sealed_predictions/1",
        "title": "Sealed pre-campaign M0/M1/M2 predictions (each under BOTH "
                 "capture models: blocking + 'c' nonblocking, CL-017) for "
                 "the exp1cl contention campaign, 2x2 design (arms 1-9 + "
                 "optional dj_flip; CloudLab r7525, 2026-08-05)",
        "selftest_dummy_constants": bool(selftest),
        "created_utc": (datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ") if stamp else None),
        "protocol": "control-plane/design/sim_validation_protocol.md "
                    "(section 3.3 seal); plan §4 discipline: sealed after "
                    "calibration, before any interference arm runs",
        "sim": gi,
        "models": {
            "M0": {"id": "count_based", "flag": "absent (default)",
                   "coupling": "any in-flight transfer on one of J's workers "
                               "multiplies J's collective by (1 + tasks), "
                               "whichever wire it rides"},
            "M1": {"id": "fabric_aware", "flag": "model.fabric_aware_coupling: "
                   "true (CL-014)",
                   "coupling": "a transfer counts iff it rides the fabric J's "
                               "collective rides; magnitude still 1 + tasks"},
            "M2": {"id": "intensity", "flag": "model.intensity_coupling: true "
                   "(CL-016 = CL-012.D S3; implies fabric scoping)",
                   "coupling": "factor = (busbw + in-flight offered rate) / "
                               "nameplate capacity when that exceeds 1, else "
                               "exactly 1.0; zero free parameters"},
            "M0c": {"id": "count_based_nonblocking_capture",
                    "flag": "model.nonblocking_capture: true (CL-017)",
                    "coupling": "M0's count coupling; the D2H capture takes "
                                "no GPU lock — the per-cadence capture floor "
                                "is absent (capture duration still gates the "
                                "persist pipeline)"},
            "M1c": {"id": "fabric_aware_nonblocking_capture",
                    "flag": "model.fabric_aware_coupling + "
                            "model.nonblocking_capture (CL-014 + CL-017)",
                    "coupling": "M1's fabric-scoped count coupling, no "
                                "capture floor"},
            "M2c": {"id": "intensity_nonblocking_capture",
                    "flag": "model.intensity_coupling + "
                            "model.nonblocking_capture (CL-016 + CL-017)",
                    "coupling": "M2's intensity coupling, no capture floor — "
                                "its '~0' predictions are literal ~0"},
        },
        "run_config": {
            "arms": len(ARMS), "seeds": list(seeds),
            "iterations": iterations, "warmup_iters": WARMUP,
            "state_gb": STATE_GB, "shard_gb_per_rank": STATE_GB / RANKS_A,
            "gradient_gb_per_rank": GRADIENT_GB,
            "ranks": {"A": RANKS_A, "B": RANKS_B},
            "capture_dur_s": round(capture_s, 6),
            "capture_models": "each coupling model is sealed under BOTH "
                              "capture semantics (CL-017): blocking (default "
                              "flag state; capture holds the GPU lock -> "
                              "per-arm capture floor = capture_dur_s once "
                              "per cadence) and nonblocking 'c' columns "
                              "(model.nonblocking_capture: true; no floor)",
            "note": "2x2 design (Sam 08-04): A and B are both 2-node DDP; "
                    "job C removed. Shard split 6.2 GB x 2 ranks mirrors "
                    "exp1's disclosed stripe-topology gap (hardware: rank 0 "
                    "streams the full 12.4 GB); total bytes per wave match, "
                    "wave duration is ~half the hardware's single stream. "
                    "B never checkpoints; gamma has no capture floor.",
        },
        "constants": {k: c[k] for k in sorted(c)},
        "constants_provenance": {k: prov.get(k, "declared") for k in sorted(c)},
        "held_out_data_access": {
            "statement": "THESE PREDICTIONS WERE PRODUCED WITHOUT ANY ACCESS "
                         "TO HELD-OUT HARDWARE DATA. Inputs are the on-site "
                         "calibration constants only (the protocol's Cal "
                         "block); no interference-arm hardware file, summary "
                         "or statistic existed when they were generated.",
            "hardware_read": [],
            "hardware_not_read": [f"({a[0]}, exp1cl)" for a in ARMS],
            "unseal_budget": "this file is a seal, not an unseal",
        },
        "how_to_score": "checkpointing/realgpu_coefficient/compare.py "
                        "--verdicts against results/exp1cl/ after the rsync; "
                        "tau per protocol section 4.2.",
        "predictions": predictions,
        "shape_claims": SHAPE_CLAIMS,
    }
    body = {k: seal[k] for k in ("constants", "run_config", "predictions",
                                 "shape_claims", "models")}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()
    seal["sha256"] = {
        "scope": "canonical JSON (sort_keys) of {constants, run_config, "
                 "predictions, shape_claims, models}",
        "value": digest,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(seal, indent=1))
    (out_path.with_suffix(out_path.suffix + ".sha256")).write_text(
        digest + "\n")
    print(f"\nwrote {out_path}\nsha256 {digest}")
    return seal


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def selftest_checks(c: dict, ag: dict, res: dict) -> list[dict]:
    """The plan-§4 alpha shapes (unchanged from the pre-08-04 generator) plus
    the NEW gamma shapes (G.1-G.3, replacing the old G.a whose 'gamma
    structurally 0' claim the 2x2 redesign deliberately retired; G.a's M2
    content survives inside G.3) plus the CL-017 capture-column shapes
    (C.1-C.3: floor gone on the 'c' columns, orderings preserved), on dummy
    constants. The 13 pre-CL-017 checks read ONLY the blocking columns, whose
    values are byte-consistent with the pre-CL-017 generator (the flag
    defaults OFF on the identical code path)."""
    capture_s = STATE_GB / RANKS_A / float(c["capture_d2h_gbps"])

    def floor_pp(block_off_arm: str) -> float:
        d = res["M0"][("sat25" if block_off_arm == "off" else "sh100")]["derived"]
        return 100.0 * capture_s / (d["checkpoint_every"] * d["iter_wall_s"])

    f25, f100 = floor_pp("off"), floor_pp("off100")
    a = {m: {arm: ag[m][arm]["alpha_pp"] for arm in ag[m]} for m in ag}
    g = {m: {arm: ag[m][arm]["gamma_pp"] for arm in ag[m]} for m in ag}
    dt = {m: {arm: res[m][arm]["mean_dt"] for arm in res[m]} for m in res}
    checks: list[dict] = []

    def add(cid, claim, ok, detail):
        checks.append({"id": cid, "claim": claim, "pass": bool(ok),
                       "detail": detail})

    # M0: alpha(sh100) == alpha(sat25) — load-blind. Exact equality cannot
    # hold end-to-end (see module docstring); assert the load-blind SIGNATURE
    # and report the literal difference.
    add("M0.a", "M0 charges the SATURATED shared link (alpha(sat25) well "
        "above the capture floor)",
        a["M0"]["sat25"] >= f25 + 0.8,
        f"alpha={a['M0']['sat25']:.3f}pp floor={f25:.3f}pp")
    add("M0.b", "M0 charges the UNSATURATED shared link the same way "
        "(alpha(sh100) well above the capture floor) — load-blind",
        a["M0"]["sh100"] >= f100 + 0.5,
        f"alpha={a['M0']['sh100']:.3f}pp floor={f100:.3f}pp")
    ratio = (a["M0"]["sh100"] / a["M0"]["sat25"]
             if a["M0"]["sat25"] else float("inf"))
    add("M0.c", "M0 alpha(sh100) same order as alpha(sat25) [plan: "
        "'identical'; exact equality is confounded by each block's "
        "ar_base/iter-wall ratio — reported, not asserted]",
        0.35 <= ratio <= 2.86,
        f"sh100={a['M0']['sh100']:.3f}pp sat25={a['M0']['sat25']:.3f}pp "
        f"ratio={ratio:.3f} literal_diff={a['M0']['sh100']-a['M0']['sat25']:+.3f}pp")
    # M1: dj 0-ish; shared arms identical to M0.
    add("M1.a", "M1 alpha(dj_t100_c25) == 0-ish (capture floor only; the "
        "stripe rides a different card)",
        a["M1"]["dj_t100_c25"] <= f100 + 0.5,
        f"alpha={a['M1']['dj_t100_c25']:.3f}pp floor={f100:.3f}pp")
    same_sh = abs(dt["M1"]["sh100"]["A"] - dt["M0"]["sh100"]["A"]) <= 1e-6
    same_sat = abs(dt["M1"]["sat25"]["A"] - dt["M0"]["sat25"]["A"]) <= 1e-6
    add("M1.b", "M1 == M0 exactly on shared-fabric arms (fabric scoping "
        "changes nothing where stripe and collective share the wire) — so "
        "M1's sh100-vs-sat25 shape is M0's",
        same_sh and same_sat,
        f"dtA sh100 {dt['M1']['sh100']['A']:.9f} vs {dt['M0']['sh100']['A']:.9f}; "
        f"sat25 {dt['M1']['sat25']['A']:.9f} vs {dt['M0']['sat25']['A']:.9f}")
    # M2: sh100 ~ 0, sat25 > 0.
    add("M2.a", "M2 alpha(sh100) ~ 0 coupling (== capture floor; unsaturated "
        "link charges nothing beyond dj's floor)",
        (a["M2"]["sh100"] <= f100 + 0.5
         and abs(a["M2"]["sh100"] - a["M2"]["dj_t100_c25"]) <= 0.1),
        f"sh100={a['M2']['sh100']:.3f}pp dj={a['M2']['dj_t100_c25']:.3f}pp "
        f"floor={f100:.3f}pp")
    add("M2.b", "M2 alpha(sat25) > 0 while sh100 ~ 0 (the discriminating "
        "signature)",
        a["M2"]["sat25"] - a["M2"]["sh100"] >= 1.0,
        f"sat25={a['M2']['sat25']:.3f}pp sh100={a['M2']['sh100']:.3f}pp "
        f"gap={a['M2']['sat25']-a['M2']['sh100']:.3f}pp")
    # tc10 -> sat25 margin is 0.05 pp (was 0.2 in the pre-08-04 generator):
    # the old sat25 wave was bound by the SINGLE donor's 25 Gb ingress
    # (3.97 s); the 2x2 design gives B two donor nodes, the sat25 wave
    # un-binds to 2.48 s, and the uncapped point's (factor-1) x duration
    # product nearly cancels against tc10's at the dummy rates. The plan's
    # claim is the ORDERING (rising with cap), which still holds strictly.
    add("M2.c", "M2 alpha rising with stream cap (tc5 < tc10 < sat25; "
        "tc10-vs-sat25 nearly cancel — see margin note)",
        (a["M2"]["sat25_tc5"] + 0.2 <= a["M2"]["sat25_tc10"]
         and a["M2"]["sat25_tc10"] + 0.05 <= a["M2"]["sat25"]),
        f"tc5={a['M2']['sat25_tc5']:.3f} tc10={a['M2']['sat25_tc10']:.3f} "
        f"sat25={a['M2']['sat25']:.3f}pp")
    add("M2.d", "M2 alpha rising with duty (d15 >> sat25)",
        a["M2"]["sat25_d15"] >= a["M2"]["sat25"] + 2.0,
        f"d15={a['M2']['sat25_d15']:.3f} sat25={a['M2']['sat25']:.3f}pp")
    add("M2.e", "M1/M2 predict sameCard == dj for ALPHA (the sim cannot "
        "price two-ports-one-card; that IS its sealed prediction)",
        (abs(a["M2"]["sameCard"] - a["M2"]["dj_t100_c25"]) <= 0.1
         and abs(a["M1"]["sameCard"] - a["M1"]["dj_t100_c25"]) <= 0.1),
        f"M2 sameCard={a['M2']['sameCard']:.3f} dj={a['M2']['dj_t100_c25']:.3f}pp")
    # ---- gamma shapes (NEW, 08-04 2x2 redesign) ---------------------------
    ckpt_arms = [r[0] for r in ARMS if r[2] is not None]
    # G.1: M0 is fabric-blind on the receiver side too — a stripe in flight
    # on B's nodes charges B's collective whichever wire it rides.
    add("G.1", "M0 gamma(sh100) > 0 (count-based charges B whenever a "
        "transfer is in flight on its nodes — fabric-blind)",
        g["M0"]["sh100"] >= 0.2,
        "M0 gamma pp: " + " ".join(f"{arm}={g['M0'][arm]:.3f}"
                                   for arm in ckpt_arms))
    # G.2: M1 scopes by fabric — zero wherever the stripe rides a different
    # wire than B's collective; nonzero where they share (sameCard: ckpt
    # eth100p1 == B's eth100p1).
    m1_zero_arms = [arm for arm in ckpt_arms if arm != "sameCard"]
    m1_zeros_ok = all(abs(g["M1"][arm]) <= 1e-6 for arm in m1_zero_arms)
    add("G.2", "M1 gamma == 0 in every arm whose ckpt fabric differs from "
        "B's fabric; > 0 where they share (sameCard: ckpt p3 = B p3)",
        m1_zeros_ok and g["M1"]["sameCard"] >= 0.2,
        f"sameCard={g['M1']['sameCard']:.3f}pp; others max |gamma| = "
        f"{max(abs(g['M1'][arm]) for arm in m1_zero_arms):.6f}pp")
    # G.3: M2 charges B only if stripe demand + B's collective busbw exceed
    # B's nameplate link. Computed from the same derived constants the arms
    # use — with the dummies NO arm over-subscribes (sameCard: 8.0 + 2.5 =
    # 10.5 of 12.5 GB/s), so gamma_M2 must be EXACTLY 0 everywhere (B never
    # captures -> no floor). Subsumes the old G.a under M2. DISCRIMINATING
    # CLAIM (sealed): hardware gamma > 0 in sameCard would refute M2's
    # clean-path assumption.
    oversub = [arm for arm in ckpt_arms
               if res["M2"][arm]["derived"]["m2_b_oversubscribed"]]
    if not oversub:
        gmax2 = max(abs(g["M2"][arm]) for arm in g["M2"])
        add("G.3", "M2 gamma == 0 in EVERY arm (no arm's stripe + B-collective "
            "over-subscribes B's link with these constants; capture-floor-free "
            "exact zero. Hardware gamma > 0 in sameCard would refute M2's "
            "clean-path assumption)",
            gmax2 <= 1e-9,
            f"max |gamma_M2| = {gmax2} (oversubscribed arms: none; sameCard "
            f"offers {res['M2']['sameCard']['derived']['b_busbw_gbps']} + "
            f"{res['M2']['sameCard']['derived']['stream_cap_gbps']} of "
            f"{res['M2']['sameCard']['derived']['b_nameplate_gbps']} GB/s)")
    else:
        zero_arms = [arm for arm in g["M2"] if arm not in oversub]
        add("G.3", "M2 gamma > 0 exactly in the over-subscribed arms "
            f"({oversub}) and 0 elsewhere",
            (all(g["M2"][arm] >= 0.2 for arm in oversub)
             and max(abs(g["M2"][arm]) for arm in zero_arms) <= 1e-6),
            "M2 gamma pp: " + " ".join(f"{arm}={g['M2'][arm]:.3f}"
                                       for arm in ckpt_arms))
    # ---- capture columns (CL-017 'c' variants; NEW) -----------------------
    # The seal carries BOTH capture models. exp1 evidence says capture
    # overlaps (413 ms capture vs 485 ms iteration, p99 stretch 1.002); if
    # the hardware's dj/off-adjacent arms show alpha at the capture floor
    # instead of ~0, the blocking model wins and CL-017 is refuted.
    add("C.1", "M2c alpha(dj_t100_c25) ~ 0 — the capture floor is GONE when "
        "the capture takes no GPU lock (blocking M2 kept dj AT the floor)",
        abs(a["M2c"]["dj_t100_c25"]) <= 0.25,
        f"M2c={a['M2c']['dj_t100_c25']:.3f}pp vs M2="
        f"{a['M2']['dj_t100_c25']:.3f}pp (floor={f100:.3f}pp)")
    drop = a["M2"]["sat25"] - a["M2c"]["sat25"]
    add("C.2", "M2c alpha(sat25) < M2 alpha(sat25) by ~ the capture floor "
        "(only the stall goes; the coupling term above the floor survives)",
        drop > 0 and abs(drop - f25) <= 0.5,
        f"M2={a['M2']['sat25']:.3f}pp M2c={a['M2c']['sat25']:.3f}pp "
        f"drop={drop:.3f}pp floor={f25:.3f}pp")
    m0c_ratio = (a["M0c"]["sh100"] / a["M0c"]["sat25"]
                 if a["M0c"]["sat25"] else float("inf"))
    m0c_ok = (0.35 <= m0c_ratio <= 2.86
              and a["M0c"]["sat25_tc5"] > a["M0c"]["sat25_tc10"]
              > a["M0c"]["sat25"] > 0)
    m1c_ok = (all(abs(g["M1c"][arm]) <= 1e-6 for arm in m1_zero_arms)
              and g["M1c"]["sameCard"] >= 0.2)
    m2c_ok = (a["M2c"]["sat25_tc5"] < a["M2c"]["sat25_tc10"]
              < a["M2c"]["sat25"])
    add("C.3", "every 'c' column preserves its model's ORDERING signature: "
        "M0c load-blind + duration ladder (sh100 same order as sat25; tc5 > "
        "tc10 > sat25 > 0); M1c gamma sameCard-only; M2c alpha rising with "
        "cap (tc5 < tc10 < sat25)",
        m0c_ok and m1c_ok and m2c_ok,
        f"M0c ratio={m0c_ratio:.3f} ladder=({a['M0c']['sat25_tc5']:.3f}, "
        f"{a['M0c']['sat25_tc10']:.3f}, {a['M0c']['sat25']:.3f}); M1c "
        f"sameCard={g['M1c']['sameCard']:.3f} others_max="
        f"{max(abs(g['M1c'][arm]) for arm in m1_zero_arms):.6f}; M2c "
        f"ladder=({a['M2c']['sat25_tc5']:.3f}, {a['M2c']['sat25_tc10']:.3f}, "
        f"{a['M2c']['sat25']:.3f})")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constants", type=Path,
                    help="on-site constants yaml (exp1cl_scenario.yaml with "
                         "cluster.measured filled, or a flat constants yaml)")
    ap.add_argument("--selftest", action="store_true",
                    help="run everything on DUMMY constants and assert the "
                         "plan-§4 alpha shapes + the 08-04 gamma shapes + "
                         "the CL-017 capture shapes")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--iterations", type=int, default=None,
                    help="total iterations per cell (default 300 measured + "
                         "30 warmup)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--single-wire", action="store_true",
                    help="2026-08-05 CloudLab reality: one 25 Gb link; swap in "
                         "ARMS_SW/B_FABRIC_SW/GAMMA_BASELINE_SW")
    ap.add_argument("--stamp", action="store_true",
                    help="fill created_utc now (use at the 11:40 seal)")
    args = ap.parse_args()
    if args.single_wire:
        global ARMS, B_FABRIC, GAMMA_BASELINE
        ARMS = ARMS_SW; B_FABRIC = B_FABRIC_SW; GAMMA_BASELINE = GAMMA_BASELINE_SW

    if args.selftest:
        c, prov = dict(DUMMY_CONSTANTS), {
            k: "DUMMY (selftest only, not measured)" for k in DUMMY_CONSTANTS}
        seeds = args.seeds or [0]
        build_dir = BUILD / "selftest"
        results_dir = SIM / "results" / "exp1cl_predict_selftest"
        out = args.out or build_dir / "sealed_predictions_exp1cl_SELFTEST.json"
    else:
        if not args.constants:
            raise SystemExit("need --constants <yaml> (or --selftest)")
        c, prov = load_constants(args.constants)
        seeds = args.seeds or [0, 1]
        build_dir = BUILD
        results_dir = SIM / "results" / "exp1cl_predict"
        out = args.out or BUILD / "sealed_predictions_exp1cl.json"
    iterations = args.iterations or (ITER_MEASURED + WARMUP)

    res = {}
    for model in MODELS:
        print(f"=== model {model} ({MODELS[model] or 'flags off'}) ===",
              flush=True)
        res[model] = run_model(model, c, seeds, iterations, build_dir,
                               results_dir)
    ag = {m: alpha_gamma(res[m]) for m in MODELS}

    cols = list(MODELS)          # M0 M0c M1 M1c M2 M2c
    hdr = (f"{'arm':<12} " + " ".join(f"{m:>8}" for m in cols)
           + f" {'stripe s':>9}")
    for metric, label in (("alpha_pp", "alpha"), ("gamma_pp", "gamma")):
        print(f"\n=== {label} (pp, block baselines, seeds {seeds}; "
              "'c' = nonblocking capture, CL-017) ===")
        print(hdr)
        print("-" * len(hdr))
        for arm_row in ARMS:
            arm = arm_row[0]
            st = res["M2"][arm]["stripe_s"]
            print(f"{arm:<12} "
                  + " ".join(f"{ag[m][arm][metric]:>8.3f}" for m in cols)
                  + f" {st if st is not None else '-':>9}")

    emit_seal(c, prov, ag, res, seeds, iterations, out,
              selftest=args.selftest, stamp=args.stamp)

    if args.selftest:
        checks = selftest_checks(c, ag, res)
        print("\n=== SELFTEST (plan §4 alpha shapes + 08-04 gamma shapes + "
              "CL-017 capture shapes, dummy constants) ===")
        failed = 0
        for ch in checks:
            mark = "PASS" if ch["pass"] else "FAIL"
            failed += 0 if ch["pass"] else 1
            print(f"[{mark}] {ch['id']}: {ch['claim']}\n       {ch['detail']}")
        print(f"\n{len(checks) - failed}/{len(checks)} assertions passed")
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
