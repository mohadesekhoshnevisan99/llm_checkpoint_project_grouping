"""Baseline-fairness audit (Mark D2, 2026-07-23) + jit_pc_fallback (2026-07-28):
JIT (EuroSys'24) recovery is parallelism-correct. JIT drains a lost rank's state
from a surviving DP replica's DRAM, which only exists when a DP peer holds an
INTACT full copy:

  * zero_stage 0 (dp>1): full DP replication -> survivor drain works, exactly one
    iteration lost (jit_peer_dram);
  * zero_stage 1: optimizer slice sharded (unique to the failed rank) -> no peer
    has it; * zero_stage 3: everything sharded -> no intact replica;
  * legacy class (no model: block) -> original survivor-drain path, unchanged.

For the no-replica stages the paper's OWN prescription (Gupta et al. Sec. 5.2)
is to degrade to PERIODIC checkpointing at c* = sqrt(N*f/(2*o)) (eq. 3) — NOT
from-scratch recovery. jit_pc_fallback (ON by default) enacts exactly that:
periodic persists to the L3 store at the c* cadence, recovery = newest store
copy. `jit_pc_fallback: false` reproduces the archived scratch rows.

Run through the real driver so it exercises _single_node_failure's JIT branch,
the parallelism.jit_dram_recoverable gate, and the fallback wiring together.
"""
import math

import run_scenario
from checkpointing import baselines


def _base_class(**over):
    # checkpoint_gb_per_rank is the legacy field; a model: override (when given)
    # replaces it at load via parallelism.apply_model_shards.
    spec = dict(count=1, ranks=4, cohorts=4, checkpoint_gb_per_rank=1.0,
                iteration_seconds=4.0, iterations=40, checkpoint_every=2, kpeers=2,
                gradient_gb_per_rank=0.5, rpo_s=120.0)
    spec.update(over)
    return spec


def _scenario(small_spec, arm_cfg=None):
    return {
        "simulation": {"seeds": [7], "max_sim_s": 400.0, "engine": "event"},
        "cluster": {"node_count": 100, "gpu_cpu_bandwidth_gbps": 12.5,
                    "network_bandwidth_gbps": 1.85, "local_ssd_bandwidth_gbps": 0.29,
                    "object_store_bandwidth_gbps": 8.0},
        # one class only: recovery_source_tiers then reflects exactly this class's
        # JIT behavior (JIT drains a survivor from the SAME job's other cohorts,
        # so no second job is needed as a donor).
        "classes": {"small": small_spec},
        "failures": {"per_node_per_second": 5.0e-3,
                     "weights": {"process": 0.4, "node": 0.4, "spot": 0.2},
                     "restart_seconds": {"process": 5.0, "node": 10.0, "spot": 15.0}},
        "store": {"in_gbps": 500.0, "out_gbps": 500.0, "disk_gbps": 500.0,
                  "stream_gbps": 1.0},
        "controller": {"slot_period_s": 30.0},
        "arms": {"jit": arm_cfg or {"baseline": "jit"}},
    }


def _run(small_spec, arm_cfg=None):
    return run_scenario.run_arm(_scenario(small_spec, arm_cfg), "jit", 7, None)


ARCHIVED = {"baseline": "jit", "jit_pc_fallback": False}   # pre-2026-07-28 arm


def test_jit_zero0_recovers_from_dp_replica():
    """dp=8, zero0: every single-node failure drains a surviving DP replica,
    losing exactly one iteration; no total loss (no eviction here). The
    fallback (default ON) must not engage — no replica-path change, and no
    jit_pc_fallback key in the row (archived schema)."""
    row = _run(_base_class(model=dict(params_b=0.5, tp=1, pp=1, dp=8, zero_stage=0)))
    tiers = row["recovery_source_tiers"]
    assert tiers.get("jit_peer_dram", 0) > 0, tiers
    assert tiers.get("initial_state", 0) == 0, tiers        # no ZeRO-3 collapse
    assert row["lost_iters_max"] == 1                        # exactly one iter
    assert "jit_pc_fallback" not in row                      # not engaged at z0


def test_jit_zero3_collapses_to_initial_state_archived():
    """zero3 with the ARCHIVED arm (jit_pc_fallback: false): no DP peer holds an
    intact replica -> JIT falls to initial_state (reproduces archived rows)."""
    row = _run(_base_class(model=dict(params_b=0.5, tp=1, pp=1, dp=8, zero_stage=3)),
               ARCHIVED)
    tiers = row["recovery_source_tiers"]
    assert tiers.get("jit_peer_dram", 0) == 0, tiers
    assert tiers.get("initial_state", 0) > 0, tiers          # total loss each failure
    assert "jit_pc_fallback" not in row


def test_jit_zero1_cannot_recover_optimizer_archived():
    """zero1, ARCHIVED arm: the failed rank's optimizer slice is sharded (unique)
    -> no peer has it -> initial_state (JIT keeps no separate checkpoint)."""
    row = _run(_base_class(model=dict(params_b=0.5, tp=1, pp=1, dp=8, zero_stage=1)),
               ARCHIVED)
    tiers = row["recovery_source_tiers"]
    assert tiers.get("jit_peer_dram", 0) == 0, tiers
    assert tiers.get("initial_state", 0) > 0, tiers


def test_jit_legacy_class_unchanged():
    """A class with no model: block keeps the original survivor-drain behavior
    (jit_dram_recoverable(None) is True), fallback default ON notwithstanding."""
    row = _run(_base_class())        # no model: block
    tiers = row["recovery_source_tiers"]
    assert tiers.get("jit_peer_dram", 0) > 0, tiers
    assert tiers.get("initial_state", 0) == 0, tiers
    assert row["lost_iters_max"] == 1
    assert "jit_pc_fallback" not in row


# --------------------------------------------------------------------------- #
# jit_pc_fallback (2026-07-28)
# --------------------------------------------------------------------------- #
def test_jit_pc_cstar_arithmetic():
    """c* = sqrt(N*f/(2*o)) with N=ranks, f=scenario rate, o=capture stall —
    hand-checked against the mini-C mega class at zero_stage 1."""
    cl = {"gpu_cpu_bandwidth_gbps": 40.0, "network_bandwidth_gbps": 50.0}
    spec = dict(ranks=256, iteration_seconds=60.0,
                checkpoint_gb_per_rank=15.029296875,   # 405B tp8 pp16 dp16 z1
                model=dict(params_b=405, tp=8, pp=16, dp=16, zero_stage=1))
    f = 3.0e-06
    doc = baselines.jit_pc_cstar(spec, cl, f)
    o = 15.029296875 / 40.0
    cstar = math.sqrt(256 * f / (2.0 * o))
    assert doc["N_gpus"] == 256
    assert abs(doc["o_stall_s"] - o) < 1e-6
    assert abs(doc["cstar_per_s"] - cstar) < 1e-8
    assert abs(doc["interval_s"] - 1.0 / cstar) < 1e-3
    assert doc["checkpoint_every"] == max(
        1, round((1.0 / cstar) / doc["iteration_wall_s"]))


def test_jit_pc_rule_gates():
    """The fallback every_rule engages ONLY for no-replica classes under a
    positive failure rate; replicated/legacy/rate-0 keep _never (1e9)."""
    cl = {"gpu_cpu_bandwidth_gbps": 12.5, "network_bandwidth_gbps": 1.85}
    z0 = _base_class(model=dict(params_b=0.5, tp=1, pp=1, dp=8, zero_stage=0))
    z3 = _base_class(model=dict(params_b=0.5, tp=1, pp=1, dp=8, zero_stage=3))
    legacy = _base_class()
    rule = baselines.jit_pc_fallback_rule(5.0e-3)
    assert rule(z0, cl) == 10 ** 9                    # replica path untouched
    assert rule(legacy, cl) == 10 ** 9                # legacy untouched
    assert 1 <= rule(z3, cl) < 10 ** 9                # periodic cadence engaged
    assert baselines.jit_pc_fallback_rule(0.0)(z3, cl) == 10 ** 9


def test_jit_pc_fallback_recovers_from_store_deterministic():
    """Deterministic (scheduled failure, no organic injection): zero3 under the
    fallback persists periodically to the store; a node loss recovers the
    newest store copy instead of scratch. Same event with the ARCHIVED arm is
    total loss — the before/after pair for the fairness fix."""
    spec = _base_class(model=dict(params_b=0.5, tp=1, pp=1, dp=8, zero_stage=3))
    sc = _scenario(spec)
    sc["failures"]["inject_random"] = False
    sc["scheduled_failures"] = [{
        "id": "node-loss", "job_id": "small0", "cohort": 1,
        "after_iteration": 3, "failure_type": "node",
    }]
    row = run_scenario.run_arm(sc, "jit", 7, None)
    tiers = row["recovery_source_tiers"]
    assert tiers.get("store", 0) == 1, tiers          # newest store copy, not scratch
    assert tiers.get("initial_state", 0) == 0, tiers
    assert row["lost_iters_max"] <= 2
    doc = row["jit_pc_fallback"]["per_class"]["small"]
    assert doc["checkpoint_every"] >= 1
    assert doc["N_gpus"] == 4 and doc["f_per_gpu_per_s"] == 5.0e-3

    sc_off = _scenario(spec, ARCHIVED)
    sc_off["failures"]["inject_random"] = False
    sc_off["scheduled_failures"] = sc["scheduled_failures"]
    row_off = run_scenario.run_arm(sc_off, "jit", 7, None)
    assert row_off["recovery_source_tiers"] == {"initial_state": 1}


def test_jit_pc_fallback_zero0_byte_identical():
    """The hard regression gate: at zero_stage 0 (replica path) the fallback
    flag must be a NO-OP — flag ON vs flag OFF produce IDENTICAL result rows
    (modulo policy_flags, which records the requested arm cfg verbatim)."""
    spec = _base_class(model=dict(params_b=0.5, tp=1, pp=1, dp=8, zero_stage=0))
    row_on = _run(dict(spec), {"baseline": "jit"})                  # default ON
    row_off = _run(dict(spec), dict(ARCHIVED))                      # archived
    row_on.pop("policy_flags"), row_off.pop("policy_flags")
    assert row_on == row_off
