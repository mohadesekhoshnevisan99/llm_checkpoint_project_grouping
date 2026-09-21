"""Pencil-math unit tests for the parallelism-aware checkpoint calculator
(checkpointing/parallelism.py). Every expected number is hand-derived from the
16 B/param rule (2 bf16 params + 2 bf16 grads + 4 fp32 master + 4 m + 4 v).

Mark meeting 2026-07-23 D2: the calculator replaces the magic per-node shard.
"""
import math

import pytest

from checkpointing import parallelism as P


def approx(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def test_bytes_per_param_constant():
    # 2 + 2 + 4 + 4 + 4 = 16
    assert P.BYTES_PER_PARAM == 16
    assert P.BYTES_OPTIMIZER == 12          # fp32 master + m + v
    assert P.BYTES_PARAMS_GRADS == 4        # bf16 params + bf16 grads
    assert P.BYTES_OPTIMIZER + P.BYTES_PARAMS_GRADS == P.BYTES_PER_PARAM


# --- config A: mega 405B TP8 PP16 DP16 zero1 ------------------------------- #
# tp*pp = 128 -> partition = 405/128 = 3.1640625 B params
# zero1 bytes/param: unique = 12/16 = 0.75, replicated = 4  -> shard 4.75 B/param
def test_config_A_mega_zero1():
    m = P.parse_model(dict(name="405B", params_b=405, tp=8, pp=16, dp=16,
                           zero_stage=1))
    r = P.shard_report(m)
    assert approx(r["partition_params_b"], 405 / 128)
    assert approx(r["per_gpu_shard_gb"], 4.75 * (405 / 128))     # 15.029296875
    assert approx(r["per_gpu_shard_gb"], 15.029296875)
    assert approx(r["unique_gb"], 0.75 * (405 / 128))
    assert approx(r["replicated_gb"], 4.0 * (405 / 128))
    # dedup collapses to 16/dp = 1.0 B/param  -> partition size in GB
    assert approx(r["effective_shard_gb"], 405 / 128)
    assert approx(P.effective_shard_gb(m, dedup=True), 405 / 128)
    assert approx(P.effective_shard_gb(m, dedup=False), 15.029296875)
    # whole-job dedup need is 16*P regardless of stage
    assert approx(r["checkpoint_need_gb"], 16 * 405)
    assert r["hbm_ok"] and r["per_gpu_state_gb"] <= 80


# --- config B: be 3B TP1 PP1 DP16 zero0 ------------------------------------ #
# tp*pp = 1 -> partition = 3 B; zero0 -> shard 16 B/param -> 48 GB, all replicated
def test_config_B_be_zero0():
    m = P.parse_model(dict(name="3B", params_b=3, tp=1, pp=1, dp=16,
                           zero_stage=0))
    r = P.shard_report(m)
    assert approx(r["per_gpu_shard_gb"], 48.0)
    assert approx(r["unique_gb"], 0.0)
    assert approx(r["replicated_gb"], 48.0)
    assert approx(r["replicated_frac"], 1.0)
    # dedup on a fully-replicated (zero0) class cuts by exactly dp=16 -> 3.0 GB
    assert approx(P.effective_shard_gb(m, dedup=True), 3.0)
    assert approx(48.0 / P.effective_shard_gb(m, dedup=True), 16.0)   # /dp
    assert r["hbm_ok"]


# --- config C: frontier 30B TP4 PP2 DP16 zero0 ----------------------------- #
def test_config_C_frontier_zero0():
    m = P.parse_model(dict(name="30B", params_b=30, tp=4, pp=2, dp=16,
                           zero_stage=0))
    r = P.shard_report(m)
    assert approx(r["partition_params_b"], 30 / 8)
    assert approx(r["per_gpu_shard_gb"], 16 * (30 / 8))     # 60.0
    assert approx(r["per_gpu_shard_gb"], 60.0)
    assert approx(P.effective_shard_gb(m, dedup=True), 30 / 8)      # 3.75
    assert r["per_gpu_state_gb"] <= 80 and r["hbm_ok"]


# --- config D: standard 13B TP2 PP2 DP16 zero0 ----------------------------- #
def test_config_D_standard_zero0():
    m = P.parse_model(dict(name="13B", params_b=13, tp=2, pp=2, dp=16,
                           zero_stage=0))
    r = P.shard_report(m)
    assert approx(r["per_gpu_shard_gb"], 16 * (13 / 4))     # 52.0
    assert approx(r["effective_shard_gb"], 13 / 4)          # 3.25
    assert r["hbm_ok"]


# --- config E: fxl 70B TP8 PP4 DP16 zero3 ---------------------------------- #
# zero3: everything sharded -> 16/dp = 1.0 B/param, all UNIQUE, dedup is a no-op
def test_config_E_fxl_zero3_dedup_noop():
    m = P.parse_model(dict(name="70B", params_b=70, tp=8, pp=4, dp=16,
                           zero_stage=3))
    r = P.shard_report(m)
    assert approx(r["partition_params_b"], 70 / 32)
    assert approx(r["per_gpu_shard_gb"], 1.0 * (70 / 32))   # 2.1875
    assert approx(r["unique_gb"], 70 / 32)
    assert approx(r["replicated_gb"], 0.0)
    # at stage 3 dedup == no-dedup (nothing replicated to slice)
    assert approx(P.effective_shard_gb(m, dedup=True),
                  P.effective_shard_gb(m, dedup=False))


# --- config F: zero2 is treated as zero1 for persisted-state math ---------- #
def test_config_F_zero2_treated_as_zero1():
    m2 = P.parse_model(dict(params_b=8, tp=1, pp=1, dp=8, zero_stage=2))
    m1 = P.parse_model(dict(params_b=8, tp=1, pp=1, dp=8, zero_stage=1))
    assert m2.zero_stage == 1
    assert approx(P.per_gpu_shard_gb(m2), P.per_gpu_shard_gb(m1))
    # zero1, dp=8, partition=8: unique 12/8=1.5, repl 4 -> shard 5.5*8 = 44 GB
    assert approx(P.per_gpu_shard_gb(m2), 5.5 * 8)
    assert approx(P.effective_shard_gb(m2, dedup=True), (16 / 8) * 8)   # 16 GB


def test_dedup_identity_effective_is_16_over_dp_all_stages():
    """The key algebraic fact the DP-aware contribution rests on: with dedup,
    per-GPU effective shard = 16/dp B/param for stage 0, 1 AND 3."""
    for stage in (0, 1, 3):
        m = P.parse_model(dict(params_b=40, tp=2, pp=4, dp=16, zero_stage=stage))
        eff = P.effective_shard_gb(m, dedup=True)
        expect = (16 / 16) * m.partition_params_b
        assert approx(eff, expect), (stage, eff, expect)


def test_dedup_savings_direction_zero0_gt_zero1():
    """Pre-registered expectation: dedup cuts zero0 bytes by ~dp; less on zero1
    (only the 4 B/param params are replicated, optimizer already sharded)."""
    dp = 16
    z0 = P.parse_model(dict(params_b=13, tp=2, pp=2, dp=dp, zero_stage=0))
    z1 = P.parse_model(dict(params_b=13, tp=2, pp=2, dp=dp, zero_stage=1))
    cut0 = P.effective_shard_gb(z0, False) / P.effective_shard_gb(z0, True)
    cut1 = P.effective_shard_gb(z1, False) / P.effective_shard_gb(z1, True)
    assert approx(cut0, dp)                       # zero0: exactly dp
    assert approx(cut1, (dp + 3) / 4)             # zero1: (4dp+12)/16 = (dp+3)/4
    assert cut0 > cut1 > 1.0


def test_checkpoint_need_invariant_across_stages():
    for stage in (0, 1, 3):
        m = P.parse_model(dict(params_b=70, tp=8, pp=4, dp=16, zero_stage=stage))
        assert approx(P.shard_report(m)["checkpoint_need_gb"], 16 * 70)


def test_jit_dram_recoverable_gate():
    z0 = P.parse_model(dict(params_b=7, tp=1, pp=1, dp=16, zero_stage=0))
    z1 = P.parse_model(dict(params_b=7, tp=1, pp=1, dp=16, zero_stage=1))
    z3 = P.parse_model(dict(params_b=7, tp=1, pp=1, dp=16, zero_stage=3))
    dp1 = P.parse_model(dict(params_b=7, tp=1, pp=1, dp=1, zero_stage=0))
    assert P.jit_dram_recoverable(z0) is True     # full replica on a DP peer
    assert P.jit_dram_recoverable(z1) is False    # optimizer slice lost
    assert P.jit_dram_recoverable(z3) is False    # everything sharded
    assert P.jit_dram_recoverable(dp1) is False   # no DP peer exists
    assert P.jit_dram_recoverable(None) is True    # legacy: unchanged


def test_apply_model_shards_injects_and_is_legacy_safe():
    sc = {"classes": {
        "withmodel": {"ranks": 8, "iteration_seconds": 10,
                      "model": dict(params_b=13, tp=2, pp=2, dp=16, zero_stage=0)},
        "legacy": {"ranks": 4, "checkpoint_gb_per_rank": 3.2},
    }}
    P.apply_model_shards(sc)
    # model class got its no-dedup per-GPU shard injected
    assert approx(sc["classes"]["withmodel"]["checkpoint_gb_per_rank"], 52.0)
    assert "_model_shard" in sc["classes"]["withmodel"]
    # legacy class untouched (no model: block)
    assert sc["classes"]["legacy"]["checkpoint_gb_per_rank"] == 3.2
    assert "_model_shard" not in sc["classes"]["legacy"]
    # idempotent
    P.apply_model_shards(sc)
    assert approx(sc["classes"]["withmodel"]["checkpoint_gb_per_rank"], 52.0)


def test_parse_model_validation():
    with pytest.raises(ValueError):
        P.parse_model(dict(tp=8, pp=16, dp=16, zero_stage=1))       # no params_b
    with pytest.raises(ValueError):
        P.parse_model(dict(params_b=-1, dp=1, tp=1, pp=1))
    with pytest.raises(ValueError):
        P.parse_model(dict(params_b=10, dp=0))                      # dp < 1
    with pytest.raises(ValueError):
        P.parse_model(dict(params_b=10, zero_stage=4))              # bad stage
