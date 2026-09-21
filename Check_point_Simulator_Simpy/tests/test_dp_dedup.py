"""DP-aware checkpointing (dp_dedup) end-to-end byte accounting + determinism
(Mark D2, 2026-07-23; goal item 1b gates). A model:-configured class under an
`dp_dedup: true` arm persists unique + replicated/dp per rank; the reduction must
flow through the real capture/flush paths (cheaper -> less stall) and be
byte-identical across identical invocations.
"""
import run_scenario


def _scenario(dedup: bool, zero_stage: int = 0):
    return {
        "simulation": {"seeds": [7], "max_sim_s": 4000.0, "engine": "event"},
        "cluster": {"node_count": 64, "gpu_cpu_bandwidth_gbps": 12.5,
                    "network_bandwidth_gbps": 20.0, "local_ssd_bandwidth_gbps": 4.0,
                    "object_store_bandwidth_gbps": 8.0},
        "classes": {
            # zero0 dp=8 -> dedup cuts the per-rank shard by 8x. Frequent
            # checkpoints (every=1) so the flush/capture saving shows in train_end.
            "m": {"count": 2, "ranks": 8, "cohorts": 8,
                  "model": dict(name="d", params_b=1.0, tp=1, pp=1, dp=8,
                                zero_stage=zero_stage),
                  "iteration_seconds": 5.0, "iterations": 20, "checkpoint_every": 1,
                  "kpeers": 3, "rpo_s": 120.0, "weights_gb_per_rank": 0.5},
        },
        "failures": {"per_node_per_second": 0.0,          # isolate checkpoint cost
                     "weights": {"process": 1.0, "node": 0.0, "spot": 0.0},
                     "restart_seconds": {"process": 5.0, "node": 20.0, "spot": 30.0}},
        "store": {"in_gbps": 200.0, "out_gbps": 200.0, "disk_gbps": 200.0,
                  "stream_gbps": 5.0, "backstop_s": 100.0},
        "controller": {"slot_period_s": 30.0},
        "arms": {"ours": {"kpeers": True, "slots": True, "dp_dedup": dedup}},
    }


def test_dedup_is_deterministic():
    """Two identical ours+dedup invocations are byte-identical (the D2 gate)."""
    a = run_scenario.run_arm(_scenario(True), "ours", 7, None)
    b = run_scenario.run_arm(_scenario(True), "ours", 7, None)
    assert a["per_class_end_s"] == b["per_class_end_s"]
    assert a["recovery_source_tiers"] == b["recovery_source_tiers"]
    assert a["rank_flushes"] == b["rank_flushes"]


def test_dedup_reduces_checkpoint_cost():
    """zero0 dp=8: dedup (unique+replicated/8) makes every flush/capture ~8x
    cheaper, so the failure-free job finishes strictly sooner than no-dedup."""
    ded = run_scenario.run_arm(_scenario(True), "ours", 7, None)
    nod = run_scenario.run_arm(_scenario(False), "ours", 7, None)
    assert ded["train_end_s"] < nod["train_end_s"], (ded["train_end_s"],
                                                     nod["train_end_s"])


def test_dedup_noop_at_zero3():
    """zero3 is fully sharded: dedup changes no bytes, so train_end matches."""
    ded = run_scenario.run_arm(_scenario(True, zero_stage=3), "ours", 7, None)
    nod = run_scenario.run_arm(_scenario(False, zero_stage=3), "ours", 7, None)
    assert ded["train_end_s"] == nod["train_end_s"]
