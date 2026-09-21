"""AUDIT regression (2026-07-25): the policy k=0 guard must route to LOCAL.

The physics audit found the kpeers==0 guard in run_scenario was dead code —
`int(k or 1)` coerces numeric 0 to 1 — and two additional bypass shapes:
JSON `"placement": null` (took the placement branch with None -> crossjob
route) and explicit `"placement": "crossjob"` with kpeers 0. All three shapes
ran the cross-job path with a zero-width stripe and a donor_drain backstop
with nothing to drain: ZERO durability. Correct behavior: any job whose
EFFECTIVE kpeers is 0 and whose placement is absent/null/crossjob takes the
local route (own-SSD fallback + owner_push store backstop), so a node loss
recovers from the STORE, never from initial_state.

These are the audit's probe4_guard.py shapes, pinned end-to-end: a scheduled
node loss after iteration 8 must recover from the store copy (@6), with zero
total loss.
"""
from __future__ import annotations

import copy

import pytest

from run_scenario import run_arm

SCENARIO = {
    "simulation": {"seeds": [7], "max_sim_s": 4000.0, "engine": "event"},
    "cluster": {
        "node_count": 10,
        "gpu_cpu_bandwidth_gbps": 10.0,
        "network_bandwidth_gbps": 2.0,
        "local_ssd_bandwidth_gbps": 1.0,
        "object_store_bandwidth_gbps": 8.0,
    },
    "classes": {
        "solo": {                      # the k=0 victim of the guard bypass
            "count": 1, "ranks": 1, "cohorts": 1,
            "checkpoint_gb_per_rank": 1.0, "iteration_seconds": 10.0,
            "iterations": 12, "checkpoint_every": 3, "kpeers": 0,
            "rpo_s": 1000.0,
        },
        "don": {                       # second job (audit probes used 2 jobs)
            "count": 1, "ranks": 1, "cohorts": 1,
            "checkpoint_gb_per_rank": 1.0, "iteration_seconds": 10.0,
            "iterations": 30, "checkpoint_every": 5, "kpeers": 1,
            "rpo_s": 1000.0,
        },
    },
    "failures": {
        "per_node_per_second": 0.0,    # only the scheduled node loss fires
        "weights": {"process": 0.4, "reboot": 0.15, "node": 0.05, "spot": 0.4},
        "restart_seconds": {"process": 5.0, "reboot": 180.0, "node": 20.0,
                            "spot": 30.0},
    },
    "scheduled_failures": [
        {"id": "solo-node-after-8", "job_id": "solo0", "cohort": 0,
         "after_iteration": 8, "failure_type": "node"},
    ],
    "store": {"in_gbps": 500.0, "out_gbps": 500.0, "disk_gbps": 500.0,
              "stream_gbps": 999.0, "backstop_s": 30.0},
    "controller": {"slot_period_s": 30.0},
    "backstop": "donor_drain",
    "arms": {"unused": {"kpeers": True}},
}


def _policy(solo_job: dict) -> dict:
    return {
        "name": "k0_guard_probe",
        "scenario": "<test>",
        "flags": {"kpeers": True},
        "jobs": {
            "solo0": {"class": "solo", "checkpoint_every": 3,
                      "capture_every": 3, "store_every": 3, "slot_s": 0.0,
                      **solo_job},
            "don0": {"class": "don", "kpeers": 1, "checkpoint_every": 5,
                     "capture_every": 5, "store_every": 5, "slot_s": 10.0},
        },
    }


@pytest.mark.parametrize("shape,solo_job", [
    ("k0_no_placement", {"kpeers": 0}),
    ("k0_null_placement", {"kpeers": 0, "placement": None}),
    ("k0_explicit_crossjob", {"kpeers": 0, "placement": "crossjob"}),
])
def test_k0_policy_routes_local_and_recovers_from_store(tmp_path, shape,
                                                        solo_job) -> None:
    sc = copy.deepcopy(SCENARIO)
    row = run_arm(sc, "unused", 7, tmp_path / f"{shape}.jsonl.gz",
                  policy=_policy(solo_job))

    # exactly the scheduled node loss fired
    assert row["failures"] == {"node": 1}

    # the guard routed solo0 LOCAL: its waves are own-SSD fallbacks (not a
    # zero-width cross-job stripe), and the owner-push store backstop fired
    assert row["fallback_flushes"] >= 3, row["fallback_flushes"]

    # node loss wipes the local SSD; recovery MUST come from the store copy
    # (the whole point of the guard) — never a total loss
    tiers = row["recovery_source_tiers"]
    assert tiers.get("store", 0) == 1, tiers
    assert "initial_state" not in tiers, tiers
    assert row["total_loss"] == 0
    # store copy @6 -> rolled back 8 - 6 = 2 iterations
    assert row["lost_iters_max"] == 2, row["lost_iters_max"]


def test_k0_explicit_local_placement_unchanged(tmp_path) -> None:
    """Control: an explicit placement: local policy (the pre-audit correct
    shape) behaves identically to the guarded shapes."""
    sc = copy.deepcopy(SCENARIO)
    row = run_arm(sc, "unused", 7, tmp_path / "control.jsonl.gz",
                  policy=_policy({"kpeers": 0, "placement": "local"}))
    assert row["recovery_source_tiers"].get("store", 0) == 1
    assert row["total_loss"] == 0
