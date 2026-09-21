"""Intensity-based checkpoint <-> all-reduce coupling (protocol CL-016,
the S3 completion of the CL-012.D pre-registration).

CL-014 made the coupling fabric-scoped but left the shared-wire magnitude a
COUNT (1 + tasks), duration-driven by construction — the half of CL-012.D it
deliberately did not implement. This change implements S3: under
`model.intensity_coupling` (default OFF, exactly parallel to
`model.fabric_aware_coupling`) a transfer sharing the collective's fabric
charges slowdown proportional to actual bandwidth contention,

    offered = collective busbw demand + in-flight transfer demand   (GB/s)
    factor  = offered / link capacity   if offered > capacity, else 1.0

with ZERO free parameters: the collective demand is the measured NCCL busbw
(`rates.nic`, the ar_base constant), the transfer demand is the streams'
offered rates (shaped tc rates / measured TCP ceiling / path NIC caps — all
declared scenario constants), and the capacity is the nameplate link rate
(min(rates.nic_in, rates.nic_out), CL-012.D S4).

What is pinned here, mirroring tests/test_fabric_coupling.py:

  1. Byte-identity: flag ABSENT => the identical pre-CL-016 code paths, the
     pencil digits of scenarios/handcheck_fabric.yaml unchanged on BOTH arms,
     and no new keys in any flag-off results row. (The committed-scenario
     byte-identity table itself is in the CL-016 protocol entry.)
  2. An UNSATURATED shared link charges NOTHING: factor exactly 1.0 even
     though the stream rides the collective's own wire.
  3. An over-offered link charges exactly offered / capacity — pencil-derived
     end-to-end and unit-level, including CL-012.G's sat1g_on_tc1 arithmetic.
  4. A separate fabric charges NOTHING regardless of rate or capacity (the
     CL-014 scoping is implied by the intensity flag, not lost by it).
  5. The demand ledger bump-ed alongside the CL-014 counters is exact: sums of
     in-flight offered rates per fabric, reversed exactly, emptied with the
     counts, and tolerant of pre-CL-016 worker mocks.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from checkpointing.base import bump_network_tasks
from run_scenario import CohortJobRuntime, run_arm

SCENARIO = (Path(__file__).resolve().parent.parent
            / "scenarios" / "handcheck_fabric.yaml")

# The pencil, from the scenario header (unchanged by this change).
BLIND_END = 15.0    # 9.000 + 2 iterations x (1.0 compute + 2 x 1.0 collective)
SCOPED_END = 13.0   # 9.000 + 2 iterations x (1.0 compute + 1 x 1.0 collective)

# Intensity pencil, over-offered case (test 3): trainer nic_in/nic_out squeezed
# to 0.8 GB/s (the declared line rate), donor stream rates raised to 0.2 GB/s
# (rates.nic AND rates.nic_in — the wave's per-stream demand cap is
# min(donor nic, donor disk), path-bounded by the donor's nic_in), and the
# checkpoint stream moved onto the collective's wire.
#   offered = rates.nic (1.0, the busbw that sets ar_base)
#           + stream offered rate (0.2)                       = 1.2 GB/s
#   factor  = offered / capacity = 1.2 / 0.8 = 1.5
# The stripe is 1.0 GB / 0.2 GB/s = 5.0 s (9.000 -> 14.000), so both post-wave
# iterations still fully overlap it, and the trainer ends at
#   9.000 + 2 x (1.0 compute + 1.0 ar_base x 1.5) = 14.000
# — chosen ON the 0.1 s grid because per_class_end_s rounds to one decimal.
OVER_OFFERED_END = 9.0 + 2 * (1.0 + 1.0 * (1.2 / 0.8))


def _load() -> dict:
    return yaml.safe_load(SCENARIO.read_text())


def _with_intensity(sc: dict) -> dict:
    """Add an `intense` arm: the scoped arm plus the CL-016 flag."""
    sc["arms"]["intense"] = {"kpeers": True, "slots": False,
                             "intensity_coupling": True}
    return sc


def _trainer_end(sc: dict, arm: str) -> float:
    return run_arm(sc, arm, 7, None)["per_class_end_s"]["trainer"]


# ------------------- 1. byte-identity: flag OFF is the old model -------------

def test_flag_absent_keeps_both_pencil_digits() -> None:
    """Neither committed arm declares intensity_coupling, so both must land on
    the CL-014 pencil to the digit — the regression contract every committed
    scenario, staircase, rack table and parallelism table rides on. (The
    committed-scenario byte-identity table — results JSON mod wall_s on
    miniC / rackpar_z1_miniC / handcheck2 / l15 / rack — is in CL-016.)"""
    sc = _load()
    assert _trainer_end(sc, "blind") == pytest.approx(BLIND_END, abs=1e-9)
    assert _trainer_end(sc, "scoped") == pytest.approx(SCOPED_END, abs=1e-9)


def test_flag_absent_adds_no_keys_to_the_results_row() -> None:
    row = run_arm(_load(), "blind", 7, None)
    assert "intensity_coupling" not in row
    row_fabric = run_arm(_load(), "scoped", 7, None)
    assert "intensity_coupling" not in row_fabric      # CL-014 flag alone
    assert row_fabric["fabric_aware_coupling"] is True  # CL-014 keys untouched


def test_flag_does_not_leak_between_arms() -> None:
    sc = _with_intensity(_load())
    sc["classes"]["trainer"]["rates"]["ckpt_fabric"] = "roce"
    assert _trainer_end(sc, "intense") == pytest.approx(SCOPED_END, abs=1e-9)
    assert _trainer_end(sc, "blind") == pytest.approx(BLIND_END, abs=1e-9)
    assert CohortJobRuntime.intensity_coupling is False


def test_scenario_model_block_enables_it_too() -> None:
    """`model.intensity_coupling: true` is the documented scenario-level
    switch, exactly parallel to CL-014's. Applied to the `blind` arm (which
    declares nothing) it must produce intensity semantics: the eth1g stream is
    on a separate card, so the trainer ends at the scoped 13.000."""
    sc = _load()
    sc["model"] = {"intensity_coupling": True}
    assert _trainer_end(sc, "blind") == pytest.approx(SCOPED_END, abs=1e-9)


def test_flag_on_row_carries_provenance_keys() -> None:
    sc = _with_intensity(_load())
    row = run_arm(sc, "intense", 7, None)
    assert row["intensity_coupling"] is True
    assert row["train_fabric_by_class"] == {"trainer": "roce", "donor": "roce"}


# --------------- 2. unsaturated shared link charges NOTHING ------------------

def test_unsaturated_shared_link_charges_exactly_zero() -> None:
    """Stream moved onto the collective's own wire (ckpt_fabric = roce).
    offered = 1.0 (busbw) + 0.01 (stream, bound by the donor's nic_in) = 1.01,
    capacity = min(10, 10) = 10 -> headroom -> factor exactly 1.0, so the
    trainer ends at 13.000, NOT the count-based 15.000. This is the digit the
    count form provably cannot produce: same wire, transfer in flight, zero
    charge."""
    sc = _with_intensity(_load())
    sc["classes"]["trainer"]["rates"]["ckpt_fabric"] = "roce"
    assert _trainer_end(sc, "intense") == pytest.approx(SCOPED_END, abs=1e-9)


# ------------------- 3. over-offered link: factor = offered/capacity ---------

def test_over_offered_link_charges_offered_over_capacity() -> None:
    """Same shared wire, line rate squeezed to 0.8 GB/s and the stream's
    offered rate raised to 0.2 GB/s so that offered (1.2) > capacity (0.8):
    the 2 overlapped iterations each stretch by exactly 1.2/0.8 = 1.5 on the
    collective -> 14.000 end-to-end (pencil above)."""
    sc = _with_intensity(_load())
    rates = sc["classes"]["trainer"]["rates"]
    rates["ckpt_fabric"] = "roce"
    rates["nic_in"] = 0.8
    rates["nic_out"] = 0.8
    donor = sc["classes"]["donor"]["rates"]
    donor["nic"] = 0.2       # per-stream demand cap: min(nic, disk) = 0.2
    donor["nic_in"] = 0.2    # path bound at the donor's ingress
    donor["nic_out"] = 0.2
    assert _trainer_end(sc, "intense") == pytest.approx(OVER_OFFERED_END,
                                                        abs=1e-9)


# ------------------- 4. separate fabric charges nothing, regardless ----------

def test_separate_fabric_is_free_at_any_capacity() -> None:
    """The committed scenario already puts the stream on eth1g. Under the
    intensity flag the collective's fabric carries no transfer demand, so the
    factor is 1.0 — even when the roce line rate is squeezed far below the
    collective's own demand."""
    sc = _with_intensity(_load())
    assert _trainer_end(sc, "intense") == pytest.approx(SCOPED_END, abs=1e-9)
    sc2 = _with_intensity(_load())
    rates = sc2["classes"]["trainer"]["rates"]
    rates["nic_in"] = 0.8            # capacity < collective demand alone —
    rates["nic_out"] = 0.8           # still free: no SHARED transfer demand
    assert _trainer_end(sc2, "intense") == pytest.approx(SCOPED_END, abs=1e-9)


# ------------------------ the factor, unit-level -----------------------------

def _worker(demand_by_fabric: dict[str, float], tasks_by_fabric=None):
    tasks = (tasks_by_fabric if tasks_by_fabric is not None
             else {f: 1 for f in demand_by_fabric})
    return SimpleNamespace(
        checkpoint_network_tasks=sum(tasks.values()),
        checkpoint_network_tasks_by_fabric=dict(tasks),
        checkpoint_network_demand_by_fabric=dict(demand_by_fabric))


def _slowdown(*, train_fabric: str, busbw: float, capacity: float,
              workers: list):
    rt = SimpleNamespace(intensity_coupling=True, fabric_aware_coupling=False,
                         train_fabric=train_fabric,
                         collective_busbw_gbps=busbw,
                         train_link_capacity_gbps=capacity,
                         workers={i: w for i, w in enumerate(workers)})
    return CohortJobRuntime._allreduce_slowdown(rt)


def test_factor_unsaturated_is_one_and_blames_nothing() -> None:
    """The exp1 shared-port numbers, all measured: NCCL RoCE busbw 4.03 GB/s,
    TCP-ceiling stream 1.57 GB/s, 100 Gb line rate 12.5 GB/s. offered 5.60 of
    12.5 -> factor exactly 1.0 (this is what H_slack claims about the whole tc
    ladder)."""
    s = _slowdown(train_fabric="roce", busbw=4.03, capacity=12.5,
                  workers=[_worker({"roce": 1.57})])
    assert s.factor == 1.0
    assert s.reasons == frozenset()


def test_factor_over_offered_is_offered_over_capacity() -> None:
    """CL-012.G's sat1g_on_tc1 arithmetic: 1 GbE line rate 0.125 GB/s, NCCL
    busbw on it 0.12 GB/s, stream shaped to 0.0125 GB/s. offered 0.1325 >
    0.125 -> factor = 0.1325/0.125 = 1.06 — order-of-magnitude BELOW the count
    form's 2.0, which is the discriminating consequence registered there."""
    s = _slowdown(train_fabric="eth1g", busbw=0.12, capacity=0.125,
                  workers=[_worker({"eth1g": 0.0125})])
    assert s.factor == pytest.approx((0.12 + 0.0125) / 0.125, abs=1e-12)
    assert s.reasons == frozenset({"checkpoint_network"})


def test_factor_at_exact_capacity_charges_nothing() -> None:
    """`offered > capacity` is strict: a link at exactly its line rate is not
    over-offered (no free parameter hides in the boundary)."""
    s = _slowdown(train_fabric="roce", busbw=10.0, capacity=12.5,
                  workers=[_worker({"roce": 2.5})])
    assert s.factor == 1.0
    assert s.reasons == frozenset()


def test_factor_separate_fabric_is_one_at_any_rate() -> None:
    s = _slowdown(train_fabric="roce", busbw=4.03, capacity=0.1,
                  workers=[_worker({"eth1g": 99.0})])
    assert s.factor == 1.0
    assert s.reasons == frozenset()


def test_factor_reduces_by_max_over_workers() -> None:
    """Mirrors the count path's max-over-workers reduction: the collective is
    gated by its most-contended member."""
    s = _slowdown(train_fabric="roce", busbw=1.0, capacity=1.0,
                  workers=[_worker({"roce": 0.25}), _worker({"roce": 1.0}),
                           _worker({})])
    assert s.factor == pytest.approx((1.0 + 1.0) / 1.0, abs=1e-12)


def test_intensity_branch_wins_over_the_count_paths() -> None:
    """With the intensity flag on, a same-fabric transfer that leaves headroom
    charges nothing even though BOTH count paths would charge 1 + tasks."""
    rt = SimpleNamespace(intensity_coupling=True, fabric_aware_coupling=True,
                         train_fabric="roce",
                         collective_busbw_gbps=4.03,
                         train_link_capacity_gbps=12.5,
                         workers={0: _worker({"roce": 1.57})})
    assert CohortJobRuntime._allreduce_slowdown(rt).factor == 1.0


def test_pre_cl016_mocks_without_the_flag_still_run() -> None:
    """test_fabric_coupling's SimpleNamespace mocks carry no intensity flag;
    the getattr default must route them to the CL-014 paths unchanged."""
    rt = SimpleNamespace(fabric_aware_coupling=False, train_fabric="roce",
                         workers={0: SimpleNamespace(
                             checkpoint_network_tasks=1,
                             checkpoint_network_tasks_by_fabric={"eth1g": 1})})
    assert CohortJobRuntime._allreduce_slowdown(rt).factor == 2.0


# ------------------------ 5. the demand ledger -------------------------------

def test_bump_maintains_the_demand_ledger_exactly() -> None:
    w = SimpleNamespace(checkpoint_network_tasks=0,
                        checkpoint_network_tasks_by_fabric={},
                        checkpoint_network_demand_by_fabric={})
    bump_network_tasks([w], "roce", +1, rate_gbps=1.57)
    bump_network_tasks([w], "roce", +1, rate_gbps=0.125)
    bump_network_tasks([w], "eth1g", +1, rate_gbps=0.091)
    assert w.checkpoint_network_demand_by_fabric == pytest.approx(
        {"roce": 1.695, "eth1g": 0.091})
    assert w.checkpoint_network_tasks_by_fabric == {"roce": 2, "eth1g": 1}
    bump_network_tasks([w], "roce", -1, rate_gbps=1.57)
    assert w.checkpoint_network_demand_by_fabric["roce"] == pytest.approx(0.125)
    bump_network_tasks([w], "roce", -1, rate_gbps=0.125)
    bump_network_tasks([w], "eth1g", -1, rate_gbps=0.091)
    assert w.checkpoint_network_demand_by_fabric == {}   # emptied with counts
    assert w.checkpoint_network_tasks == 0


def test_bump_drops_the_entry_with_the_count_so_residue_cannot_accumulate() -> None:
    """The demand entry is popped the moment its fabric's COUNT returns to 0
    (not when the float reaches 0.0), so mixed-rate churn cannot leave dust."""
    w = SimpleNamespace(checkpoint_network_tasks=0,
                        checkpoint_network_tasks_by_fabric={},
                        checkpoint_network_demand_by_fabric={})
    for _ in range(1000):
        bump_network_tasks([w], "roce", +1, rate_gbps=0.1)
        bump_network_tasks([w], "roce", +1, rate_gbps=0.30000000000000004)
        bump_network_tasks([w], "roce", -1, rate_gbps=0.1)
        bump_network_tasks([w], "roce", -1, rate_gbps=0.30000000000000004)
    assert w.checkpoint_network_demand_by_fabric == {}


def test_bump_tolerates_pre_cl016_worker_mocks() -> None:
    """A worker without the demand dict (the CL-014 test mocks) keeps both
    counters and raises nothing — the ledger is additive, never required."""
    w = SimpleNamespace(checkpoint_network_tasks=0,
                        checkpoint_network_tasks_by_fabric={})
    bump_network_tasks([w], "roce", +1, rate_gbps=1.0)
    assert w.checkpoint_network_tasks == 1
    assert w.checkpoint_network_tasks_by_fabric == {"roce": 1}
    bump_network_tasks([w], "roce", -1, rate_gbps=1.0)
    assert w.checkpoint_network_tasks == 0
