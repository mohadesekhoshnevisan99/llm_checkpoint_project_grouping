"""Fabric-scoped checkpoint <-> all-reduce coupling (protocol CL-012 / CL-014).

The hardware says a checkpoint stream on a physically separate NIC costs the
training collective nothing (alpha = -0.05 pp, CI [-0.322, +0.422] on the
`eth1g` arm); the simulator said +4.24 pp, because its coupling was a COUNT of
in-flight transfers with no reference to which wire they ride. This change makes
the count fabric-scoped. What it deliberately does NOT do is change the
magnitude on a shared wire -- that is still 1 + tasks, still duration-driven;
making it intensity-based is CL-012.D S3, a separate C3.

What is pinned here:

  1. The pencil hand-check, to the digit (scenarios/handcheck_fabric.yaml):
     a separate-fabric transfer produces EXACTLY zero all-reduce slowdown
     (trainer ends 13.000 s), the same transfer on the collective's own wire
     produces the documented 1 + tasks (trainer ends 15.000 s).
  2. The regression contract: flag ABSENT => the fabric-blind code path, the
     same numbers, and no new keys in the results row. Every committed result in
     this repo depends on that.
  3. The three degenerate declarations all collapse onto the old behaviour --
     shared fabric, ckpt_fabric undeclared, and no fabric declared anywhere --
     so turning the flag on cannot silently delete coupling that a scenario
     never asked to have deleted.
  4. The slowdown function itself, unit-level, including the multi-transfer
     factor and the max-over-workers reduction.
  5. The two counters can never disagree: the fabric-blind aggregate is exactly
     the sum of the per-fabric counts, at every point in a bump sequence.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from checkpointing.base import (DEFAULT_FABRIC, bump_network_tasks,
                                ckpt_fabric_of, train_fabric_of)
from run_scenario import CohortJobRuntime, run_arm

SCENARIO = (Path(__file__).resolve().parent.parent
            / "scenarios" / "handcheck_fabric.yaml")

# The pencil, from the scenario header. Derivation there; digits here.
BLIND_END = 15.0        # 9.000 + 2 iterations x (1.0 compute + 2 x 1.0 collective)
SCOPED_END = 13.0       # 9.000 + 2 iterations x (1.0 compute + 1 x 1.0 collective)


def _load() -> dict:
    return yaml.safe_load(SCENARIO.read_text())


def _trainer_end(sc: dict, arm: str) -> float:
    return run_arm(sc, arm, 7, None)["per_class_end_s"]["trainer"]


# --------------------------- 1. the pencil hand-check ------------------------

def test_separate_fabric_costs_the_collective_exactly_zero() -> None:
    """ckpt_fabric (eth1g) != nic_fabric (roce) => multiplier is exactly 1.0."""
    assert _trainer_end(_load(), "scoped") == pytest.approx(SCOPED_END, abs=1e-9)


def test_same_fabric_still_costs_the_documented_one_plus_tasks() -> None:
    """Same scenario, same arm, stream moved onto the collective's own wire."""
    sc = _load()
    sc["classes"]["trainer"]["rates"]["ckpt_fabric"] = "roce"
    assert _trainer_end(sc, "scoped") == pytest.approx(BLIND_END, abs=1e-9)


def test_the_gap_is_exactly_two_iterations_of_all_reduce() -> None:
    """15.000 - 13.000 = 2 iterations x ar_base x ((1 + tasks) - 1), tasks = 1."""
    sc = _load()
    ar_base = (2 * (2 - 1) / 2
               * sc["classes"]["trainer"]["gradient_gb_per_rank"]
               / sc["classes"]["trainer"]["rates"]["nic"])
    overlapped_iterations = 2
    assert BLIND_END - SCOPED_END == pytest.approx(
        overlapped_iterations * ar_base * ((1 + 1) - 1), abs=1e-9)


# ------------------- 2. the regression contract: flag OFF --------------------

def test_flag_absent_is_the_fabric_blind_model() -> None:
    """The `blind` arm declares no flag anywhere, so the separate card still
    couples -- exactly the CL-007 disagreement, reproduced in miniature."""
    assert _trainer_end(_load(), "blind") == pytest.approx(BLIND_END, abs=1e-9)


def test_flag_absent_adds_no_keys_to_the_results_row() -> None:
    """Committed mini-C staircases / rack tables / parallelism tables are
    compared byte-for-byte mod wall_s; a new key would break every one."""
    row = run_arm(_load(), "blind", 7, None)
    assert "fabric_aware_coupling" not in row
    assert "train_fabric_by_class" not in row
    row_on = run_arm(_load(), "scoped", 7, None)
    assert row_on["fabric_aware_coupling"] is True
    assert row_on["train_fabric_by_class"] == {"trainer": "roce", "donor": "roce"}


def test_scenario_model_block_enables_it_too() -> None:
    """`model.fabric_aware_coupling: true` is the documented scenario-level
    switch; the arm-level key exists only so one file can carry the A/B."""
    sc = _load()
    sc["model"] = {"fabric_aware_coupling": True}
    assert _trainer_end(sc, "blind") == pytest.approx(SCOPED_END, abs=1e-9)


def test_flag_does_not_leak_between_arms() -> None:
    """A class attribute set per run: `scoped` then `blind` must not carry over."""
    sc = _load()
    assert _trainer_end(sc, "scoped") == pytest.approx(SCOPED_END, abs=1e-9)
    assert _trainer_end(sc, "blind") == pytest.approx(BLIND_END, abs=1e-9)
    assert CohortJobRuntime.fabric_aware_coupling is False


# --------------- 3. degenerate declarations collapse onto today --------------

@pytest.mark.parametrize("mutate,label", [
    (lambda sc: sc["classes"]["trainer"]["rates"].update(ckpt_fabric="roce"),
     "checkpoints on the collective's own wire"),
    (lambda sc: sc["classes"]["trainer"]["rates"].pop("ckpt_fabric"),
     "ckpt_fabric undeclared => falls back to nic_fabric"),
    (lambda sc: [c["rates"].pop(k, None)
                 for c in sc["classes"].values()
                 for k in ("nic_fabric", "ckpt_fabric")],
     "no fabric declared anywhere => one default fabric"),
])
def test_flag_on_without_a_separate_fabric_reproduces_today(mutate, label) -> None:
    sc = _load()
    mutate(sc)
    assert _trainer_end(sc, "scoped") == pytest.approx(BLIND_END, abs=1e-9), label


# ------------------------ 4. the slowdown function ---------------------------

def _worker(by_fabric: dict[str, int]):
    return SimpleNamespace(
        checkpoint_network_tasks=sum(by_fabric.values()),
        checkpoint_network_tasks_by_fabric=dict(by_fabric))


def _slowdown(*, fabric_aware: bool, train_fabric: str, workers: list):
    rt = SimpleNamespace(fabric_aware_coupling=fabric_aware,
                         train_fabric=train_fabric,
                         workers={i: w for i, w in enumerate(workers)})
    return CohortJobRuntime._allreduce_slowdown(rt)


def test_slowdown_separate_fabric_is_exactly_one_and_blames_nothing() -> None:
    s = _slowdown(fabric_aware=True, train_fabric="roce",
                  workers=[_worker({"eth1g": 1})])
    assert s.factor == 1.0
    assert s.reasons == frozenset()


def test_slowdown_same_fabric_is_one_plus_tasks() -> None:
    for tasks in (1, 2, 3):
        s = _slowdown(fabric_aware=True, train_fabric="roce",
                      workers=[_worker({"roce": tasks})])
        assert s.factor == 1.0 + tasks
        assert s.reasons == frozenset({"checkpoint_network"})


def test_slowdown_counts_only_the_collectives_own_fabric() -> None:
    """A worker carrying both: only the roce one may reach the collective."""
    s = _slowdown(fabric_aware=True, train_fabric="roce",
                  workers=[_worker({"roce": 1, "eth1g": 5})])
    assert s.factor == 2.0


def test_slowdown_reduces_by_max_over_workers_on_both_paths() -> None:
    workers = [_worker({"roce": 1}), _worker({"roce": 3}), _worker({})]
    assert _slowdown(fabric_aware=True, train_fabric="roce",
                     workers=workers).factor == 4.0
    assert _slowdown(fabric_aware=False, train_fabric="roce",
                     workers=workers).factor == 4.0


def test_slowdown_fabric_blind_path_ignores_the_fabric_split() -> None:
    """Flag off: the aggregate counter alone decides, exactly as before."""
    s = _slowdown(fabric_aware=False, train_fabric="roce",
                  workers=[_worker({"eth1g": 1})])
    assert s.factor == 2.0
    assert s.reasons == frozenset({"checkpoint_network"})


# ------------------------ 5. the counters agree ------------------------------

def test_bump_keeps_the_aggregate_equal_to_the_per_fabric_sum() -> None:
    w = SimpleNamespace(checkpoint_network_tasks=0,
                        checkpoint_network_tasks_by_fabric={})
    for fabric, delta in (("roce", +1), ("eth1g", +1), ("roce", +1),
                          ("roce", -1), ("eth1g", -1), ("roce", -1)):
        bump_network_tasks([w], fabric, delta)
        assert w.checkpoint_network_tasks == sum(
            w.checkpoint_network_tasks_by_fabric.values())
        assert all(v > 0 for v in w.checkpoint_network_tasks_by_fabric.values())
    assert w.checkpoint_network_tasks == 0
    assert w.checkpoint_network_tasks_by_fabric == {}   # zeros are dropped


def test_fabric_resolution_rules() -> None:
    assert ckpt_fabric_of({}, "j") == DEFAULT_FABRIC
    assert ckpt_fabric_of({"j": {"nic_fabric": "roce"}}, "j") == "roce"
    assert ckpt_fabric_of(
        {"j": {"nic_fabric": "roce", "ckpt_fabric": "eth1g"}}, "j") == "eth1g"
    assert train_fabric_of(None) == DEFAULT_FABRIC
    assert train_fabric_of({"nic": 4.03}) == DEFAULT_FABRIC
    assert train_fabric_of({"nic_fabric": "roce"}) == "roce"
