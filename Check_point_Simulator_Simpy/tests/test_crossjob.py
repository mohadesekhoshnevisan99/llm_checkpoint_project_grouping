"""Unit tests for the cross-job donor registry (checkpointing/crossjob.py)."""
from types import SimpleNamespace

from checkpointing.crossjob import DonorRegistry


def _worker(name, failed=False):
    return SimpleNamespace(name=name, failed=failed)


def make_registry():
    reg = DonorRegistry()
    for i in range(2):
        reg.register(_worker(f"jobA-rank-{i}"), "jobA")
        reg.register(_worker(f"jobB-rank-{i}"), "jobB")
    return reg


def test_reserve_excludes_own_job():
    reg = make_registry()
    granted = reg.reserve(2, exclude_job="jobA")
    assert len(granted) == 2
    assert all(g.job_id == "jobB" for g in granted)


def test_endogenous_refusal_while_flushing():
    reg = make_registry()
    for slot in reg.slots.values():
        if slot.job_id == "jobB":
            slot.flushing = True
    assert reg.reserve(2, exclude_job="jobA") == []
    assert reg.refusals["own_flush"] >= 2


def test_hosted_cap_and_release():
    reg = make_registry()
    first = reg.reserve(2, exclude_job="jobA")
    second = reg.reserve(2, exclude_job="jobA")   # cap=2 -> still fits
    third = reg.reserve(2, exclude_job="jobA")    # now full
    assert first and second and third == []
    assert reg.refusals["full"] >= 2
    reg.release(first)
    assert reg.reserve(2, exclude_job="jobA")     # slots freed


def test_all_or_nothing_reservation():
    reg = make_registry()
    for slot in reg.slots.values():
        if slot.worker.name == "jobB-rank-1":
            slot.flushing = True
    granted = reg.reserve(2, exclude_job="jobA")  # only 1 idle jobB donor
    assert granted == []
    assert all(s.hosted == 0 for s in reg.slots.values())


def test_best_peer_set_requires_all_shards_alive():
    from types import SimpleNamespace
    from checkpointing.crossjob import CrossJobPeerStrategy
    reg = make_registry()
    ns = SimpleNamespace(registry=reg, peer_copies={
        ("jobA", 0, "jobB-rank-0"): {"iteration": 20, "size_gb": 0.5, "donor_job": "jobB", "k": 2},
        ("jobA", 0, "jobB-rank-1"): {"iteration": 20, "size_gb": 0.5, "donor_job": "jobB", "k": 2},
    })
    best = CrossJobPeerStrategy._best_peer_set(ns, "jobA", 0)
    assert best is not None and best[0] == 20
    # kill one donor -> the it20 set is incomplete -> no usable peer set
    reg.slots["jobB-rank-1"].worker.failed = True
    assert CrossJobPeerStrategy._best_peer_set(ns, "jobA", 0) is None
