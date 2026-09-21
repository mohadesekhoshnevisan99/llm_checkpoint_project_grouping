from types import SimpleNamespace

from checkpointing.crossjob import DonorRegistry, _DonorSlot
from checkpointing.group_policy import normalize_group_policy


def make_slot(name, job_id):
    worker = SimpleNamespace(
        name=name,
        failed=False,
    )
    return _DonorSlot(
        worker=worker,
        job_id=job_id,
        represents=1,
    )


def test_policy_rejects_cross_group_donor():
    policy = {
        "jobs": {
            "alpha0": {
                "group_id": 0,
                "allowed_donor_job_ids": [],
            },
            "bravo0": {
                "group_id": 0,
                "allowed_donor_job_ids": ["alpha0"],
            },
            "be0": {
                "group_id": 2,
                "allowed_donor_job_ids": [],
            },
        }
    }

    allowed = normalize_group_policy(policy)

    assert allowed["bravo0"] == frozenset({"alpha0"})


def test_registry_filters_donors_by_group_policy():
    registry = DonorRegistry()
    registry.slots = {
        "alpha-slot": make_slot("alpha-slot", "alpha0"),
        "be-slot": make_slot("be-slot", "be0"),
    }

    registry.allowed_donors_by_job = {
        "bravo0": frozenset({"alpha0"}),
    }

    donors = registry._valid_donors(
        "bravo0",
        tier="ssd",
        piece_gb=1.0,
    )

    assert [slot.job_id for slot in donors] == ["alpha0"]


def test_empty_policy_keeps_legacy_behavior():
    registry = DonorRegistry()
    registry.slots = {
        "alpha-slot": make_slot("alpha-slot", "alpha0"),
        "be-slot": make_slot("be-slot", "be0"),
    }

    registry.allowed_donors_by_job = {}

    donors = registry._valid_donors(
        "bravo0",
        tier="ssd",
        piece_gb=1.0,
    )

    assert {slot.job_id for slot in donors} == {"alpha0", "be0"}
