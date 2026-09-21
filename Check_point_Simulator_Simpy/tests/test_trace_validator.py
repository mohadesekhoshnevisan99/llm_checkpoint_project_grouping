"""Focused tests for trace-level physical invariants."""

from __future__ import annotations

import json
from pathlib import Path

from trace_validator import represented_ranks, validate


def _scenario() -> dict:
    return {
        "cluster": {
            "gpu_cpu_bandwidth_gbps": 10.0,
            "network_bandwidth_gbps": 10.0,
            "local_ssd_bandwidth_gbps": 10.0,
        },
        "classes": {
            "owner": {"ranks": 1, "cohorts": 1},
            "donor": {"ranks": 5, "cohorts": 2},
        },
        "store": {
            "in_gbps": 10.0,
            "out_gbps": 10.0,
            "disk_gbps": 10.0,
            "stream_gbps": 10.0,
        },
    }


def _event(**overrides) -> dict:
    event = {
        "category": "Checkpoint",
        "data_gb": 0.0,
        "details": {},
        "end": 2.0,
        "iteration": 1,
        "job_id": "owner0",
        "operation": "checkpoint_stage_dram_to_peers",
        "rank": 0,
        "start": 1.0,
    }
    event.update(overrides)
    return event


def _validate(tmp_path: Path, events: list[dict]) -> dict:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    return validate(trace, _scenario())


def test_represented_ranks_distributes_uneven_remainder() -> None:
    classes = _scenario()["classes"]
    assert represented_ranks("donor0-cohort-0", classes) == 3
    assert represented_ranks("donor0-cohort-1", classes) == 2


def test_i5_uses_actual_remainder_aware_cohort_size(tmp_path: Path) -> None:
    result = _validate(tmp_path, [
        _event(details={
            "path": "peer",
            "donors": ["donor0-cohort-0"],
            "shards_by_donor": {"donor0-cohort-0": 5},
        }),
    ])

    assert not any(message.startswith("I5:") for message in result["failures"])


def test_i5_still_rejects_overfull_smaller_remainder_cohort(tmp_path: Path) -> None:
    result = _validate(tmp_path, [
        _event(details={
            "path": "peer",
            "donors": ["donor0-cohort-1"],
            "shards_by_donor": {"donor0-cohort-1": 5},
        }),
    ])

    assert any(message.startswith("I5:") for message in result["failures"])


def test_i13_rejects_overlapping_store_writes_above_aggregate_cap(
    tmp_path: Path,
) -> None:
    events = []
    for rank, donor in enumerate(("donor0-cohort-0", "donor0-cohort-1")):
        events.append(_event(
            end=0.5,
            iteration=3,
            rank=rank,
            start=0.0,
            details={"path": "peer", "donors": [donor]},
        ))
        events.append(_event(
            data_gb=8.0,
            end=2.0,
            iteration=3,
            operation="checkpoint_donor_ssd_to_l3_drain_chunk",
            rank=rank,
            start=1.0,
            destination="__store__",
            details={
                "capacity_mode": True,
                "drain_pieces": {donor: 8.0},
                "path": "l3_drain",
            },
        ))

    result = _validate(tmp_path, events)

    assert any(message.startswith("I13:") for message in result["failures"])
    assert not any(message.startswith("I11:") for message in result["failures"])


def test_i13_accepts_overlapping_store_writes_within_aggregate_cap(
    tmp_path: Path,
) -> None:
    events = []
    for rank, donor in enumerate(("donor0-cohort-0", "donor0-cohort-1")):
        events.append(_event(
            end=0.5,
            iteration=3,
            rank=rank,
            start=0.0,
            details={"path": "peer", "donors": [donor]},
        ))
        events.append(_event(
            data_gb=5.0,
            end=2.0,
            iteration=3,
            operation="checkpoint_donor_ssd_to_l3_drain_chunk",
            rank=rank,
            start=1.0,
            destination="__store__",
            details={
                "capacity_mode": True,
                "drain_pieces": {donor: 5.0},
                "path": "l3_drain",
            },
        ))

    result = _validate(tmp_path, events)

    assert not any(message.startswith("I13:") for message in result["failures"])

