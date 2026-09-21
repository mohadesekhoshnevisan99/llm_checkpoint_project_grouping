"""Rack failure domains + injob_ssd placement (rack_failure_spec.md + Sam's
post-spec overrides, 2026-07-27).

These tests pin, to the digit, the pencil math in scenarios/handcheck_rack.yaml
(see the comment block there for the full derivation):

  1. ours_full (crossjob k=1, anti-affinity, donor-drain L3): recovery source
     per phase — BEFORE the rack event a node loss restores crossjob_peer @3
     (loss 2); DURING the dark-rack window it falls to the drained L3 pieces
     (crossjob_peer_l3_stitch @3, loss 3); the OWNER-rack event restarts both
     cohorts on replacements from the best OFF-RACK copy (crossjob_peer @6,
     loss 2 each). Destroyed/dark bookkeeping and the exact completion times.
  2. ours_injob_ssd: the ring replica survives owner node loss twice
     (injob_ssd_replica @3), then dies WITH the owner's rack (initial_state x2)
     — the same-rack-by-packing bet the 2x2 prices.
  3. ideal: no rack machinery at all (and no rack keys in the row).
  4. DonorRegistry rack anti-affinity at the unit level: k rack-disjoint
     donors, never the owner's rack; infeasibility falls back best-effort and
     INCREMENTS the counter (never silent).
  5. trace_validator I19/I21 actually catch violations (tampered traces).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

import trace_validator
from checkpointing.crossjob import DonorRegistry
from run_scenario import run_arm

SCENARIO = Path(__file__).resolve().parent.parent / "scenarios" / "handcheck_rack.yaml"


def _load() -> dict:
    return yaml.safe_load(SCENARIO.read_text())


def _events(trace: Path) -> list[dict]:
    with gzip.open(trace, "rt") as fh:
        return [json.loads(line) for line in fh]


# --------------------------- end-to-end hand-check ---------------------------

def test_rack_handcheck_ours_full(tmp_path) -> None:
    sc = _load()
    trace = tmp_path / "full.jsonl.gz"
    row = run_arm(sc, "ours_full", 7, trace)

    # phases (pencil): BEFORE -> peer @3; DURING dark rack -> L3 stitch @3;
    # owner-rack event -> off-rack peer @6 x2; host rack event -> scratch x2.
    assert row["recovery_source_tiers"] == {
        "crossjob_peer": 3, "crossjob_peer_l3_stitch": 1, "initial_state": 2}
    assert row["failures"] == {"node": 2, "rack": 2}
    assert row["staleness_by_tier"] == {
        "crossjob_peer": 2, "crossjob_peer_l3_stitch": 3, "initial_state": 8}
    assert row["lost_iters_p50"] == 3 and row["lost_iters_max"] == 8
    assert row["total_loss"] == 2                      # host's two scratch restores
    assert row["stitch_recoveries"] == 1
    assert row["l3_drains"] == 6 and row["rank_flushes"] == 6
    assert row["durable_frac"] == 1.0
    assert row["refusals"] == {}       # dark idle donors leave the board silently

    # completion to the digit (see the YAML's timeline)
    assert row["per_class_end_s"] == {"own": 252.1, "host": 500.2, "fil": 30.0}

    # rack blast bookkeeping
    assert row["rack_events"] == 3
    assert row["rack_nodes_dark"] == 6
    assert row["rack_nodes_destroyed"] == 3            # node 2 + fil's 4,5
    assert row["rack_local_copies_lost"] == 2          # the @6 DRAM captures
    assert row["rack_dram_pieces_lost"] == 0
    assert row["rack_ssd_pieces_destroyed"] == 1       # @3 piece on node 2
    assert row["rack_ssd_pieces_dark"] == 1            # @3 piece on node 3
    assert row["rack_affinity_fallbacks"] == 0         # k=1 always feasible

    # trace-level pencil: every peer flush is exactly 1.000 s (1.0 GB at
    # min(nic 2, 1 x disk 1)) and the anti-affine stripe lands on host0 (rack
    # 1), never on the owner's rack-0 neighbor.
    ev = _events(trace)
    flushes = [e for e in ev
               if e.get("operation") == "checkpoint_dram_to_crossjob_peers_chunk"]
    assert len(flushes) == 6
    assert all(abs((e["end"] - e["start"]) - 1.0) < 1e-6 for e in flushes)
    assert {e["destination"] for e in flushes} == {
        "host0-cohort-0/ssd", "host0-cohort-1/ssd"}
    racks = [e for e in ev if e.get("operation") == "rack_failure"]
    assert [e["start"] for e in racks] == [75.0, 100.0, 165.0]
    assert racks[0]["details"]["destroyed_nodes"] == [2]

    # the validator (I18-I21 live on this trace) stays green
    result = trace_validator.validate(trace, sc)
    assert result["failures"] == []


def test_rack_handcheck_injob_ssd(tmp_path) -> None:
    sc = _load()
    trace = tmp_path / "injob.jsonl.gz"
    row = run_arm(sc, "ours_injob_ssd", 7, trace)

    # the ring replica survives owner node loss twice, then dies with the
    # owner's rack: nothing left -> initial_state for both cohorts (+ host x2).
    assert row["recovery_source_tiers"] == {
        "injob_ssd_replica": 2, "initial_state": 4}
    assert row["failures"] == {"node": 2, "rack": 2}
    assert row["lost_iters_p50"] == 8 and row["lost_iters_max"] == 8
    assert row["total_loss"] == 4
    assert row["per_class_end_s"] == {"own": 311.9, "host": 500.2, "fil": 30.0}
    assert row["injob_ssd_flushes"] == 10
    assert row["rank_flushes"] == 10 and row["durable_frac"] == 1.0
    assert row["l3_drains"] == 0                       # owner_push, no store_every
    assert row["fallback_flushes"] == 0
    assert row["rack_ssd_pieces_dark"] == 2            # the @6 replicas, own rack
    assert row["rack_ssd_pieces_destroyed"] == 0
    assert row["rack_local_copies_lost"] == 2
    assert row["rack_affinity_fallbacks"] == 0         # injob path never reserves

    # replica waves: same-job ring neighbor, on SSD, 1.000 s each
    ev = _events(trace)
    reps = [e for e in ev
            if e.get("operation") == "checkpoint_dram_to_injob_ssd_chunk"]
    assert len(reps) == 10
    assert all(abs((e["end"] - e["start"]) - 1.0) < 1e-6 for e in reps)
    assert {e["destination"] for e in reps} == {
        "own0-cohort-0/ssd", "own0-cohort-1/ssd"}
    # recovery labels carry the placement (audit hook for the 2x2)
    recs = [e for e in ev
            if e.get("operation") == "checkpoint_injob_ssd_to_dram_recovery_chunk"]
    assert len(recs) == 2
    assert all(e["source"] == "own0-cohort-1/ssd" for e in recs)

    result = trace_validator.validate(trace, sc)
    assert result["failures"] == []


def test_rack_handcheck_ideal(tmp_path) -> None:
    sc = _load()
    row = run_arm(sc, "ideal", 7, tmp_path / "ideal.jsonl.gz")
    assert row["per_class_end_s"] == {"own": 120.0, "host": 400.0, "fil": 30.0}
    assert row["failures"] == {}
    assert "rack_events" not in row      # ideal: no rack machinery, no keys


# --------------------- registry rack anti-affinity (unit) ---------------------

def _registry(rack_of: dict[str, int | None], owner_rack: int | None = 0):
    reg = DonorRegistry()
    for name, rack in rack_of.items():
        reg.register(SimpleNamespace(name=name, failed=False, represents=1),
                     job_id=f"job-{name}")
    reg.rack_anti_affinity = True
    reg.racks_by_slot = {
        name: (frozenset() if rack is None else frozenset({rack}))
        for name, rack in rack_of.items()}
    reg.racks_by_slot["owner-w"] = (
        frozenset() if owner_rack is None else frozenset({owner_rack}))
    return reg


def test_rack_anti_affinity_strict_distinct_racks() -> None:
    """k pieces on k distinct racks, never the owner's rack: the owner-rack
    donor and the second rack-1 donor are both skipped in the strict pass."""
    reg = _registry({"a": 0, "b": 1, "c": 1, "d": 2})   # owner rack = 0
    granted = reg.reserve(2, exclude_job="own", allow_partial=True,
                          owner="owner-w")
    assert [s.worker.name for s in granted] == ["b", "d"]
    assert reg.rack_fallbacks == 0 and reg.last_rack_fallback is False
    assert reg.refusals["rack_conflict"] == 2           # a (owner rack) + c


def test_rack_anti_affinity_fallback_counts_and_flags() -> None:
    """Infeasible strictly (both donors on rack 1) -> best-effort spread:
    the rack-clean grant is kept, the rest fills anyway, and the fallback is
    COUNTED + flagged (never silent)."""
    reg = _registry({"b": 1, "c": 1})
    granted = reg.reserve(2, exclude_job="own", allow_partial=True,
                          owner="owner-w")
    assert [s.worker.name for s in granted] == ["b", "c"]
    assert reg.rack_fallbacks == 1 and reg.last_rack_fallback is True


def test_rack_anti_affinity_owner_rack_is_last_resort() -> None:
    """Fallback fills non-owner-rack candidates first; the owner's own rack is
    granted only when nothing else exists."""
    reg = _registry({"a": 0, "b": 1})                   # a shares owner rack 0
    granted = reg.reserve(2, exclude_job="own", allow_partial=True,
                          owner="owner-w")
    assert [s.worker.name for s in granted] == ["b", "a"]
    assert reg.rack_fallbacks == 1


def test_rack_anti_affinity_relocated_donors_never_conflict() -> None:
    """Rack-less (relocated) donors have empty rack sets: always eligible."""
    reg = _registry({"b": None, "c": None})
    granted = reg.reserve(2, exclude_job="own", allow_partial=True,
                          owner="owner-w")
    assert [s.worker.name for s in granted] == ["b", "c"]
    assert reg.rack_fallbacks == 0


def test_rack_anti_affinity_cohort_shards() -> None:
    """Cohort reserve_shards: rack-disjoint slots first (a take>1 bundle is
    one piece-group on one slot), deferred same-rack slots only via the
    counted fallback."""
    reg = DonorRegistry()
    for name in ("b", "c"):
        reg.register(SimpleNamespace(name=name, failed=False, represents=2),
                     job_id=f"job-{name}")
    reg.rack_anti_affinity = True
    reg.racks_by_slot = {"b": frozenset({1}), "c": frozenset({1}),
                         "owner-w": frozenset({0})}
    granted = reg.reserve_shards(6, exclude_job="own", allow_partial=True,
                                 owner="owner-w")
    assert [(s.worker.name, t) for s, t in granted] == [("b", 4), ("c", 2)]
    assert reg.rack_fallbacks == 1 and reg.last_rack_fallback is True


def test_registry_off_keeps_legacy_reserve() -> None:
    """rack_anti_affinity False (every rack-less scenario): the owner kwarg is
    inert and the legacy least-loaded path is taken unchanged."""
    reg = _registry({"a": 0, "b": 1})
    reg.rack_anti_affinity = False
    granted = reg.reserve(2, exclude_job="own", allow_partial=True,
                          owner="owner-w")
    assert [s.worker.name for s in granted] == ["a", "b"]   # insertion order
    assert reg.rack_fallbacks == 0 and reg.last_rack_fallback is False


# ------------------------ validator negative controls ------------------------

def _tampered(tmp_path, base_events: list[dict], extra: dict) -> Path:
    path = tmp_path / "tampered.jsonl"
    with path.open("w") as fh:
        for e in base_events:
            fh.write(json.dumps(e) + "\n")
        fh.write(json.dumps(extra) + "\n")
    return path


def _fake_event(**kw) -> dict:
    base = {"event_id": 999999, "start": 0.0, "end": 0.0, "duration": 0.0,
            "rank": 0, "node": "own0-cohort-0", "category": "Checkpoint",
            "operation": "x", "run_id": 0, "job_id": "own0",
            "pipeline_stage": 0, "data_parallel_rank": 0,
            "physical_node": None, "resources": [], "iteration": 3,
            "source": None, "destination": None, "data_gb": 1.0,
            "failure_type": None, "details": {}}
    base.update(kw)
    return base


def test_validator_i19_catches_dark_ssd_recovery(tmp_path) -> None:
    """A recovery reading a dark donor's SSD inside its dark window must FAIL
    I19 (the invariant is live, not vacuous)."""
    sc = _load()
    trace = tmp_path / "full.jsonl.gz"
    run_arm(sc, "ours_full", 7, trace)
    fake = _fake_event(
        start=80.0, end=80.5, duration=0.5, category="Recovery",
        operation="checkpoint_crossjob_peers_to_dram_recovery_chunk",
        source="host0-cohort-0/ssd", destination="own0-cohort-0/dram",
        details={"checkpoint_source_tier": "crossjob_peer"})
    result = trace_validator.validate(_tampered(tmp_path, _events(trace), fake), sc)
    assert any(msg.startswith("I19") for msg in result["failures"]), \
        result["failures"]


def test_validator_i21_catches_owner_rack_piece(tmp_path) -> None:
    """A non-fallback stripe putting a piece in the OWNER's rack must FAIL I21."""
    sc = _load()
    trace = tmp_path / "full.jsonl.gz"
    run_arm(sc, "ours_full", 7, trace)
    fake = _fake_event(
        start=45.0, end=46.0, duration=1.0,
        operation="checkpoint_dram_to_crossjob_peers_chunk",
        source="own0-cohort-0/dram", destination="own0-cohort-1/ssd",
        details={"path": "peer", "donors": ["own0-cohort-1"], "kpeers": 1,
                 "capacity_mode": True})
    result = trace_validator.validate(_tampered(tmp_path, _events(trace), fake), sc)
    assert any(msg.startswith("I21") for msg in result["failures"]), \
        result["failures"]
