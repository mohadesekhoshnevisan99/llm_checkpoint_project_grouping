"""Reboot failure type (reboot_failure_type_spec.md, Sam-approved 2026-07-21).

A `reboot` restarts the host on the SAME node: DRAM/GPU state is LOST, but the
local disk SURVIVES — the job's OWN local-SSD copy and any hosted donor pieces
on that node's SSD both persist across the restart. These tests pin:

  1. End-to-end (scenarios/handcheck_reboot.yaml): a LOCAL-placed job hit by a
     scheduled reboot recovers from its OWN surviving local-SSD copy at the last
     landed persist iteration (NOT L3/store, NOT scratch/initial_state), and the
     ~180 s host downtime is unmistakable in the completion time.
  2. Strategy-level: the reboot DRAM-tier wipe (drop_local_copies(tier='dram'))
     deletes only the on-node DRAM copy — the on-node SSD copy AND the hosted
     donor pieces (peer_copies) on that node SURVIVE, unlike a node/spot wipe.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from checkpointing.crossjob import CrossJobPeerStrategy
from checkpointing.tiered import CheckpointCopy
from run_scenario import run_arm

SCENARIO = Path(__file__).resolve().parent.parent / "scenarios" / "handcheck_reboot.yaml"


def _load(name: str = "handcheck_reboot.yaml") -> dict:
    return yaml.safe_load(SCENARIO.read_text())


def test_reboot_recovers_from_own_local_ssd_at_pre_reboot_iteration(tmp_path) -> None:
    """PENCIL MATH (handcheck_reboot.yaml, loc: k=0 local, every 3, 12 iters,
    iter 10 s): persist waves land their SSD write ~one iter late (@3~t41, @6~t71,
    @9~t101). The scheduled reboot AFTER it=8 fires ~t80 — a full iteration past
    @6's landing and before @9 fires — so the @6 SSD copy SURVIVES the host
    restart while DRAM is wiped. Recovery MUST come from that OWN SSD copy at
    it=6 (loss = 8 - 6 = 2), NOT L3/store (there is none) and NOT scratch."""
    sc = _load()
    row = run_arm(sc, "ours_local", 7, tmp_path / "reboot.jsonl.gz")

    # exactly one organic-free scheduled reboot fired
    assert row["failures"] == {"reboot": 1}
    assert [e["failure_type"] for e in row["scheduled_failures_fired"]] == ["reboot"]

    # recovery came from the node's OWN surviving local SSD — and ONLY that
    assert row["recovery_source_tiers"] == {"ssd": 1}
    assert row["reboot_local_ssd_recoveries"] == 1
    # an own-SSD reboot recovery is a legitimate insured recovery, NOT stale/scratch
    assert row["scratch_stale_recoveries"] == 0
    assert row["total_loss"] == 0
    assert "store" not in row["recovery_source_tiers"]
    assert "initial_state" not in row["recovery_source_tiers"]

    # rolled back to the pre-reboot persist iteration @6 -> lost it 7,8 (= 2)
    assert row["lost_iters_max"] == 2

    # the ~180 s host downtime is unmistakable: completion is at LEAST ideal+180.
    ideal = run_arm(sc, "ideal", 7, tmp_path / "ideal.jsonl.gz")
    loc = row["per_class_end_s"]["loc"]
    loc_ideal = ideal["per_class_end_s"]["loc"]
    assert loc >= loc_ideal + 180.0
    # tighter pencil bound: 180 down + 2 lost iters (20 s) + recovery/capture (~1.3 s)
    assert loc == 321.4, loc
    assert loc_ideal == 120.0, loc_ideal


def _bare_strategy() -> CrossJobPeerStrategy:
    """A CrossJobPeerStrategy shell for exercising drop_local_copies in isolation
    (it only reads/writes self.copies). Bypasses the heavy runtime constructor —
    we populate the two copy stores by hand."""
    s = object.__new__(CrossJobPeerStrategy)
    s.copies = {}
    s.peer_copies = {}
    return s


def _copy(owner: int, loc: int, tier: str, it: int) -> CheckpointCopy:
    return CheckpointCopy(owner_rank=owner, location_rank=loc, tier=tier,
                          iteration=it, size_gb=1.0, completed_at=0.0)


def test_reboot_dram_wipe_keeps_own_ssd_and_hosted_donor_pieces() -> None:
    """Reboot semantics at the strategy level: wiping the DRAM tier on node rank=2
    (drop_local_copies(tier='dram')) removes ONLY that node's on-node DRAM copy.
    The on-node SSD copy survives the host restart, and the hosted donor pieces
    (peer_copies keyed by donor NODE NAME) are never touched by a local-copy wipe
    — so a donor piece physically on the rebooted node's SSD SURVIVES."""
    s = _bare_strategy()
    # own copies on node rank=2 (DRAM + SSD), plus one on a different node rank=5
    s.copies = {
        (2, 2, "dram"): _copy(2, 2, "dram", 6),
        (2, 2, "ssd"): _copy(2, 2, "ssd", 6),
        (5, 5, "ssd"): _copy(5, 5, "ssd", 6),
    }
    # a hosted donor piece for another job, physically sitting on node rank=2
    s.peer_copies = {("otherjob", 0, "loc-cohort-2"): {"iteration": 6, "k": 1}}

    s.drop_local_copies(2, tier="dram")

    # DRAM on the rebooted node is gone; its SSD copy survives the restart
    assert (2, 2, "dram") not in s.copies
    assert (2, 2, "ssd") in s.copies
    # an unrelated node's copy is untouched
    assert (5, 5, "ssd") in s.copies
    # hosted donor pieces are NEVER destroyed by a reboot's DRAM wipe
    assert ("otherjob", 0, "loc-cohort-2") in s.peer_copies


def test_node_loss_wipe_drops_every_on_node_tier() -> None:
    """Contrast: a node/spot wipe (tier=None) drops EVERY located tier on the
    lost node — both DRAM and SSD — leaving other nodes intact. (Hosted pieces
    are dropped separately by the driver's drop_hosted_copies on node/spot; this
    method only clears the owner's own on-node copies.)"""
    s = _bare_strategy()
    s.copies = {
        (2, 2, "dram"): _copy(2, 2, "dram", 6),
        (2, 2, "ssd"): _copy(2, 2, "ssd", 6),
        (5, 5, "ssd"): _copy(5, 5, "ssd", 6),
    }

    s.drop_local_copies(2, tier=None)

    assert (2, 2, "dram") not in s.copies
    assert (2, 2, "ssd") not in s.copies      # SSD gone too under true node loss
    assert (5, 5, "ssd") in s.copies
