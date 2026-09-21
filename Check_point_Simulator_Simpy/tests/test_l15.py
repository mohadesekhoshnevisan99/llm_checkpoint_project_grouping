"""L1.5 peer-DRAM tier hand-checks (l15_peer_dram_spec.md, gate 2c).

Pencil math asserted TO THE DIGIT on scenarios/handcheck_l15{,b}.yaml
(2 jobs x 1 rank, no organic failures, one scheduled host reboot + one
scheduled owner node loss; full derivations in the YAML headers):

  stripe (k=1)      1.0 GB / min(NIC_out 2.0, 1 x NIC_in 2.0)  = 0.500 s
                    — the donor DISK is not on the write path (the speed win)
  demotion          1.0 GB / disk                              = 1.000 s (A)
                                                               = 20.000 s (B)
  L3 drain          chained strictly AFTER the demotion (reads donor SSD)

  CASE A (handcheck_l15): host reboot at t=50 lands AFTER @3's demotion
    (41.6) -> the demoted piece survives on the host SSD (the existing
    hosted-SSD reboot exemption, untouched); the owner's node loss then
    recovers at the DEMOTED-piece age: crossjob_peer @3, loss = 5-3 = 2.
  CASE B (handcheck_l15b): slow disk (20 s demotion window); host reboot at
    t=80 lands INSIDE @6's demotion -> the @6 dram piece is WIPED (its
    demotion aborts) while the demoted @3 piece survives (write-then-rename
    shadow) -> recovery FALLS TO THE L2 AGE: crossjob_peer @3, loss = 8-3 = 5.

Plus strategy-level pins for the volatility taxonomy: dram-tier hosted pieces
never survive a host reboot; ssd-tier hosted pieces always do (the exemption
stays EXACTLY as before); the DRAM slot/byte ledger balances on every exit.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import yaml

from checkpointing.crossjob import CrossJobPeerStrategy, DonorRegistry, _DonorSlot
from run_scenario import run_arm

SCEN = Path(__file__).resolve().parent.parent / "scenarios"


def _run(name: str, tmp_path, arm: str = "ours_full"):
    sc = yaml.safe_load((SCEN / f"{name}.yaml").read_text())
    trace = tmp_path / f"{name}_{arm}.jsonl.gz"
    row = run_arm(sc, arm, 7, trace)
    events = [json.loads(l) for l in gzip.open(trace, "rt")]
    return row, events


def _spans(events, path, job="own0"):
    out = []
    for e in events:
        d = e.get("details") or {}
        if d.get("path") == path and e.get("job_id") == job:
            out.append((round(e["start"], 4), round(e["end"], 4),
                        round(e["end"] - e["start"], 4), e.get("iteration")))
    return sorted(out)


def test_l15_case_a_stripe_demote_drain_pencil_and_demoted_age_recovery(
        tmp_path) -> None:
    row, ev = _run("handcheck_l15", tmp_path)

    # exactly the two scheduled events fired
    assert row["failures"] == {"node": 1, "reboot": 1}

    # STRIPE at NIC_in rate, exact duration: 1.0 GB / 2.0 GB/s = 0.500 s
    waves = _spans(ev, "peer_dram")
    assert [w[2] for w in waves] == [0.5, 0.5, 0.5], waves
    # first wave lands 40.1 -> 40.6 (async landing, one iteration late,
    # +0.1 s capture; see the YAML header derivation)
    assert waves[0][:2] == (40.1, 40.6), waves[0]

    # DEMOTION at disk rate, exact duration: 1.0 GB / 1.0 GB/s = 1.000 s,
    # starting at the wave's end (background, host-local RES_DISK only)
    demotes = _spans(ev, "dram_demote")
    assert [d[2] for d in demotes] == [1.0, 1.0, 1.0], demotes
    assert demotes[0][:2] == (40.6, 41.6), demotes[0]
    for w, d in zip(waves, demotes):
        assert d[0] >= w[1] - 1e-9, (w, d)          # demote never precedes wave

    # L3 drain chains strictly AFTER the demotion (reads the donor SSD)
    drains = _spans(ev, "l3_drain")
    assert len(drains) == 3 and row["l3_drains"] == 3
    for d, dr in zip(demotes, drains):
        assert dr[0] >= d[1] - 1e-9, (d, dr)
    assert drains[0][:2] == (41.6, 42.6), drains[0]  # 1.0 GB / min(1,2,999)=1 s

    # host reboot AFTER demotion: the demoted piece SURVIVES the restart
    # (hosted-SSD exemption) and the owner's node loss recovers at the
    # DEMOTED-piece age — an ordinary L2 recovery of @3, loss = 5 - 3 = 2
    assert row["dram_piece_losses"] == 0
    assert row["dram_demote_aborted"] == 0
    assert row["recovery_source_tiers"]["crossjob_peer"] == 1
    assert row["staleness_by_tier"]["crossjob_peer"] == 2
    # (the host job keeps no checkpoints, so ITS reboot rolls it to 0 —
    # documented scaffolding, not part of the L1.5 claim)
    assert row["recovery_source_tiers"].get("initial_state", 0) == 1
    assert row["staleness_by_tier"]["initial_state"] == 5

    # completion pencil (own): 120 ideal + 20 node down + 2 lost iters (20 s)
    # + recovery fetch 0.5 + restore 0.1 + 3 captures x 0.1 = 160.9
    assert row["per_class_end_s"]["own"] == 160.9, row["per_class_end_s"]

    ideal, _ = _run("handcheck_l15", tmp_path, arm="ideal")
    assert ideal["per_class_end_s"]["own"] == 120.0


def test_l15_case_b_reboot_before_demotion_falls_to_l2_age(tmp_path) -> None:
    row, ev = _run("handcheck_l15b", tmp_path)

    assert row["failures"] == {"node": 1, "reboot": 1}

    # the stripe is UNCHANGED by the slow disk (never touches it): 0.500 s
    waves = _spans(ev, "peer_dram")
    assert all(w[2] == 0.5 for w in waves), waves
    assert waves[0][:2] == (40.1, 40.6), waves[0]

    # demotion at the slow disk: 1.0 / 0.05 = 20.000 s exact; @6's demotion
    # (70.7 -> would be 90.7) is cut by the host reboot at t=80 and ABORTS
    demotes = _spans(ev, "dram_demote")
    assert all(d[2] == 20.0 for d in demotes), demotes
    assert demotes[0][:2] == (40.6, 60.6), demotes[0]
    assert row["dram_demote_aborted"] == 1
    assert row["dram_piece_losses"] == 1        # the @6 piece died in DRAM

    # recovery FALLS TO THE L2 AGE: the demoted @3 piece survives the host
    # reboot (write-then-rename shadow; the @6 wave never replaced its file)
    assert row["recovery_source_tiers"]["crossjob_peer"] == 1
    assert row["staleness_by_tier"]["crossjob_peer"] == 5      # 8 - 3
    assert row["recovery_source_tiers"].get("initial_state", 0) == 1  # host, doc'd

    # completion pencil (own): 8 iters (80 s) + 0.2 captures + 20 down +
    # fetch 0.5 + restore 0.1 + re-execute 9 iters (90 s) + 0.2 captures = 191.0
    assert row["per_class_end_s"]["own"] == 191.0, row["per_class_end_s"]

    # no drains in this file (backstop owner_push, store_every unset): the slow
    # donor disk belongs to the demotion alone, keeping the 20 s window exact
    assert row["l3_drains"] == 0


# ---------------------------------------------------------------------------
# strategy-level volatility pins (no runtime; direct piece-store surgery)
# ---------------------------------------------------------------------------

def _bare(run_id: int = 990) -> CrossJobPeerStrategy:
    s = object.__new__(CrossJobPeerStrategy)
    s.peer_copies = {}
    s.dram_shadow = {}
    s.stats = __import__("collections").Counter()
    s.registry = DonorRegistry()
    s.backend = type("B", (), {"now": 100.0})()
    return s


def _slot(reg, name="don-cohort-0"):
    class W:  # minimal worker stand-in
        pass
    w = W(); w.name = name; w.failed = False
    reg.slots[name] = _DonorSlot(worker=w, job_id="don")
    return reg.slots[name]


def test_reboot_wipes_dram_tier_but_never_ssd_tier_hosted_pieces() -> None:
    """The spec's volatility table, at the store level: a host reboot kills
    dram-tier hosted pieces (DRAM wiped) and NEVER ssd-tier ones (the disk
    survives — the pre-L1.5 exemption, byte-for-byte)."""
    s = _bare()
    slot = _slot(s.registry)
    s.registry.dram_host_gb = 4.0
    s.registry._dram_take(slot, 1, 2.0)
    s.peer_copies[("job", 0, "don-cohort-0")] = {
        "iteration": 6, "size_gb": 2.0, "k": 1, "piece_idx": 0,
        "tier": "dram", "shards": 1, "hosted_at": 90.0}
    s.peer_copies[("job2", 0, "don-cohort-0")] = {
        "iteration": 4, "size_gb": 2.0, "k": 1, "piece_idx": 0}   # ssd-tier

    # reboot: only the dram piece is dropped (driver filters tier == dram)
    s.drop_hosted_piece(("job", 0, "don-cohort-0"), lost=True,
                        disk_survives=True)
    assert ("job", 0, "don-cohort-0") not in s.peer_copies
    assert ("job2", 0, "don-cohort-0") in s.peer_copies      # ssd survives
    # the DRAM ledger balanced: slot + bytes freed, residency + loss booked
    assert slot.dram_hosted == 0 and slot.dram_gb == 0.0
    assert s.stats["dram_pieces_lost"] == 1
    assert s.stats["dram_gb_seconds"] == 2.0 * 10.0          # 2 GB x (100-90) s


def test_reboot_restores_write_then_rename_shadow() -> None:
    """A dram piece that superseded a DEMOTED piece on the same donor never
    replaced its SSD file (write-then-rename): a reboot that wipes the dram
    piece RESTORES the old L2 piece; node/spot (disk gone) never do."""
    s = _bare()
    slot = _slot(s.registry)
    old = {"iteration": 3, "size_gb": 1.0, "k": 1, "piece_idx": 0}   # demoted
    key = ("job", 0, "don-cohort-0")
    s.dram_shadow[key] = old
    s.registry._dram_take(slot, 1, 1.0)
    s.peer_copies[key] = {"iteration": 6, "size_gb": 1.0, "k": 1,
                          "piece_idx": 0, "tier": "dram", "shards": 1,
                          "hosted_at": 99.0}

    s.drop_hosted_piece(key, lost=True, disk_survives=True)   # reboot
    assert s.peer_copies[key] is old                          # @3 restored
    assert key not in s.dram_shadow
    assert s.stats["dram_shadow_restored"] == 1

    # contrast: node/spot loss (disk_survives=False) destroys shadow AND piece
    s2 = _bare()
    slot2 = _slot(s2.registry)
    s2.dram_shadow[key] = dict(old)
    s2.registry._dram_take(slot2, 1, 1.0)
    s2.peer_copies[key] = {"iteration": 6, "size_gb": 1.0, "k": 1,
                           "piece_idx": 0, "tier": "dram", "shards": 1,
                           "hosted_at": 99.0}
    s2.drop_hosted_piece(key, lost=True, disk_survives=False)
    assert key not in s2.peer_copies and key not in s2.dram_shadow


def test_dram_reservation_refusal_protocol() -> None:
    """Capacity is enforced at RESERVATION with the same refusal protocol as
    the SSD slots: a donor whose single DRAM slot (or byte budget) is full
    refuses, and the refusal is counted under dram_full."""
    reg = DonorRegistry()
    slot = _slot(reg)
    reg.dram_host_slots = 1
    reg.dram_host_gb = 2.0

    got = reg.reserve(1, exclude_job="other", tier="dram", piece_gb=1.0)
    assert len(got) == 1 and slot.dram_hosted == 1 and slot.dram_gb == 1.0
    # slot budget exhausted -> refusal
    assert reg.reserve(1, exclude_job="other", tier="dram", piece_gb=0.5) == []
    assert reg.refusals["dram_full"] == 1
    reg.release_dram("don-cohort-0", 1.0, 1)
    # byte budget: a 2.5 GB piece never fits the 2.0 GB cap
    assert reg.reserve(1, exclude_job="other", tier="dram", piece_gb=2.5) == []
    assert reg.refusals["dram_full"] == 2
    # the SSD wave-slot counter was never touched by any of this
    assert slot.hosted == 0
