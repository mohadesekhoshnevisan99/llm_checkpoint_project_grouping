"""Gate-4 hand-check for the donor-drain L3 backstop (goal 2026-07-19).

Drives scenarios/handcheck_drain.yaml's setup END-TO-END through the real
strategy: the owner (alpha) stripes k=2 to two donors (bravo, charlie); after
the wave the donors forward their pieces to L3; then ONE donor (charlie) dies
before the next wave and the owner suffers a node loss. Recovery MUST stitch
1 live peer piece + 1 L3 piece at the SAME iteration.

Pencil math (handcheck_drain.yaml, nic 2.0 / disk 1.0 / gpu_cpu 10.0 GB/s):
  capture      1.0 GB / 10.0                       = 0.100 s
  peer flush   0.5 GB piece / 1.0 GB/s per stream  = 0.500 s
  L3 drain     0.5 GB piece / 1.0 GB/s per stream  = 0.500 s
"""
from pathlib import Path

import simpy
import yaml

from checkpointing.capacity import CapacityRegistry, NodeCaps
from checkpointing.crossjob import (STORE_NODE, CrossJobPeerStrategy,
                                    DonorRegistry)
from nodes.node import EventLogger
from run_scenario import CohortWorker, build_config
from simulation import SimPyBackend

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "handcheck_drain.yaml"
WAVE_IT = 3


def _cluster(sc):
    from jobs.config import ClusterConfig
    cl = sc["cluster"]
    return ClusterConfig(
        node_count=int(cl["node_count"]), cpu_cores_per_node=1,
        cpu_memory_gb=3.75, gpus_per_node=1, gpu_memory_gb=80.0,
        gpu_cpu_bandwidth_gbps=float(cl["gpu_cpu_bandwidth_gbps"]),
        network_bandwidth_gbps=float(cl["network_bandwidth_gbps"]),
        communication_launch_seconds=0.01,
        local_ssd_bandwidth_gbps=float(cl["local_ssd_bandwidth_gbps"]),
        object_store_bandwidth_gbps=float(cl.get("object_store_bandwidth_gbps", 8.0)),
        object_store_concurrency=4)


def _worker(backend, job_id):
    return CohortWorker(
        job_id=job_id, rank=0, data_parallel_rank=0, pipeline_stage=0,
        physical_node=f"{job_id}-cohort-0", gpu=backend.resource(1),
        cpu=backend.priority_resource(1), represents=1,
        current_iteration=WAVE_IT)


def _build():
    sc = yaml.safe_load(SCENARIO.read_text())
    cluster = _cluster(sc)
    env = simpy.Environment()
    backend = SimPyBackend(env)
    logger = EventLogger()
    # deterministic class-attr config (donor-drain default backstop)
    DonorRegistry.reset(0)
    CapacityRegistry.reset(0)
    CrossJobPeerStrategy.kpeers_by_job = {"alpha0": 2, "bravo0": 2, "charlie0": 2}
    CrossJobPeerStrategy.async_persist = False
    CrossJobPeerStrategy.capacity_mode = True
    CrossJobPeerStrategy.engine = "event"
    CrossJobPeerStrategy.store_mode = False
    CrossJobPeerStrategy.slot_period = None
    CrossJobPeerStrategy.slot_by_job = {}
    CrossJobPeerStrategy.rates_by_job = {}
    CrossJobPeerStrategy.matching = "registry"
    CrossJobPeerStrategy.spread_jobs = False
    CrossJobPeerStrategy.window_bias = False
    CrossJobPeerStrategy.backstop = "donor_drain"
    CrossJobPeerStrategy.store_stream_gbps = float(sc["store"]["stream_gbps"])

    alpha = _worker(backend, "alpha0")
    bravo = _worker(backend, "bravo0")
    charlie = _worker(backend, "charlie0")
    cfg = build_config("alpha0", sc["classes"]["alpha"])
    strat = CrossJobPeerStrategy(
        run_id=0, cluster=cluster, logger=logger, object_store=backend.resource(4),
        workers={0: alpha}, backend=backend, paired=False, persist_to_ssd=False)
    st = sc["store"]
    strat.capacity.set_node(STORE_NODE, NodeCaps(
        nic_in=float(st["in_gbps"]), nic_out=float(st["out_gbps"]),
        disk_w=float(st["disk_gbps"])))
    for w, jid in ((alpha, "alpha0"), (bravo, "bravo0"), (charlie, "charlie0")):
        strat.register_worker(w, jid)
    return env, backend, logger, strat, cfg, alpha, bravo, charlie


def test_donor_drain_lands_two_l3_pieces():
    """After one k=2 peer wave, both donor pieces reach L3 at the wave's
    iteration (a COMPLETE L3 copy)."""
    env, backend, logger, strat, cfg, alpha, bravo, charlie = _build()
    backend.process(strat.checkpoint(alpha, cfg, iteration=WAVE_IT))
    env.run()
    if strat.background_flushes:                       # drain the L3 forwards
        env.run(until=backend.all_of(strat.background_flushes))

    peers = {k: v for k, v in strat.peer_copies.items() if v["iteration"] == WAVE_IT}
    assert len(peers) == 2, f"expected 2 peer pieces, got {peers}"
    l3 = {k: v for k, v in strat.l3_pieces.items() if v["iteration"] == WAVE_IT}
    assert len(l3) == 2, f"expected 2 L3 pieces, got {l3}"
    assert {k[2] for k in l3} == {0, 1}                # both piece indices landed
    assert all(v["k"] == 2 for v in l3.values())
    # pencil math: each 0.5 GB piece drained at 1.0 GB/s = 0.5 s (background)
    drain_ev = [e for e in logger.events if "l3_drain" in (e.operation or "")]
    assert drain_ev, "no l3_drain event recorded"
    for e in drain_ev:
        for _dn, pg in (e.details.get("drain_pieces") or {}).items():
            assert abs(pg - 0.5) < 1e-9
            assert abs((e.end - e.start) - pg / 1.0) < 0.02 + 1e-6


def test_stitch_recovery_one_peer_one_l3_same_iteration():
    """Gate 4: kill ONE donor after the drains land, then fail the owner. The
    recovery must stitch k-1 (=1) live peer piece + 1 L3 piece at the SAME
    iteration; the recovery event must name BOTH source tiers."""
    env, backend, logger, strat, cfg, alpha, bravo, charlie = _build()
    backend.process(strat.checkpoint(alpha, cfg, iteration=WAVE_IT))
    env.run()
    if strat.background_flushes:
        env.run(until=backend.all_of(strat.background_flushes))

    # ONE donor (charlie) dies before the next wave: its hosted piece is gone
    # (driver donor-loss path), but the L3 copy on the store survives.
    charlie.failed = True
    charlie.failure_generation += 1
    strat.drop_hosted_copies("charlie0-cohort-0")
    # the bravo peer piece must still be alive; charlie's is gone
    assert ("alpha0", 0, "bravo0-cohort-0") in strat.peer_copies, \
        "bravo peer piece vanished"
    assert ("alpha0", 0, "charlie0-cohort-0") not in strat.peer_copies, \
        "charlie peer piece not dropped"

    # owner node loss: local (dram/ssd) copies wiped; generation bumped once
    strat.drop_local_copies(0, fraction=1.0)
    alpha.failed = True
    alpha.failure_generation += 1

    result = {}

    def drive():
        result["ok"] = yield from strat.recover(alpha, cfg)
    backend.process(drive())
    env.run()

    assert result.get("ok") is True, "stitched recovery did not succeed"
    stitch_ev = [e for e in logger.events
                 if "stitch_to_dram" in (e.operation or "")]
    assert stitch_ev, "no stitch recovery event recorded"
    e = stitch_ev[0]
    tiers = e.details.get("stitch_sources")
    assert "crossjob_peer" in tiers and "l3" in tiers, \
        f"stitch did not use both sources: {tiers}"
    assert e.details.get("peer_pieces") == 1 and e.details.get("l3_pieces") == 1
    # ONE consistent iteration across the whole stitch
    assert e.iteration == WAVE_IT
    assert e.details.get("stitch_iteration") == WAVE_IT
    # the dram->GPU restore is tagged as a stitch and at the same iteration
    restore = [e for e in logger.events if e.operation == "dram_to_gpu_restore"]
    assert restore and restore[-1].details.get(
        "checkpoint_source_tier") == "crossjob_peer_l3_stitch"
    assert restore[-1].iteration == WAVE_IT


def _build_parity():
    """4-job harness (owner + 3 donors) with backstop == parity, k=2 -> 2 data
    + 1 XOR parity piece striped across 3 donors."""
    env, backend, logger, strat, cfg, alpha, bravo, charlie = _build()
    CrossJobPeerStrategy.backstop = "parity"
    strat.backstop = "parity"
    delta = _worker(backend, "delta0")
    strat.register_worker(delta, "delta0")
    return env, backend, logger, strat, cfg, alpha, bravo, charlie, delta


def test_parity_stripes_data_plus_parity_and_reconstructs_single_loss():
    """STRETCH gate: a parity wave stripes k=2 data + 1 parity across 3 donors;
    ONE donor loss is reconstructed PEER-SIDE (XOR), no L3 fetch at all."""
    env, backend, logger, strat, cfg, alpha, bravo, charlie, delta = _build_parity()
    backend.process(strat.checkpoint(alpha, cfg, iteration=WAVE_IT))
    env.run()
    if strat.background_flushes:
        env.run(until=backend.all_of(strat.background_flushes))

    pieces = {k: v for k, v in strat.peer_copies.items() if v["iteration"] == WAVE_IT}
    assert len(pieces) == 3, f"expected 3 pieces (2 data + 1 parity), got {pieces}"
    assert sum(1 for v in pieces.values() if v.get("parity")) == 1, "no parity piece"
    assert all(v["data_k"] == 2 for v in pieces.values())
    assert not strat.l3_pieces, "parity mode must not drain to L3"

    # ONE data donor dies; owner suffers node loss -> its data piece is
    # reconstructed by XOR of the surviving data piece + the parity piece.
    dead = "bravo0-cohort-0"
    strat.registry.slots[dead].worker.failed = True
    strat.registry.slots[dead].worker.failure_generation += 1
    strat.drop_hosted_copies(dead)
    strat.drop_local_copies(0, fraction=1.0)
    alpha.failed = True
    alpha.failure_generation += 1

    result = {}

    def drive():
        result["ok"] = yield from strat.recover(alpha, cfg)
    backend.process(drive())
    env.run()

    assert result.get("ok") is True, "parity reconstruction did not succeed"
    ev = [e for e in logger.events if "parity_reconstruct" in (e.operation or "")]
    assert ev, "no parity reconstruct recovery event"
    assert ev[0].iteration == WAVE_IT
    assert ev[0].details.get("reconstructed_pieces") == 1
    assert ev[0].details.get("data_k") == 2
    assert not any("l3_drain" in (e.operation or "") for e in logger.events)
    restore = [e for e in logger.events if e.operation == "dram_to_gpu_restore"]
    assert restore and restore[-1].details.get(
        "checkpoint_source_tier") == "crossjob_peer_parity"


def _build_cutoff(engine="event", slow="charlie0", slow_disk=0.1):
    """4-job harness (owner alpha + donors bravo/charlie/delta) with backstop ==
    parity_cutoff and ONE deliberately-slow donor, so its stream becomes the
    straggler. Rates are set BEFORE register_worker so the slow donor's capacity
    node carries the slow disk."""
    sc = yaml.safe_load(SCENARIO.read_text())
    cluster = _cluster(sc)
    env = simpy.Environment()
    backend = SimPyBackend(env)
    logger = EventLogger()
    DonorRegistry.reset(0)
    CapacityRegistry.reset(0)
    CrossJobPeerStrategy.kpeers_by_job = {
        "alpha0": 2, "bravo0": 2, "charlie0": 2, "delta0": 2}
    CrossJobPeerStrategy.async_persist = False
    CrossJobPeerStrategy.capacity_mode = True
    CrossJobPeerStrategy.engine = engine
    CrossJobPeerStrategy.store_mode = False
    CrossJobPeerStrategy.slot_period = None
    CrossJobPeerStrategy.slot_by_job = {}
    CrossJobPeerStrategy.matching = "registry"
    CrossJobPeerStrategy.spread_jobs = False
    CrossJobPeerStrategy.window_bias = False
    CrossJobPeerStrategy.backstop = "parity_cutoff"
    CrossJobPeerStrategy.store_stream_gbps = float(sc["store"]["stream_gbps"])
    # slow donor: its disk is the bottleneck -> slowest stream -> the straggler
    CrossJobPeerStrategy.rates_by_job = {slow: {"nic": 2.0, "disk": slow_disk}}

    alpha = _worker(backend, "alpha0")
    bravo = _worker(backend, "bravo0")
    charlie = _worker(backend, "charlie0")
    delta = _worker(backend, "delta0")
    cfg = build_config("alpha0", sc["classes"]["alpha"])
    strat = CrossJobPeerStrategy(
        run_id=0, cluster=cluster, logger=logger, object_store=backend.resource(4),
        workers={0: alpha}, backend=backend, paired=False, persist_to_ssd=False)
    st = sc["store"]
    strat.capacity.set_node(STORE_NODE, NodeCaps(
        nic_in=float(st["in_gbps"]), nic_out=float(st["out_gbps"]),
        disk_w=float(st["disk_gbps"])))
    for w, jid in ((alpha, "alpha0"), (bravo, "bravo0"),
                   (charlie, "charlie0"), (delta, "delta0")):
        strat.register_worker(w, jid)
    return env, backend, logger, strat, cfg, alpha, bravo, charlie, delta


def _cutoff_check(engine):
    """Choreograph a k=2 parity_cutoff wave with charlie slow. Pencil math
    (nic 2.0, alpha shard 1.0 -> 3 pieces of 0.5 GB; bravo/delta disk 1.0,
    charlie disk 0.1):
      phase 1 (all 3): alpha NIC-out 2.0 shared; charlie caps at 0.1, bravo &
        delta take 0.95 each -> land their 0.5 GB at 0.5/0.95 = 0.5263 s (CUTOFF,
        2 of 3). charlie has sent 0.0526 GB, 0.4474 GB left.
      phase 2 (charlie alone): 0.4474 / 0.1 = 4.4737 s -> lands at 5.000 s.
    So the wave RETURNS at 2 pieces (~0.53 s) and the straggler lands ~5.0 s later."""
    env, backend, logger, strat, cfg, alpha, bravo, charlie, delta = \
        _build_cutoff(engine)
    result = {}

    def drive():
        # checkpoint() returns None (it doesn't propagate the persist result), so
        # capture the peer state the instant the FOREGROUND wave RETURNED (cutoff):
        # exactly the 2 fast pieces are recoverable state; charlie still in flight
        yield from strat.checkpoint(alpha, cfg, iteration=WAVE_IT)
        result["peers_at_cutoff"] = {
            k[2] for k, v in strat.peer_copies.items() if v["iteration"] == WAVE_IT}
    backend.process(drive())
    env.run()                                  # drains the background straggler too
    if strat.background_flushes:
        env.run(until=backend.all_of(strat.background_flushes))

    # the wave completed at the k-th (=2nd) piece — charlie NOT among them
    assert result["peers_at_cutoff"] == {"bravo0-cohort-0", "delta0-cohort-0"}, \
        f"cutoff did not complete at the 2 fast pieces: {result['peers_at_cutoff']}"

    cutoff_ev = [e for e in logger.events
                 if (e.details or {}).get("path") == "peer_parity"]
    strag_ev = [e for e in logger.events
                if (e.details or {}).get("path") == "peer_parity_straggler"]
    assert len(cutoff_ev) == 1, f"expected 1 cutoff wave event, got {len(cutoff_ev)}"
    assert len(strag_ev) == 1, f"expected 1 straggler event, got {len(strag_ev)}"
    ce, se = cutoff_ev[0], strag_ev[0]
    assert ce.details["cutoff"] is True
    assert set(ce.details["donors"]) == {"bravo0-cohort-0", "delta0-cohort-0"}
    assert ce.details["straggler"] == "charlie0-cohort-0"
    assert set(se.details["donors"]) == {"charlie0-cohort-0"}
    # pencil math: cutoff at ~0.526 s, straggler lands ~5.0 s after the wave start
    assert abs((ce.end - ce.start) - 0.5263) < 0.02, \
        f"cutoff duration {ce.end - ce.start:.4f}s != 0.5263s"
    assert abs((se.end - se.start) - 5.0000) < 0.05, \
        f"straggler duration {se.end - se.start:.4f}s != 5.000s"
    assert (se.end - se.start) > (ce.end - ce.start) + 4.0, \
        "straggler did not land clearly later than the cutoff"
    # the straggler's piece became recoverable state after it landed (n+1 = 3)
    landed = {k[2] for k, v in strat.peer_copies.items() if v["iteration"] == WAVE_IT}
    assert landed == {"bravo0-cohort-0", "charlie0-cohort-0", "delta0-cohort-0"}
    assert not strat.l3_pieces, "parity_cutoff must not drain to L3"


def test_cutoff_completes_at_k_and_straggler_lands_later_event():
    _cutoff_check("event")


def test_cutoff_completes_at_k_and_straggler_lands_later_polling():
    _cutoff_check("polling")


def test_cutoff_recovers_from_two_landed_pieces_before_straggler():
    """The cutoff state (2 of 3 pieces) is ALREADY single-shard recoverable: with
    the slow charlie piece still in flight, kill a landed data donor's... no — the
    2 landed = bravo(data) + delta(parity); reconstruct charlie's data by XOR.
    Fail the owner right after cutoff and recover from the 2 landed pieces only."""
    env, backend, logger, strat, cfg, alpha, bravo, charlie, delta = \
        _build_cutoff("event")

    def drive():
        yield from strat.checkpoint(alpha, cfg, iteration=WAVE_IT)
    backend.process(drive())
    env.run(until=backend.timeout(1.0))        # past cutoff (~0.63 s), before 5 s

    at_cut = {k[2] for k, v in strat.peer_copies.items() if v["iteration"] == WAVE_IT}
    assert at_cut == {"bravo0-cohort-0", "delta0-cohort-0"}, \
        f"straggler already landed or cutoff wrong: {at_cut}"

    # owner node loss now; recovery must XOR-reconstruct from the 2 landed pieces
    strat.drop_local_copies(0, fraction=1.0)
    alpha.failed = True
    alpha.failure_generation += 1
    result = {}

    def recover():
        result["ok"] = yield from strat.recover(alpha, cfg)
    backend.process(recover())
    env.run()

    assert result.get("ok") is True, "recovery from the cutoff (2-piece) state failed"
    ev = [e for e in logger.events if "parity_reconstruct" in (e.operation or "")]
    assert ev and ev[0].details.get("data_k") == 2
    assert ev[0].iteration == WAVE_IT


def test_owner_push_backstop_still_selectable():
    """The legacy owner_push backstop stays selectable: no donor drains fire,
    and the owner uploads the whole shard to the store instead."""
    env, backend, logger, strat, cfg, alpha, bravo, charlie = _build()
    strat.backstop = "owner_push"
    backend.process(strat.checkpoint(alpha, cfg, iteration=WAVE_IT))
    env.run()
    if strat.background_flushes:
        env.run(until=backend.all_of(strat.background_flushes))
    # peer flush happened, but NO donor-drain pieces under owner_push
    assert any(v["iteration"] == WAVE_IT for v in strat.peer_copies.values())
    assert not strat.l3_pieces
    assert not any("l3_drain" in (e.operation or "") for e in logger.events)
