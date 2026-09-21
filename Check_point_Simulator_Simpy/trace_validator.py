"""Trace validator: machine-checked physical invariants over simulation traces.

Runs after any simulation (standalone or in CI) and re-verifies, directly on
the recorded events, that the run obeyed its own physics. It cannot prove the
model matches the real world (the hardware anchors do that); it proves every
reported run is internally consistent — and it catches any future engine
change that silently breaks conservation or capacity semantics.

Invariants (H = hard failure, W = warning):
  I1  H  sanity: end >= start, data_gb >= 0, no NaNs
  I2  H  per-stream average rate <= min physical cap along its path
         (a single shard stream can never beat its bottleneck; per-node caps
         scaled by details.represents for cohort workers)
  I3  H  effective_slowdown >= 1 - eps for capacity-mode transfers (the audit
         found impossible <1 values in pre-fix traces — never again)
  I4  H  capture duration == shard_gb / gpu_cpu_bandwidth (exact, per event)
  I5  H  donor hosting: concurrent shards on a donor <= 2 x represents
  I6  H  recovery causality: a peer/store recovery's checkpoint iteration was
         flushed durable for that (job, rank) BEFORE the recovery started
  I7  H  restore ordering: dram->GPU restore begins only after that rank's
         fetch (same recovery episode) has ended
  I8  H  flush iterations non-decreasing per (job, rank), EXCEPT where a restore
         of that job (at or below the flushed iteration) rolled it back in
         between — rollback semantics re-execute the destroyed iterations
  I9  W  node disk long-run: bytes into a node's disk within its busy union
         <= disk_w x busy time x (1+tol)  [uniform-rate approximation]
  I10 W  durable accounting: sum of peer/store flush bytes == report total
  I11 H  donor-drain L3 causality: a drained piece had a durable peer flush of
         that (job, rank, iteration) BEFORE the drain started; and each donor
         drain stream <= min(donor disk, donor nic, store single-stream ceiling)
  I12 H  piece-level stitch recovery causality: the stitched iteration was a
         durable peer wave for (job, rank) ending before the fetch started
  I13 H  aggregate object-store ingress conservation: store-bound capacity-mode
         checkpoint bytes within each connected busy interval fit within the
         shared store ingress/disk budget
  I14 H  reboot causality (reboot_failure_type_spec.md): an on-node local-SSD
         recovery (the disk-survives reboot path) recovers an iteration that was
         durably written to that (job, rank)'s OWN local SSD before it started
         (or made durable off-node). node/spot destroy the local disk and can
         never take this path; the hosted-shard / node-loss invariants EXEMPT
         reboot (a host restart destroys no disk).
  I15 H  L1.5 demotion conservation (l15_peer_dram_spec.md): every DRAM->SSD
         demote event forwards a piece a peer_dram wave landed on that donor
         for that (job, rank, iteration) BEFORE the demote started, with
         byte-identical size; demote stream rate <= donor disk cap.
  I16 H  L1.5 dram recovery causality: a recovery labeled crossjob_peer_dram /
         crossjob_peer_dram_stitch (or a stitch using dram pieces) recovers an
         iteration a peer_dram wave landed before it started, AND none of its
         named dram donors had COMPLETED the demotion of that piece before the
         recovery started (a demoted piece is ordinary L2 and must be labeled
         so). Note: "dram pieces never survive host reboot" is not trace-
         checkable (failures are not trace events); it is pinned by
         tests/test_l15.py at the driver level instead.
  I17 W  L1.5 dram slot capacity: concurrent resident dram pieces on a donor
         (wave landing -> demote end, or trace end if never demoted) <=
         model.dram_host_slots x represented ranks (warning: host losses free
         slots invisibly to the trace, so this reconstruction can overcount).
  I18 H  rack failure (rack_failure_spec.md): DRAM on a failed rack's nodes is
         DESTROYED at the event — no recovery may use a dark worker's /dram as
         a source inside its dark window (event -> the worker's own first
         post-event recovery, i.e. its relocation onto replacements; end of
         run/return for workers that never relocate).
  I19 H  rack failure: SSD on a failed rack is UNAVAILABLE from the event to
         end-of-run/return — never used as a recovery source while dark, and
         no flush may land a piece on a dark donor in that window.
  I20 H  destructive-residue node never returns: a worker whose nodes were ALL
         destroyed (rack_p_destroy residue) and that never relocates must not
         appear as a donor or recovery source from the event to trace end.
  I21 H  rack anti-affinity: a cross-job stripe's donors have pairwise-disjoint
         rack sets, none intersecting the flusher's racks (<=1 piece-group per
         rack, none in the owner's rack) — EXCEPT waves flagged rack_fallback
         (best-effort spread after infeasibility; the run row counts them).
         Relocated workers are rack-less from their event on. peer_parity
         waves are exempt (parity is out of the rack gates' scope).

Usage:
  python3 trace_validator.py TRACE.jsonl[.gz] --scenario scenarios/X.yaml
  python3 trace_validator.py TRACE... --scenario X.yaml --strict   (exit 1 on W)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml

TOL = 0.02          # 2% numerical tolerance on rates/durations
DISK_TOL = 0.05     # uniform-rate approximation tolerance for I9


def load_events(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def class_of(job_id: str, classes: dict) -> dict:
    for cname, spec in classes.items():
        if job_id.startswith(cname):
            return spec
    return {}


def rates_for(job_id: str, sc: dict) -> tuple[float, float]:
    spec = class_of(job_id or "", sc["classes"])
    r = spec.get("rates", {})
    cl = sc["cluster"]
    return (float(r.get("nic", cl["network_bandwidth_gbps"])),
            float(r.get("disk", cl["local_ssd_bandwidth_gbps"])))


def rates_duplex(job_id: str, sc: dict) -> tuple[float, float, float]:
    """(nic_in, nic_out, disk) with per-direction overrides (L1.5 profiles)."""
    spec = class_of(job_id or "", sc["classes"])
    r = spec.get("rates", {})
    nic, disk = rates_for(job_id, sc)
    return (float(r.get("nic_in", nic)), float(r.get("nic_out", nic)), disk)


def represented_ranks(node_name: str, classes: dict) -> int:
    """Return the physical ranks represented by a scenario cohort node.

    ``run_scenario`` distributes the remainder over the first cohorts.  Using
    only ``ranks // cohorts`` understates those cohorts' disk and hosted-shard
    capacity whenever the division is uneven.
    """
    job_id = node_name
    cohort_index = None
    if "-cohort-" in node_name:
        job_id, raw_index = node_name.rsplit("-cohort-", 1)
        try:
            cohort_index = int(raw_index)
        except ValueError:
            cohort_index = None
    elif "-rank-" in node_name:
        job_id = node_name.rsplit("-rank-", 1)[0]

    spec = class_of(job_id, classes)
    ranks = max(1, int(spec.get("ranks", 1)))
    cohorts = min(ranks, max(1, int(spec.get("cohorts", ranks))))
    base, extra = divmod(ranks, cohorts)
    if cohort_index is None or not 0 <= cohort_index < cohorts:
        return max(1, base)
    return base + (1 if cohort_index < extra else 0)


def validate(path: Path, sc: dict) -> dict:
    ev = load_events(path)
    ev.sort(key=lambda e: e["start"])
    gpu_cpu = float(sc["cluster"]["gpu_cpu_bandwidth_gbps"])
    fails: list[str] = []
    warns: list[str] = []
    def F(inv, msg):
        fails.append(f"{inv}: {msg}")

    def W(inv, msg):
        warns.append(f"{inv}: {msg}")

    def det(e): return e.get("details") or {}
    def dur(e): return e["end"] - e["start"]
    def m_of(e): return int(det(e).get("represents", 1))
    def node_job(name):        # "<job>-cohort-3" or "<job>-rank-3" -> job id
        for sep in ("-cohort-", "-rank-"):
            if sep in name:
                return name.split(sep)[0]
        return name

    flushes, recoveries, restores, captures = [], [], [], []
    drains, stitches = [], []
    demotes = []                       # L1.5 DRAM -> host-SSD demotions
    for e in ev:
        # I1
        if e["end"] < e["start"] or (e.get("data_gb") or 0) < 0 or \
           math.isnan(e["start"]) or math.isnan(e["end"]):
            F("I1", f"malformed event at t={e['start']}: {e.get('operation')}")
        op = e.get("operation") or ""
        p = det(e).get("path", "")
        if p.startswith("peer") or p == "injob_ssd":
            flushes.append(e)
        if p == "dram_demote":
            demotes.append(e)
        if "recovery" in op:
            recoveries.append(e)
        if "l3_drain" in op:
            drains.append(e)
        if "stitch_to_dram" in op:
            stitches.append(e)
        if "parity_reconstruct" in op:
            stitches.append(e)          # same causality check (I12)
        if op == "dram_to_gpu_restore":
            restores.append(e)
        if op == "checkpoint_stage_gpu_to_dram":
            captures.append(e)
        # I3
        slw = det(e).get("effective_slowdown")
        if det(e).get("capacity_mode") and slw is not None and slw < 1 - TOL:
            F("I3", f"impossible slowdown {slw:.3f} at t={e['start']:.1f} ({op})")
        # I4
        if op == "checkpoint_stage_gpu_to_dram" and e.get("data_gb"):
            want = e["data_gb"] / gpu_cpu
            if abs(dur(e) - want) > TOL * want + 1e-6:
                F("I4", f"capture {dur(e):.3f}s != {want:.3f}s at t={e['start']:.1f}")

    # I2: per-stream average rate vs bottleneck cap
    for e in flushes:
        d = det(e)
        donors = d.get("donors") or []
        if not donors or dur(e) <= 0:
            continue
        shards = d.get("shards_by_donor") or {x: 1 for x in donors}
        total_shards = sum(shards.values()) or 1
        s_nic, _ = rates_for(e.get("job_id"), sc)
        m = m_of(e)
        dram = d.get("path") == "peer_dram"
        for dn in donors:
            d_nic, d_disk = rates_for(node_job(dn), sc)
            frac = shards.get(dn, 1) / total_shards
            stream_bytes = (e.get("data_gb") or 0) * frac
            if dram:
                # L1.5: the piece lands in donor DRAM at min(NIC_in, DRAM) ~=
                # NIC_in — the donor disk is NOT on the write path.
                d_in, _d_out, _dd = rates_duplex(node_job(dn), sc)
                per_shard_cap = min(d_in, s_nic)
            else:
                per_shard_cap = min(d_nic, d_disk, s_nic)  # one shard = one node-pair
            cap = per_shard_cap * shards.get(dn, 1)     # bundle of `take` shards
            rate = stream_bytes / dur(e)
            if rate > cap * (1 + TOL):
                F("I2", f"stream to {dn} at {rate:.3f} GB/s > cap {cap:.3f} "
                        f"(t={e['start']:.1f}, m={m}{', dram' if dram else ''})")

    # I5: concurrent hosted shards per donor (SSD write slots; L1.5 dram waves
    # hold DRAM slots instead — budgeted separately by I17)
    times = defaultdict(list)
    for e in flushes:
        d = det(e)
        if d.get("path") == "peer_dram":
            continue
        shards = d.get("shards_by_donor") or {x: 1 for x in (d.get("donors") or [])}
        for dn, k in shards.items():
            times[dn].append((e["start"], k))
            times[dn].append((e["end"], -k))
    for dn, marks in times.items():
        rep = represented_ranks(dn, sc["classes"])
        cap = 2 * rep
        cur = 0
        for _t, delta in sorted(marks):
            cur += delta
            if cur > cap:
                F("I5", f"donor {dn} hosts {cur} shards > cap {cap}")
                break

    # ---- L1.5 peer-DRAM invariants (I15/I16/I17) ---------------------------
    # demote index: (job, rank, donor, iteration) -> [(start, end, data_gb, take)]
    demote_by_piece: dict[tuple, list] = defaultdict(list)
    for e in demotes:
        d = det(e)
        demote_by_piece[(e.get("job_id"), e.get("rank"), d.get("donor"),
                         e.get("iteration"))].append(
            (e["start"], e["end"], float(e.get("data_gb") or 0.0),
             int(d.get("piece_shards", 1))))
    # dram wave landing index: same key -> [(end_t, piece_bytes, take)]
    dram_piece_by_key: dict[tuple, list] = defaultdict(list)
    for e in flushes:
        d = det(e)
        if d.get("path") != "peer_dram":
            continue
        donors = d.get("donors") or []
        shards = d.get("shards_by_donor") or {x: 1 for x in donors}
        tot = sum(shards.values()) or 1
        for dn in donors:
            dram_piece_by_key[(e.get("job_id"), e.get("rank"), dn,
                               e.get("iteration"))].append(
                (e["end"], (e.get("data_gb") or 0.0) * shards.get(dn, 1) / tot,
                 int(shards.get(dn, 1))))

    # I15: demotion conservation — a demote forwards a piece a peer_dram wave
    # landed on that donor for that (job, rank, iteration) BEFORE the demote
    # started, byte-identical; and the demote stream can never beat the donor
    # disk (it is a host-local RES_DISK copy).
    for e in demotes:
        d = det(e)
        key = (e.get("job_id"), e.get("rank"), d.get("donor"),
               e.get("iteration"))
        waves = dram_piece_by_key.get(key, [])
        gb = float(e.get("data_gb") or 0.0)
        if not any(t_end <= e["start"] + 1e-6 and abs(pb - gb) <= 1e-6 + TOL * pb
                   for t_end, pb, _tk in waves):
            F("I15", f"demote of {key} ({gb:.4f} GB) at t={e['start']:.1f} has "
                     f"no earlier byte-identical peer_dram landing")
        _dn_nic, dn_disk = rates_for(node_job(d.get("donor") or ""), sc)
        take = int(d.get("piece_shards", 1))
        if dur(e) > 0 and gb / dur(e) > dn_disk * take * (1 + TOL):
            F("I15", f"demote stream from {d.get('donor')} at "
                     f"{gb / dur(e):.3f} GB/s > disk cap {dn_disk * take:.3f} "
                     f"(t={e['start']:.1f})")

    # I16: dram recovery causality. A recovery labeled crossjob_peer_dram(_stitch)
    # (or an L3 stitch using dram pieces) must (a) recover an iteration a
    # peer_dram wave landed before it started (I6 covers the durable part), and
    # (b) name only dram donors whose piece was NOT yet demoted when it started —
    # a demoted piece is ordinary L2 and must not carry the dram label.
    for e in recoveries:
        d = det(e)
        tier = d.get("checkpoint_source_tier")
        dram_donors = d.get("dram_donors") or []
        if tier in ("crossjob_peer_dram", "crossjob_peer_dram_stitch") and \
                not dram_donors:
            # source string carries the tiers when donors weren't detailed
            dram_donors = [s.split("/")[0] for s in (e.get("source") or "").split("+")
                           if s.endswith("/dram")]
        if tier not in ("crossjob_peer_dram", "crossjob_peer_dram_stitch") \
                and not dram_donors:
            continue
        it = e.get("iteration")
        for dn in dram_donors:
            key = (e.get("job_id"), e.get("rank"), dn, it)
            if not any(t_end <= e["start"] + 1e-6
                       for t_end, _pb, _tk in dram_piece_by_key.get(key, [])):
                F("I16", f"dram recovery of {key} at t={e['start']:.1f} has no "
                         f"earlier peer_dram landing on that donor")
            if any(d_end <= e["start"] + 1e-6
                   for _ds, d_end, _gb, _tk in demote_by_piece.get(key, [])):
                F("I16", f"recovery at t={e['start']:.1f} labels {dn} dram, but "
                         f"that piece finished demoting at iteration {it} — it "
                         f"is ordinary L2 (label lies about the tier)")

    # I17: dram slot capacity — resident pieces per donor <= model.
    # dram_host_slots x rep. A piece's residency ends at the FIRST of: its
    # demotion's end, the NEXT landing for the same (job, rank) on this donor
    # (supersede / write-then-rename frees the old buffer), or trace end.
    dram_slots_cap = int((sc.get("model") or {}).get("dram_host_slots", 1))
    trace_end = max((e["end"] for e in ev), default=0.0)
    by_jrd: dict[tuple, list] = defaultdict(list)   # (job,rank,donor) -> landings
    for (j, r, dn, it), waves in dram_piece_by_key.items():
        for land_t, _pb, take in waves:
            by_jrd[(j, r, dn)].append((land_t, it, take))
    residency = defaultdict(list)
    for (j, r, dn), landings in by_jrd.items():
        landings.sort()
        for i, (land_t, it, take) in enumerate(landings):
            dems = [d_end for d_start, d_end, _gb, _tk
                    in demote_by_piece.get((j, r, dn, it), [])
                    if d_start >= land_t - 1e-9]
            end_t = min(dems) if dems else trace_end
            if i + 1 < len(landings):               # superseded by the next wave
                end_t = min(end_t, landings[i + 1][0])
            residency[dn].append((land_t, take))
            residency[dn].append((max(end_t, land_t), -take))
    for dn, marks in residency.items():
        rep = represented_ranks(dn, sc["classes"])
        cap = dram_slots_cap * rep
        cur = 0
        for _t, delta in sorted(marks):
            cur += delta
            if cur > cap:
                # WARNING, not hard: a piece lost with its host frees the slot,
                # but failures are not trace events, so the reconstruction here
                # conservatively holds lost pieces resident until trace end.
                W("I17", f"donor {dn} holds {cur} reconstructed-resident dram "
                         f"pieces > dram slot cap {cap} (host losses can "
                         f"overcount — see comment)")
                break

    # I6: recovery causality (peer recoveries against durable flush history)
    durable_by_rank: dict[tuple, list] = defaultdict(list)   # (job,rank)->[(t,it)]
    for e in flushes:
        durable_by_rank[(e.get("job_id"), e.get("rank"))].append(
            (e["end"], e.get("iteration")))
    for e in recoveries:
        op6 = e.get("operation") or ""
        if "peers_to_dram" not in op6 and "injob_ssd_to_dram" not in op6:
            continue
        key = (e.get("job_id"), e.get("rank"))
        it = e.get("iteration")
        ok = any(t <= e["start"] + 1e-6 and fit == it
                 for t, fit in durable_by_rank.get(key, []))
        if not ok:
            F("I6", f"recovery of {key} it={it} at t={e['start']:.1f} "
                    f"has no earlier durable flush of that iteration")

    # I14: reboot causality — an on-node local-SSD recovery must recover an
    # iteration previously written to that (job, rank)'s OWN local SSD (or made
    # durable off-node) before the recovery started. This is the disk-survives
    # reboot path; node/spot wipe the local disk and can never reach it. The
    # local-SSD copy is written by a `checkpoint_dram_to_local_ssd_chunk` flush.
    local_ssd_by_rank: dict[tuple, list] = defaultdict(list)
    for e in ev:
        if (e.get("operation") or "") == "checkpoint_dram_to_local_ssd_chunk":
            local_ssd_by_rank[(e.get("job_id"), e.get("rank"))].append(
                (e["end"], e.get("iteration")))
    for e in recoveries:
        if (e.get("operation") or "") != "checkpoint_ssd_to_dram_recovery_chunk":
            continue
        key = (e.get("job_id"), e.get("rank"))
        it = e.get("iteration")
        ok = any(t <= e["start"] + 1e-6 and fit == it
                 for t, fit in local_ssd_by_rank.get(key, [])) or \
             any(t <= e["start"] + 1e-6 and fit == it
                 for t, fit in durable_by_rank.get(key, []))
        if not ok:
            F("I14", f"on-node SSD (reboot) recovery of {key} it={it} at "
                     f"t={e['start']:.1f} has no earlier durable local-SSD flush "
                     f"of that iteration")

    # I11: donor-drain L3 causality + per-piece rate cap. A drained L3 piece is
    # forwarded from a piece the donor was HOSTING, so a durable peer flush of
    # that (job, rank, iteration) must have ENDED before the drain started; and
    # each donor stream can never beat min(donor disk, donor nic, store single-
    # stream ceiling) — the demand the drain path declares.
    store_stream = float((sc.get("store") or {}).get("stream_gbps", 999.0))
    for e in drains:
        key = (e.get("job_id"), e.get("rank"))
        it = e.get("iteration")
        if not any(t <= e["start"] + 1e-6 and fit == it
                   for t, fit in durable_by_rank.get(key, [])):
            F("I11", f"L3 drain of {key} it={it} at t={e['start']:.1f} has no "
                     f"earlier durable peer flush of that iteration")
        pieces = det(e).get("drain_pieces") or {}
        if dur(e) > 0:
            for dn, pg in pieces.items():
                d_nic, d_disk = rates_for(node_job(dn), sc)
                cap = min(d_disk, d_nic, store_stream)
                rate = pg / dur(e)
                if rate > cap * (1 + TOL):
                    F("I11", f"drain stream from {dn} at {rate:.3f} GB/s > cap "
                             f"{cap:.3f} (t={e['start']:.1f})")

    # I12: piece-level stitch / XOR-parity recovery causality. A stitch or a
    # parity reconstruction reassembles a shard at ONE iteration from live peer
    # pieces (+ L3 pieces for stitch) — that iteration must have been a durable
    # peer wave for (job, rank) ending before the fetch started.
    for e in stitches:
        key = (e.get("job_id"), e.get("rank"))
        it = e.get("iteration")
        if not any(t <= e["start"] + 1e-6 and fit == it
                   for t, fit in durable_by_rank.get(key, [])):
            F("I12", f"stitch recovery of {key} it={it} at t={e['start']:.1f} "
                     f"has no earlier durable flush of that iteration")

    # I7: restore after fetch within the same episode (same rank, nearest fetch)
    fetch_end = defaultdict(list)
    for e in recoveries:
        fetch_end[(e.get("job_id"), e.get("rank"))].append(e["end"])
    for e in restores:
        key = (e.get("job_id"), e.get("rank"))
        cands = [t for t in fetch_end.get(key, []) if t <= e["start"] + 1e-6]
        tier = det(e).get("checkpoint_source_tier")
        if tier in ("crossjob_peer", "store", "crossjob_peer_l3_stitch",
                    "crossjob_peer_parity", "crossjob_peer_dram",
                    "crossjob_peer_dram_stitch", "injob_ssd_replica") and not cands \
                and fetch_end.get(key):
            F("I7", f"restore at t={e['start']:.1f} for {key} precedes its fetch")

    # I8: flush iteration monotonicity — EXCEPT across a rollback.
    # Rollback semantics (run_scenario, 2026-07-14): a restore rewinds the job to
    # the restored iteration and the destroyed work is RE-EXECUTED, so flushes
    # legitimately revisit iteration numbers already flushed. Whole-job eviction
    # rewinds every cohort to the OLDEST restored one, so the restore that
    # excuses a rank's backwards flush may belong to ANOTHER rank of that job.
    # A backwards flush is only a violation when no restore of that job, at or
    # below the flushed iteration, landed since the rank's previous flush.
    restore_by_job: dict = defaultdict(list)      # job -> [(end_t, restored_it)]
    for e in restores:
        restore_by_job[e.get("job_id")].append((e["end"], e.get("iteration") or 0))
    last_it: dict[tuple, float] = {}
    last_flush_t: dict[tuple, float] = {}
    for e in flushes:
        key = (e.get("job_id"), e.get("rank"))
        it = e.get("iteration") or 0
        if it < last_it.get(key, -1):
            prev = last_flush_t.get(key, -math.inf)
            if not any(prev <= t_end <= e["start"] + 1e-6 and r <= it
                       for t_end, r in restore_by_job.get(e.get("job_id"), ())):
                F("I8", f"flush iteration went backwards for {key} at "
                        f"t={e['start']:.1f} ({last_it[key]} -> {it}) with no "
                        f"intervening rollback")
        last_it[key] = it          # a rollback RESETS the baseline to re-execute
        last_flush_t[key] = e["start"]

    # I9 (warning): long-run disk feasibility per donor node. L1.5: dram waves
    # write NO donor disk at wave time — their disk bytes arrive later via the
    # demote events, which are accounted here instead.
    disk_bytes = defaultdict(float)
    busy = defaultdict(list)
    for e in flushes:
        d = det(e)
        if d.get("path") == "peer_dram":
            continue
        shards = d.get("shards_by_donor") or {x: 1 for x in (d.get("donors") or [])}
        tot = sum(shards.values()) or 1
        for dn, k in shards.items():
            disk_bytes[dn] += (e.get("data_gb") or 0) * k / tot
            busy[dn].append((e["start"], e["end"]))
    for e in demotes:
        dn = det(e).get("donor")
        if dn:
            disk_bytes[dn] += float(e.get("data_gb") or 0.0)
            busy[dn].append((e["start"], e["end"]))
    for dn, spans in busy.items():
        spans.sort()
        merged, (cs, ce) = [], spans[0]
        for s, x in spans[1:]:
            if s <= ce:
                ce = max(ce, x)
            else:
                merged.append((cs, ce))
                cs, ce = s, x
        merged.append((cs, ce))
        window = sum(x - s for s, x in merged)
        rep = represented_ranks(dn, sc["classes"])
        _, d_disk = rates_for(node_job(dn), sc)
        cap = d_disk * rep * window
        if window > 0 and disk_bytes[dn] > cap * (1 + DISK_TOL):
            W("I9", f"donor {dn}: {disk_bytes[dn]:.1f} GB in {window:.1f}s busy "
                    f"> disk budget {cap:.1f} GB")

    # I13: aggregate object-store ingress conservation.  A capacity-mode event
    # records all of its bytes inside [start, end], even though its instantaneous
    # fair share can change as other flows join and leave.  The trace therefore
    # cannot reconstruct every instantaneous allocation, but it *can* prove the
    # necessary conservation condition below: within each connected union of
    # store-write intervals, total bytes cannot exceed shared capacity x union
    # duration.  This catches aggregate oversubscription that per-stream I11
    # cannot see, without assuming each event held its average rate throughout.
    store = sc.get("store") or {}
    store_cap = min(float(store.get("in_gbps", 999.0)),
                    float(store.get("disk_gbps", 999.0)))
    store_writes = [
        e for e in ev
        if det(e).get("capacity_mode")
        and e.get("category") == "Checkpoint"
        and e.get("destination") in {"__store__", "object-store"}
        and (e.get("data_gb") or 0) > 0
        and dur(e) > 0
    ]
    if store_writes and store_cap > 0:
        component: list[dict] = []
        component_end = -math.inf

        def check_store_component(events: list[dict]) -> None:
            start = min(e["start"] for e in events)
            end = max(e["end"] for e in events)
            volume = sum(float(e.get("data_gb") or 0) for e in events)
            budget = store_cap * (end - start)
            if volume > budget * (1 + TOL) + 1e-6:
                F("I13", f"object-store ingress moved {volume:.1f} GB in "
                  f"{end - start:.3f}s busy > shared budget {budget:.1f} GB "
                  f"at {store_cap:.3f} GB/s (t={start:.1f}..{end:.1f})")

        for e in sorted(store_writes, key=lambda item: (item["start"], item["end"])):
            if component and e["start"] > component_end + 1e-9:
                check_store_component(component)
                component = []
            component.append(e)
            component_end = max(component_end, e["end"])
        if component:
            check_store_component(component)

    # ---- I18-I21: rack failure domains (rack_failure_spec.md, 2026-07-27) ----
    # Driven by the trace's own rack_topology + rack_failure events, so the
    # checks need no re-derivation of the packing. Node identity is welded to
    # the worker NAME: after a dark worker's first post-event recovery it runs
    # on REPLACEMENT nodes (relocated, rack-less), so its dark window closes
    # there; a worker that never relocates (finished job) stays dark to
    # end-of-run/return.
    topo_ev = next((e for e in ev
                    if (e.get("operation") or "") == "rack_topology"), None)
    rack_evs = [e for e in ev
                if (e.get("operation") or "") == "rack_failure"]
    if topo_ev is not None:
        td = det(topo_ev)
        rk_size = int(td["rack_size"])
        slices = {n: tuple(v) for n, v in (td.get("slices") or {}).items()}
        trace_end_all = max((e["end"] for e in ev), default=0.0)
        # earliest rack event that hit each worker => rack-less from then on
        reloc_t: dict[str, float] = {}
        for re_ in rack_evs:
            for wname in (det(re_).get("affected") or {}):
                reloc_t[wname] = min(reloc_t.get(wname, math.inf), re_["start"])

        def racks_at(name: str, t: float) -> frozenset:
            if reloc_t.get(name, math.inf) <= t + 1e-9:
                return frozenset()          # relocated: replacements, rack-less
            sl = slices.get(name)
            if not sl or sl[1] <= sl[0]:
                return frozenset()
            lo, hi = sl
            return frozenset(range(lo // rk_size, (hi - 1) // rk_size + 1))

        recovery_events = [e for e in ev if e.get("category") == "Recovery"]
        rec_by_worker: dict[tuple, list] = defaultdict(list)
        for e in recovery_events:
            rec_by_worker[(e.get("job_id"), e.get("rank"))].append(
                (e["start"], e["end"]))
        for v in rec_by_worker.values():
            v.sort()

        def cohort_rank(name: str) -> int | None:
            for sep in ("-cohort-", "-rank-"):
                if sep in name:
                    try:
                        return int(name.rsplit(sep, 1)[1])
                    except ValueError:
                        return None
            return None

        for re_ in sorted(rack_evs, key=lambda e: e["start"]):
            d = det(re_)
            t0 = re_["start"]
            outage_end = float(d.get("outage_end_s", math.inf))
            destroyed_by = d.get("destroyed_by_worker") or {}
            for wname, n_dark in (d.get("affected") or {}).items():
                job = node_job(wname)
                rank = cohort_rank(wname)
                rep = represented_ranks(wname, sc["classes"])
                if int(n_dark) < rep:
                    # partially-dark cohort: its SURVIVING nodes legitimately
                    # keep serving — node-level attribution is below the cohort
                    # model's resolution, so the source/donor checks apply only
                    # to fully-dark workers (racks align with whole workers in
                    # every rack scenario in the gates).
                    continue
                fully_destroyed = int(destroyed_by.get(wname, 0)) >= rep
                # relocation = the worker's first recovery episode after t0
                rel = next((end for (s, end) in rec_by_worker.get((job, rank), [])
                            if s >= t0 - 1e-9), None)
                if rel is not None:
                    dark_end = rel
                elif fully_destroyed:
                    dark_end = trace_end_all        # I20: never returns
                else:
                    dark_end = min(outage_end, trace_end_all)
                inv = "I20" if fully_destroyed and rel is None else "I19"
                # I18/I19: no recovery may read the dark node while dark
                for e in recovery_events:
                    if not (t0 - 1e-6 <= e["start"] < dark_end - 1e-6):
                        continue
                    src = e.get("source") or ""
                    parts = src.split("+")
                    if any(s == f"{wname}/dram" for s in parts):
                        F("I18", f"recovery at t={e['start']:.1f} reads "
                                 f"{wname}/dram inside the dark window "
                                 f"[{t0:.1f}, {dark_end:.1f}) — rack DRAM is "
                                 f"destroyed at the event")
                    if any(s == f"{wname}/ssd" for s in parts):
                        F(inv, f"recovery at t={e['start']:.1f} reads "
                               f"{wname}/ssd inside the dark window "
                               f"[{t0:.1f}, {dark_end:.1f})")
                # I19/I20: no flush may land a piece on the dark donor
                for e in flushes:
                    if not (t0 - 1e-6 <= e["start"] < dark_end - 1e-6):
                        continue
                    if wname in (det(e).get("donors") or []):
                        F(inv, f"flush at t={e['start']:.1f} lands a piece on "
                               f"dark donor {wname} inside "
                               f"[{t0:.1f}, {dark_end:.1f})")

        # I21: anti-affinity on cross-job stripes (fallback waves exempt).
        for e in flushes:
            d = det(e)
            if d.get("path") not in ("peer", "peer_partial", "peer_dram"):
                continue                        # injob_ssd/parity out of scope
            if d.get("rack_fallback"):
                continue                        # best-effort spread, counted
            donors = d.get("donors") or []
            t = e["start"]
            owner_racks = racks_at(e.get("node") or "", t)
            used: dict[int, str] = {}
            for dn in donors:
                racks = racks_at(dn, t)
                if racks & owner_racks:
                    F("I21", f"flush at t={t:.1f} puts a piece on {dn} in the "
                             f"owner's rack {sorted(racks & owner_racks)}")
                clash = [r for r in racks if r in used]
                if clash:
                    F("I21", f"flush at t={t:.1f} puts pieces on {used[clash[0]]}"
                             f" and {dn} in the same rack {clash[0]} without "
                             f"rack_fallback")
                for r in racks:
                    used[r] = dn

    return {"trace": str(path), "events": len(ev), "flushes": len(flushes),
            "recoveries": len(recoveries), "failures": fails, "warnings": warns}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--strict", action="store_true", help="warnings also fail")
    args = ap.parse_args()
    sc = yaml.safe_load(args.scenario.read_text())
    bad = 0
    for t in args.traces:
        r = validate(t, sc)
        status = "PASS" if not r["failures"] and not (args.strict and r["warnings"]) \
            else "FAIL"
        bad += status == "FAIL"
        print(f"[{status}] {t.name}: {r['events']} events, {r['flushes']} flushes, "
              f"{len(r['failures'])} failures, {len(r['warnings'])} warnings")
        for msg in r["failures"][:10]:
            print(f"    FAIL {msg}")
        for msg in r["warnings"][:5]:
            print(f"    warn {msg}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
