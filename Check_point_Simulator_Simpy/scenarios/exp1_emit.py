#!/usr/bin/env python3
"""exp1_emit — turn a simulator event trace into the METRICS.md JSONL schema.

The hardware harness (checkpointing/realgpu_coefficient/{train_probe,ckpt_probe}.py)
and the simulator must produce the SAME files so `compare.py` can join them on
(arm, seed, job, iteration). This module writes the simulator side.

=============================================================================
PROVENANCE OF EVERY FIELD — directly emitted vs reconstructed
=============================================================================
DIRECTLY EMITTED BY THE SIMULATOR (one trace event per record, no inference)
  train_{job}_rank{r}.jsonl   <- Training/training_iteration events. These exist
      only because run_scenario.py gained an opt-in per-iteration record for this
      validation (`simulation.emit_iterations: true`, CohortJobRuntime.
      _record_iteration). Before that hook the simulator emitted ONLY coalesced
      `training_iterations` bars spanning many iterations, from which per-iteration
      dt_s CANNOT be recovered without assuming uniformity across the block — and
      the block is exactly where an async stripe stretches some iterations and not
      others. So this is a real addition, not a reconstruction.
        t_rel_s, iter, rank, dt_s, stall_s, compute_s, allreduce_s  -> exact
  ckpt_sender.jsonl op=capture  <- Checkpoint/checkpoint_stage_gpu_to_dram
        t_rel_s, bytes, dur_s                                       -> exact
  ckpt_sender.jsonl op=stripe   <- Checkpoint/checkpoint_dram_to_crossjob_peers_chunk
        t_rel_s, bytes, dur_s, peer(s)                              -> exact
  ckpt_sender.jsonl op=drain    <- checkpoint_donor_ssd_to_l3_drain_chunk
                                   or checkpoint_stage_dram_to_object_store
        t_rel_s, bytes, dur_s                                       -> exact

RECONSTRUCTED (derived here; NOT independent simulator output)
  ckpt_receiver.jsonl — the simulator has no donor-side receive event; the stripe
      is one transfer whose flow already traverses (donor:nic_in, donor:disk).
      Each receiver line is the donor's view of a sender stripe line: same
      interval, same bytes, recv_dt_s == the sender's dur_s. It therefore CANNOT
      be used to cross-check the sender (on hardware it can: the receiver measures
      recv separately from fsync).
  netcounters.jsonl — the simulator has no 1 Hz interface sampler and no notion of
      a named interface. Bytes are spread UNIFORMLY over each transfer's interval
      and attributed to the fabric the ARM DECLARES. On hardware this file is
      evidence about which wire carried the traffic; here it is a tautology (the
      arm's fabric is an input). Emitted for schema parity only. Marked
      `"reconstructed": true` on every line.
  tokens — the simulator has no token/batch notion; tokens_per_iter is copied from
      the scenario's `mirror.tokens_per_iter` (batch x seq_len from the hardware
      spec) and multiplied out. Constant per iteration, so it cannot disagree.
  phase — hardware writes "warmup"/"measure" (train_probe.py); METRICS.md §1
      documents "train"/"stalled_capture"/"waiting_recovery". Both are emitted:
      `phase` uses the harness vocabulary (that is what compare.py will join on)
      and `phase_metrics` uses METRICS.md's.

NOT EMITTED AT ALL (structural gaps, listed rather than faked)
  loss           — no numerical model; field present and null.
  gpu.jsonl      — no GPU utilization / memory / PCIe model.
  clock offsets  — the simulator has one exact clock; run.json reports zeros and
                   flags them, it is not a measurement.
  fsync_dt_s     — the donor disk is a max-min capacity resource inside the stripe
                   flow, not a separately timed fsync.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

TRAIN_OP = "training_iteration"
CAPTURE_OP = "checkpoint_stage_gpu_to_dram"
STRIPE_OPS = {
    "checkpoint_dram_to_crossjob_peers_chunk",
    "checkpoint_dram_to_local_ssd_chunk",
    "checkpoint_dram_to_injob_ssd_chunk",
}
DRAIN_OPS = {
    "checkpoint_donor_ssd_to_l3_drain_chunk",
    "checkpoint_stage_dram_to_object_store",
}

GB = 1e9            # the simulator's `data_gb` is decimal GB, matching the
                    # hardware's MB/s figures (91 MB/s, 5.2 GB/s were computed
                    # with decimal denominators).


def read_trace(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _eff_gbps(nbytes: float, dur_s: float) -> float | None:
    """METRICS.md `eff_gbps` is GIGABITS per second (its own example:
    2147483648 B in 16.4 s -> 1.05)."""
    if dur_s <= 0:
        return None
    return round(nbytes * 8.0 / 1e9 / dur_s, 6)


def emit(trace_path: Path, out_dir: Path, *, arm: str, seed: int,
         meta: dict) -> dict:
    """Write the METRICS.md files for one (arm, seed). Returns the run.json dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fabric = meta["arms"][arm]["fabric"]
    tokens_per_iter = int(meta.get("tokens_per_iter", 0))
    warmup = int(meta.get("warmup_iters", 0))
    job_of_class = {}          # job_id -> class label used on hardware (A / B)

    iters: dict[tuple[str, int], list[dict]] = defaultdict(list)
    sender_rows: list[dict] = []
    receiver_rows: list[dict] = []
    ckpt_ids: dict[tuple[str, int], int] = {}
    next_ckpt: dict[str, int] = defaultdict(lambda: 1)
    # (node, direction) -> list of (start, end, bytes) for the netcounter rebuild
    wire: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    horizon = 0.0

    for ev in read_trace(trace_path):
        cat, op = ev.get("category"), ev.get("operation")
        det = ev.get("details") or {}
        horizon = max(horizon, float(ev.get("end") or 0.0))
        job_id = ev.get("job_id")
        if cat == "Training" and op == TRAIN_OP:
            cls = det.get("class") or (job_id or "")[:-1]
            job_of_class[job_id] = cls
            rank = int(ev["rank"])
            it = int(ev["iteration"])
            stalled = float(det.get("stall_s", 0.0)) > 1e-9
            iters[(cls, rank)].append({
                "schema": "train_probe.v1",       # harness parity
                "record": "iter",
                "t_rel_s": round(float(ev["start"]), 6),
                "t_wall": round(float(ev["start"]), 6),   # sim t0 == 0
                "iter": it,
                "global_step": it - 1,            # harness counts from 0
                "job": cls,
                "rank": rank,
                "dt_s": round(float(det["dt_s"]), 9),
                "stall_s": round(float(det.get("stall_s", 0.0)), 9),
                "compute_s": round(float(det.get("compute_s", 0.0)), 9),
                "allreduce_s": round(float(det.get("allreduce_s", 0.0)), 9),
                "tokens": tokens_per_iter,
                "loss": None,                     # GAP: no numerical model
                "phase": "warmup" if it <= warmup else "measure",
                "phase_metrics": "stalled_capture" if stalled else "train",
                "source": "simulator",
            })
            continue
        if cat != "Checkpoint":
            continue
        cls = job_of_class.get(job_id) or (job_id or "")[:-1]
        it = int(ev.get("iteration") or 0)
        key = (job_id, it)
        if key not in ckpt_ids:
            ckpt_ids[key] = next_ckpt[job_id]
            next_ckpt[job_id] += 1
        cid = ckpt_ids[key]
        nbytes = float(ev.get("data_gb") or 0.0) * GB
        dur = float(ev.get("duration") or 0.0)
        t0, t1 = float(ev["start"]), float(ev["end"])
        if op == CAPTURE_OP:
            sender_rows.append({
                "t_rel_s": round(t0, 6), "job": cls, "ckpt_id": cid,
                "op": "capture", "bytes": int(nbytes), "dur_s": round(dur, 9),
                "src": "gpu", "dst": "host_dram", "rank": ev.get("rank"),
                "iteration": it, "source": "simulator"})
        elif op in STRIPE_OPS:
            donors = det.get("donors") or []
            sender_rows.append({
                "t_rel_s": round(t0, 6), "job": cls, "ckpt_id": cid,
                "op": "stripe", "bytes": int(nbytes), "dur_s": round(dur, 9),
                "peer": "+".join(donors) if donors else ev.get("destination"),
                "fabric": fabric, "eff_gbps": _eff_gbps(nbytes, dur),
                "rank": ev.get("rank"), "iteration": it,
                "path": det.get("path"), "source": "simulator",
                "_t_end": t1})
            wire[(ev.get("node"), "tx")].append((t0, t1, nbytes))
            share = nbytes / max(len(donors), 1)
            for d in donors:
                wire[(d, "rx")].append((t0, t1, share))
                receiver_rows.append({
                    "t_rel_s": round(t0, 6), "job": cls, "ckpt_id": cid,
                    "op": "receive", "peer": ev.get("node"),
                    "recv_bytes": int(share), "recv_dt_s": round(dur, 9),
                    "recv_MBps": (round(share / 1e6 / dur, 3) if dur > 0 else None),
                    "fsync_dt_s": None,          # GAP: disk is inside the flow
                    "fabric": fabric, "iteration": it,
                    "reconstructed": True,
                    "reconstructed_from": "sender stripe event (no donor-side "
                                          "receive event exists in the simulator)",
                    "source": "simulator"})
        elif op in DRAIN_OPS:
            sender_rows.append({
                "t_rel_s": round(t0, 6), "job": cls, "ckpt_id": cid,
                "op": "drain", "bytes": int(nbytes), "dur_s": round(dur, 9),
                "dst": "store", "fabric": fabric, "rank": ev.get("rank"),
                "iteration": it, "source": "simulator"})
            wire[(ev.get("node"), "tx")].append((t0, t1, nbytes))

    # ---- WAVE-LEVEL aggregate (RECONSTRUCTED grouping, exact arithmetic) ----
    # The hardware sends ONE 2 GiB shard from node30, so its ckpt_sender line is
    # already the whole wave. The mirror gives each of A's two ranks a shard, so
    # it emits one stripe line PER RANK; a per-stream eff_gbps is not comparable
    # to the hardware's. These fields restate each wave as the hardware would see
    # it: total bytes over the wall interval from the first stream's start to the
    # last stream's finish. Grouping is by (job, ckpt_id) — no interpolation.
    waves: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in sender_rows:
        if r["op"] == "stripe":
            waves[(r["job"], r["ckpt_id"])].append(r)
    wave_stats: dict[tuple[str, int], tuple[float, float, int]] = {}
    for key, group in waves.items():
        t0 = min(r["t_rel_s"] for r in group)
        t1 = max(r["_t_end"] for r in group)
        nb = sum(r["bytes"] for r in group)
        wave_stats[key] = (nb, t1 - t0, len(group))
        for r in group:
            r["wave_bytes"] = int(nb)
            r["wave_dur_s"] = round(t1 - t0, 9)
            r["wave_eff_gbps"] = _eff_gbps(nb, t1 - t0)
            r["wave_streams"] = len(group)
            r["wave_aggregate_reconstructed"] = True
    for r in sender_rows:
        r.pop("_t_end", None)

    # ---- write the per-iteration files -------------------------------------
    for (cls, rank), rows in sorted(iters.items()):
        rows.sort(key=lambda r: r["iter"])
        path = out_dir / f"train_{cls}_rank{rank}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    sender_rows.sort(key=lambda r: (r["t_rel_s"], r["op"]))
    receiver_rows.sort(key=lambda r: r["t_rel_s"])
    with (out_dir / "ckpt_sender.jsonl").open("w", encoding="utf-8") as fh:
        for r in sender_rows:
            fh.write(json.dumps(r) + "\n")
    with (out_dir / "ckpt_receiver.jsonl").open("w", encoding="utf-8") as fh:
        for r in receiver_rows:
            fh.write(json.dumps(r) + "\n")

    # ---- netcounters: RECONSTRUCTED, 1 Hz, uniform spreading ---------------
    with (out_dir / "netcounters.jsonl").open("w", encoding="utf-8") as fh:
        nodes = sorted({n for n, _ in wire})
        for node in nodes:
            cum = {"tx": 0.0, "rx": 0.0}
            spans = {d: wire.get((node, d), []) for d in ("tx", "rx")}
            for sec in range(int(math.ceil(horizon)) + 1):
                delta = {"tx": 0.0, "rx": 0.0}
                for d in ("tx", "rx"):
                    for (a, b, nb) in spans[d]:
                        if b <= sec or a >= sec + 1 or b <= a:
                            continue
                        overlap = min(b, sec + 1.0) - max(a, float(sec))
                        delta[d] += nb * overlap / (b - a)
                    cum[d] += delta[d]
                fh.write(json.dumps({
                    "t_rel_s": float(sec), "node": node, "iface": fabric,
                    "tx_bytes": int(cum["tx"]), "rx_bytes": int(cum["rx"]),
                    "tx_delta_gbps": round(delta["tx"] * 8 / 1e9, 6),
                    "rx_delta_gbps": round(delta["rx"] * 8 / 1e9, 6),
                    "reconstructed": True,
                    "reconstructed_from": "transfer events spread uniformly over "
                                          "their interval; fabric is the arm's "
                                          "declared input, not an observation",
                    "source": "simulator"}) + "\n")

    # ---- run.json -----------------------------------------------------------
    jobs_summary = {}
    for (cls, rank), rows in iters.items():
        if rank != 0:
            continue
        m = [r["dt_s"] for r in rows if r["phase"] == "measure"]
        if not m:
            continue
        s = sorted(m)
        jobs_summary[cls] = {
            "iters_completed": len(rows),
            "iters_measured": len(m),
            "mean_dt_s": round(statistics.mean(m), 9),
            "p50": round(s[len(s) // 2], 9),
            "p99": round(s[min(len(s) - 1, int(0.99 * len(s)))], 9),
            "tokens": len(rows) * tokens_per_iter,
        }
    stripes = [r for r in sender_rows if r["op"] == "stripe"]
    wv = list(wave_stats.values())
    run = {
        "arm": arm, "seed": seed, "source": "simulator",
        "duration_s": round(horizon, 6),
        "jobs": jobs_summary,
        "checkpoints": {
            # `count`/`mean_*` are WAVE-level (one per checkpoint event), which is
            # what the hardware's ckpt_sender.jsonl records. `*_per_stream` are the
            # simulator's raw per-rank lines.
            "count": len(wv),
            "bytes_total": int(sum(nb for nb, _d, _n in wv)),
            "mean_stripe_s": (round(statistics.mean(
                [d for _nb, d, _n in wv]), 6) if wv else None),
            "mean_eff_gbps": (round(statistics.mean(
                [nb * 8 / 1e9 / d for nb, d, _n in wv if d > 0]), 6)
                if wv else None),
            "streams_per_wave": (wv[0][2] if wv else None),
            "count_per_stream": len(stripes),
            "mean_stripe_s_per_stream": (round(statistics.mean(
                [r["dur_s"] for r in stripes]), 6) if stripes else None),
        },
        "fabric_bytes": {fabric: int(sum(r["bytes"] for r in stripes))},
        "clock_offsets_ms": {"__note__": "simulator has a single exact clock; "
                                         "this is not a measurement"},
        "emitter_notes": {
            "directly_emitted": ["train_*.jsonl", "ckpt_sender.jsonl"],
            "reconstructed": ["ckpt_receiver.jsonl", "netcounters.jsonl",
                              "tokens"],
            "not_modelled": ["loss", "gpu.jsonl", "clock offsets", "fsync_dt_s"],
        },
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=1))
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--meta", type=Path, required=True,
                    help="JSON written by exp1_arms.py (arm table + token count)")
    a = ap.parse_args()
    run = emit(a.trace, a.out_dir, arm=a.arm, seed=a.seed,
               meta=json.loads(a.meta.read_text()))
    print(json.dumps(run["jobs"], indent=1))


if __name__ == "__main__":
    main()
