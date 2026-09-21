#!/usr/bin/env python3
"""exp1_arms — generate + run the 8 simulator arms that mirror the hardware exp1.

One command runs the whole campaign:

    python3 scenarios/exp1_arms.py --run

which, for each of the 8 arms in checkpointing/realgpu_coefficient/exp1_scenario.yaml:
  1. writes  scenarios/exp1_mirror_build/exp1_mirror_<arm>.yaml         (scenario)
  2. writes  scenarios/exp1_mirror_build/exp1_mirror_<arm>_policy.json  (policy)
  3. runs    run_scenario.py over all seeds, with a per-run event trace
  4. converts each trace to the METRICS.md JSONL schema (scenarios/exp1_emit.py)
  5. computes alpha and gamma exactly as METRICS.md §7 defines them
     alpha = mean dt_s(job A, arm) / mean dt_s(job A, off) - 1
     gamma = mean dt_s(job B, arm) / mean dt_s(job B, off) - 1
     over the MEASURE phase only (warmup excluded), rank 0, pooled across seeds.

Only ONE thing varies between arms: the checkpoint-stream rate cap
(rates.nic_in / rates.nic_out on both classes). Training always rides the RoCE
all-reduce rate (rates.nic = 4.03 GB/s) in every arm, which is the dual-fabric
premise the hardware `tc` shaping implements.

=============================================================================
ARM RATES — provenance
=============================================================================
  off       no stream (A's cadence is pushed past the horizon)
  eth1g     0.091 GB/s   MEASURED effective 1 GbE checkpoint stream (91 MB/s)
  roce_tcp  1.57  GB/s   MEASURED NCCL busbw, same NIC, sockets (RDMA disabled).
                         This is a PROXY: busbw for a 2-node ring all-reduce
                         equals the per-link rate, but a single TCP stream is not
                         the same workload. exp1_scenario.yaml's
                         `peer_stream_roce_gbps` is still null (unmeasured).
  tcN       N/8   GB/s   the tbf shaped rate, converted gigabit -> GB/s.

  TCP CEILING (default ON, --no-tcp-ceiling to disable): tc25 (3.125 GB/s) and
  tc50 (6.25 GB/s) nominally exceed the measured 1.57 GB/s that this NIC actually
  achieves with kernel sockets. Two independent rate limiters compose as their
  min, so those arms are capped at 1.57. This is a modelling statement about how
  limiters compose, NOT a fit to any hardware result -- without it the simulator
  claims a SHAPED stripe is faster than the UNSHAPED roce_tcp arm, which is
  physically impossible. Both variants are reported; run with --no-tcp-ceiling
  to see the uncapped numbers.

=============================================================================
WHAT THE SIMULATOR CANNOT EXPRESS (see also the header of exp1_mirror.yaml)
=============================================================================
Printed by --gaps, and appended to the results JSON, so it travels with the data.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
TEMPLATE = HERE / "exp1_mirror.yaml"
BUILD = HERE / "exp1_mirror_build"
RESULTS = SIM / "results" / "exp1_mirror"

sys.path.insert(0, str(HERE))
import exp1_emit  # noqa: E402

# MEASURED constants carried over from exp1_scenario.yaml `cluster.measured`
# and the campaign log. Changing any of these changes a MODEL INPUT.
ROCE_TCP_CEILING_GBPS = 1.57     # NCCL busbw, same NIC, RDMA disabled
ETH1G_STREAM_GBPS = 0.091        # measured effective 1 GbE stream (91 MB/s)
CADENCE_S = 60.0                 # exp1_scenario jobs[A].checkpoint.cadence_s
TOKENS_PER_ITER = 4 * 1024       # train.batch x train.seq_len (hardware spec)
WARMUP_ITERS = 30                # run.warmup_iters

# (arm, nominal stream GB/s or None, fabric label, note)
ARMS = [
    ("off",      None,        "none",  "baseline iteration time for both jobs"),
    ("eth1g",    ETH1G_STREAM_GBPS, "eth1g",
     "physically separate 1 GbE card - the paper's premise (MEASURED rate)"),
    ("roce_tcp", ROCE_TCP_CEILING_GBPS, "roce",
     "same NIC as training, kernel sockets - the AWS EFA/ENA case"),
    ("tc1",      1 / 8,  "roce", "tbf 1 gbit/s on the RoCE NIC's kernel egress"),
    ("tc5",      5 / 8,  "roce", "tbf 5 gbit/s"),
    ("tc10",     10 / 8, "roce", "tbf 10 gbit/s"),
    ("tc25",     25 / 8, "roce", "tbf 25 gbit/s"),
    ("tc50",     50 / 8, "roce", "tbf 50 gbit/s"),
]

GAPS = [
    "GAMMA IS STRUCTURALLY ZERO. The donor's workers do enter the transfer's "
    "watched set (crossjob.py: watched = (owner, *donors)), so the donor job's "
    "checkpoint_network_tasks counter rises while it hosts. But that counter is "
    "consumed ONLY by CohortJobRuntime._allreduce_slowdown, i.e. it can stretch "
    "nothing but the donor job's ALL-REDUCE. Job B has ranks: 1, so its "
    "ar_base = 2(R-1)/R * g / nic = 0 exactly. There is no term to multiply. The "
    "simulator therefore reports gamma = 0.000 in every arm as a matter of code "
    "structure, not of physics. Hosting a stripe costs a donor's donor-side NIC "
    "and disk capacity in the max-min solver, but the solver's allocations never "
    "feed back into ANY job's compute clock.",

    "ALPHA IS COUNT-BASED, NOT INTENSITY-BASED. (Unchanged by CL-014, which "
    "scoped WHICH transfers count without touching HOW MUCH they cost.) While any "
    "checkpoint network transfer is in flight on a job's own workers, that job's "
    "all-reduce runs at "
    "1/(1+tasks). A 0.091 GB/s stripe and a 6.25 GB/s stripe both give exactly "
    "factor 2. Across arms the simulator's alpha varies ONLY through how long the "
    "stripe is in flight. It cannot express 'a faster stream hurts more per "
    "second', which is the effect the tc ladder was built to measure.",

    "TRAINING TRAFFIC IS NOT IN THE CAPACITY MODEL. All-reduce is an "
    "advance_work() timeout, not a Flow. It never registers against nic_in/nic_out, "
    "so a checkpoint stream and the training collective can never actually contend "
    "for a shared NIC in the max-min solver. HALF-CLOSED by protocol CL-014: with "
    "--fabric-aware the coupling is now SCOPED to the fabric (rates.ckpt_fabric vs "
    "rates.nic_fabric), so the eth1g arm (separate card) contributes exactly zero "
    "and no longer looks like roce_tcp. What is NOT closed, and is still the "
    "biggest structural gap: on a SHARED wire the term is still a count, not an "
    "allocation -- the collective is still not a Flow, still cannot saturate "
    "anything, and the shared-wire rungs still vary only through stripe duration "
    "(see gap 2). Closing that is CL-012.D S3, a separate change. Default is OFF: "
    "without --fabric-aware every arm is modelled exactly as at fda2360.",

    "NO PCIe, NO NUMA. METRICS.md §8 names both. nvidia-smi topo reports the A2 on "
    "NUMA node 1 with the RoCE NIC at SYS. The simulator has one gpu_cpu_bandwidth "
    "number and no interconnect. If the hardware shows alpha > 0 in the eth1g arm, "
    "the simulator has no term that could reproduce it.",

    "NO TCP DYNAMICS. Rates are fluid max-min: no slow start, no cwnd, no "
    "congestion collapse, no per-connection setup. The 2 GB stripes here take "
    "0.3-22 s, so slow start is a small but real bias the simulator does not carry. "
    "It also cannot show tc's burst/latency parameters (burst 32mbit, latency 50ms) "
    "doing anything.",

    "CAPTURE IS A HARD GPU LOCK. The simulator's GPU->DRAM capture takes "
    "worker.gpu, so training blocks for the FULL 1.0/5.2 = 0.192 s. On the hardware "
    "the capture runs in a SEPARATE process (ckpt_probe.py) with its own CUDA "
    "context; it contends for the GPU but does not serialize with the training step "
    "the way an exclusive resource does. Expect the simulator to over-attribute "
    "capture stall.",

    "STRIPE TOPOLOGY DIFFERS. Hardware streams ONE 2 GiB shard from node30 only. "
    "The mirror gives each of A's two ranks a 1.0 GB shard, so TWO senders stripe "
    "concurrently into B. The donor's nic_in is the binding cap in both cases, so "
    "the WAVE DURATION matches, but sender-side egress load per node differs by 2x "
    "and both of A's ranks take the capture stall (hardware: only rank 0).",

    "BYTE COUNT DIFFERS BY 7.4%. The harness SHARD constant is 2 GiB = 2.147 GB; "
    "the mirror moves 2.0 GB as instructed. Pass --shard-gb-per-rank 1.0737 to "
    "close it.",

    "NO DONOR fsync. The receiver's write+fsync is inside the stripe's flow as a "
    "disk resource, not a separately timed phase, so ckpt_receiver.jsonl carries "
    "fsync_dt_s: null and recv_dt_s identical to the sender's dur_s. The hardware "
    "sender waits on the fsync ack, so its stripe includes a term the simulator "
    "cannot isolate.",

    "DONOR NVMe WRITE BANDWIDTH IS UNMEASURED (exp1_scenario.yaml phase-0 left "
    "dram_to_nvme_gbps null). It is set NON-BINDING (8.0 GB/s) so no invented disk "
    "term enters. If the real NVMe binds below the shaped rate at tc25/tc50, the "
    "simulator under-predicts stripe time there.",

    "NO PER-ITERATION VARIANCE. Every simulated iteration of a job has identical "
    "compute time, so p99 stretch, jitter and straggler tails are degenerate: the "
    "dt_s distribution is a two-point mass (stalled vs not). METRICS.md's p99 "
    "stretch row is computable but carries no information here.",

    "NO LOSS CURVE, NO GPU TELEMETRY, NO CLOCK SKEW. loss is null, gpu.jsonl is "
    "not emitted, and run.json's clock_offsets_ms is a note rather than numbers.",

    "ALL-REDUCE COST IS 5.3% SHORT. The ring formula with measured busbw gives "
    "61.5 ms; hardware measures 87 ms of all-reduce (485 - 398). The residual is "
    "launch + small-message inefficiency. Reported, not absorbed: this shifts the "
    "simulator's absolute iteration time to 0.4595 s vs 0.485 s measured, though "
    "alpha/gamma are ratios and are only second-order sensitive to it.",
]


def build(shard_gb: float, tcp_ceiling: bool, drain: bool, iters: int | None,
          seeds: list[int] | None, fabric_aware: bool = False) -> dict:
    """Write one scenario + one policy per arm. Returns the meta dict.

    `fabric_aware` writes `model.fabric_aware_coupling: true` into every arm's
    scenario (protocol CL-014). DEFAULT OFF: with it off the generated arms are
    the fabric-blind ones the sealed predictions at fda2360 were produced under,
    and the per-arm `ckpt_fabric` labels below are inert text."""
    tpl = yaml.safe_load(TEMPLATE.read_text())
    BUILD.mkdir(parents=True, exist_ok=True)
    if iters:
        for spec in tpl["classes"].values():
            spec["iterations"] = iters
    if seeds:
        tpl["simulation"]["seeds"] = seeds

    a = tpl["classes"]["A"]
    # A's BASE iteration wall = measured compute + the ring all-reduce the runtime
    # computes from measured busbw and physical gradient bytes. Same arithmetic as
    # CohortJobRuntime.__init__, reproduced here only to turn the 60 s cadence into
    # an iteration count.
    ranks = int(a["ranks"])
    ar_base = 2 * (ranks - 1) / ranks * float(a["gradient_gb_per_rank"]) / float(
        a["rates"]["nic"])
    iter_wall = float(a["iteration_seconds"]) + ar_base
    every = max(1, round(CADENCE_S / iter_wall))

    meta = {
        "cadence_s": CADENCE_S, "checkpoint_every": every,
        "iteration_wall_s": round(iter_wall, 9), "ar_base_s": round(ar_base, 9),
        "shard_gb_per_rank": shard_gb, "tcp_ceiling_applied": tcp_ceiling,
        "fabric_aware_coupling": bool(fabric_aware),
        "tokens_per_iter": TOKENS_PER_ITER, "warmup_iters": WARMUP_ITERS,
        "seeds": list(tpl["simulation"]["seeds"]),
        "iterations": int(a["iterations"]), "arms": {},
    }

    for arm, nominal, fabric, note in ARMS:
        rate = nominal
        capped = False
        if rate is not None and tcp_ceiling and fabric == "roce":
            if rate > ROCE_TCP_CEILING_GBPS:
                rate, capped = ROCE_TCP_CEILING_GBPS, True
        sc = yaml.safe_load(TEMPLATE.read_text())
        if iters:
            for spec in sc["classes"].values():
                spec["iterations"] = iters
        if seeds:
            sc["simulation"]["seeds"] = seeds
        # `off` has no stream at all (A's cadence is pushed past the horizon), so
        # its rate is INERT — no flow is ever created. It is still written as a
        # finite number because the runtime divides by rates.nic when it computes
        # the ring all-reduce base; the value cannot influence the `off` arm.
        eff = rate if rate is not None else ROCE_TCP_CEILING_GBPS
        for cname in ("A", "B"):
            r = sc["classes"][cname]["rates"]
            r["nic_in"] = eff
            r["nic_out"] = eff
            # CL-014: which WIRE this arm's checkpoint stream rides. `eth1g` is
            # the separate Intel 1 GbE card (0000:1c); `roce` is the same
            # Mellanox card the collective uses (0000:5e) -- rates.nic_fabric in
            # the template. `off` streams nothing, so it declares no fabric at
            # all rather than pretending to. A LABEL, inert unless
            # model.fabric_aware_coupling is on.
            if fabric != "none":
                r["ckpt_fabric"] = fabric
            sc["classes"][cname]["checkpoint_gb_per_rank"] = shard_gb
        sc["classes"]["B"]["rates"]["nic"] = eff   # donor demand cap
        sc["arms"] = {arm: {"kpeers": True, "slots": False}}
        if fabric_aware:
            sc.setdefault("model", {})["fabric_aware_coupling"] = True
        sc["_mirror_arm"] = {"arm": arm, "stream_gbps": rate,
                             "nominal_gbps": nominal, "fabric": fabric,
                             "tcp_ceiling_applied": capped, "note": note}
        spath = BUILD / f"exp1_mirror_{arm}.yaml"
        spath.write_text(yaml.safe_dump(sc, sort_keys=False))

        never = int(sc["classes"]["A"]["iterations"]) + 1
        policy = {
            "name": arm,
            "scenario": str(spath),
            "solver": {"method": "hand_set_mirror_of_exp1_scenario.yaml",
                       "note": "not a solved policy: it enacts the HARDWARE's "
                               "hand-set configuration (A: 60 s cadence, k=1, "
                               "cross-job; B: no checkpointing)"},
            "flags": {"kpeers": True, "slots": False,
                      "backstop": "donor_drain" if drain else "owner_push"},
            "jobs": {
                "A0": {"class": "A",
                       "kpeers": 1,
                       "checkpoint_every": never if arm == "off" else every,
                       "capture_every": None,
                       "store_every": every if drain else None,
                       "placement": "crossjob",
                       "enacted_interval_s": (None if arm == "off"
                                              else round(every * iter_wall, 3))},
                "B0": {"class": "B",
                       "kpeers": 0,
                       "checkpoint_every": never,
                       "capture_every": None,
                       "store_every": None,
                       "placement": "local"},
            },
        }
        ppath = BUILD / f"exp1_mirror_{arm}_policy.json"
        ppath.write_text(json.dumps(policy, indent=1))
        meta["arms"][arm] = {
            "stream_gbps": rate, "nominal_gbps": nominal, "fabric": fabric,
            "tcp_ceiling_applied": capped, "note": note,
            "scenario": str(spath), "policy": str(ppath)}
    (BUILD / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def run(meta: dict) -> dict:
    """Run every arm, emit METRICS.md files, return {arm: {job: [dt means]}}."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    traces = RESULTS / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    metap = BUILD / "meta.json"
    means: dict[str, dict[str, list[float]]] = {}
    stripe_s: dict[str, list[float]] = {}
    for arm, info in meta["arms"].items():
        cmd = [sys.executable, "run_scenario.py",
               "--scenario", info["scenario"],
               "--policy", info["policy"],
               "--trace-dir", str(traces),
               "--out", str(RESULTS / f"scenario_exp1_mirror_{arm}.json"),
               "--progress-interval", "30"]
        print(f"--- arm {arm}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=SIM, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:], file=sys.stderr)
            raise SystemExit(f"arm {arm} failed")
        means[arm] = {}
        stripe_s[arm] = []
        for seed in meta["seeds"]:
            trace = traces / f"exp1_mirror_{arm}_{arm}_s{seed}.jsonl.gz"
            out_dir = RESULTS / arm / f"seed{seed}"
            run_json = exp1_emit.emit(trace, out_dir, arm=arm, seed=seed,
                                      meta={**meta, "arms": meta["arms"]})
            for job, js in run_json["jobs"].items():
                means[arm].setdefault(job, []).append(js["mean_dt_s"])
            ms = run_json["checkpoints"]["mean_stripe_s"]
            if ms:
                stripe_s[arm].append(ms)
    return {"means": means, "stripe_s": stripe_s}


def report(meta: dict, res: dict) -> dict:
    means, stripe_s = res["means"], res["stripe_s"]
    base_A = statistics.mean(means["off"]["A"])
    base_B = statistics.mean(means["off"]["B"])
    rows = []
    for arm in meta["arms"]:
        mA = statistics.mean(means[arm]["A"])
        mB = statistics.mean(means[arm]["B"])
        rows.append({
            "arm": arm,
            "stream_gbps": meta["arms"][arm]["stream_gbps"],
            "fabric": meta["arms"][arm]["fabric"],
            "tcp_ceiling_applied": meta["arms"][arm]["tcp_ceiling_applied"],
            "mean_dt_A_s": round(mA, 6),
            "mean_dt_B_s": round(mB, 6),
            "alpha": round(mA / base_A - 1.0, 6),
            "gamma": round(mB / base_B - 1.0, 6),
            "mean_stripe_s": (round(statistics.mean(stripe_s[arm]), 4)
                              if stripe_s[arm] else None),
        })
    w = f"{'arm':<9} {'stream GB/s':>11} {'stripe s':>9} " \
        f"{'mean dt A':>10} {'alpha':>9} {'mean dt B':>10} {'gamma':>9}"
    print("\n=== SIMULATOR alpha / gamma (pooled over seeds "
          f"{meta['seeds']}, measure phase only) ===")
    print(w)
    print("-" * len(w))
    for r in rows:
        sg = "-" if r["stream_gbps"] is None else f"{r['stream_gbps']:.3f}"
        ss = "-" if r["mean_stripe_s"] is None else f"{r['mean_stripe_s']:.3f}"
        print(f"{r['arm']:<9} {sg:>11} {ss:>9} "
              f"{r['mean_dt_A_s']:>10.5f} {r['alpha']:>+9.4f} "
              f"{r['mean_dt_B_s']:>10.5f} {r['gamma']:>+9.4f}")
    out = {"meta": meta, "rows": rows, "cannot_express": GAPS}
    (RESULTS / "alpha_gamma.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULTS / 'alpha_gamma.json'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="build AND run all 8 arms")
    ap.add_argument("--gaps", action="store_true",
                    help="print what the simulator cannot express and exit")
    ap.add_argument("--no-tcp-ceiling", action="store_true",
                    help="use the raw nominal tbf rates for tc25/tc50 even though "
                         "they exceed the measured 1.57 GB/s TCP ceiling")
    ap.add_argument("--drain", action="store_true",
                    help="enable the donor-drain L3 backstop (the hardware "
                         "harness does NOT drain; default off mirrors it)")
    ap.add_argument("--fabric-aware", action="store_true",
                    help="scope the checkpoint<->all-reduce coupling to the "
                         "FABRIC (protocol CL-014): a stream on a physically "
                         "separate card no longer stretches the collective. "
                         "DEFAULT OFF -- off reproduces the sealed fda2360 "
                         "predictions byte for byte")
    ap.add_argument("--shard-gb-per-rank", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--results-dir", default=None,
                    help="where the per-arm METRICS.md output and "
                         "alpha_gamma.json go. Default results/exp1_mirror, or "
                         "results/exp1_mirror_fabric_aware under --fabric-aware "
                         "-- so a fabric-aware run can never overwrite the "
                         "SEALED fda2360 prediction artifacts")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these arm ids (default: all 8). `off` is "
                         "always included -- alpha/gamma are ratios against it")
    a = ap.parse_args()
    global RESULTS
    RESULTS = (Path(a.results_dir) if a.results_dir else
               SIM / "results" / ("exp1_mirror_fabric_aware" if a.fabric_aware
                                  else "exp1_mirror"))
    if a.gaps:
        for i, g in enumerate(GAPS, 1):
            print(f"{i:2}. {g}\n")
        return
    meta = build(a.shard_gb_per_rank, not a.no_tcp_ceiling, a.drain,
                 a.iterations, a.seeds, fabric_aware=a.fabric_aware)
    print(f"built {len(meta['arms'])} arms in {BUILD}")
    print(f"cadence: every {meta['checkpoint_every']} iterations "
          f"= {meta['checkpoint_every'] * meta['iteration_wall_s']:.1f} s "
          f"(target {CADENCE_S} s); A iteration wall "
          f"{meta['iteration_wall_s']:.4f} s")
    if a.arms:
        keep = {"off", *a.arms}
        meta["arms"] = {k: v for k, v in meta["arms"].items() if k in keep}
    if a.run:
        report(meta, run(meta))


if __name__ == "__main__":
    main()
