"""Cross-job peer-sharing scenario driver (docs/CROSSJOB_GAPS.md items 3, 5, 6).

Runs a multi-job config with the crossjob_peer strategy, adding what main.py doesn't have:
- per-job PHASE (start offset, seconds) — coordinated vs uncoordinated schedules
- controller POLICY input (the same policy.json the real peerd daemons consume:
  {job_id: {"kpeers": int, "phase_s": float}})
- contention METRICS out: per-job peer/fallback flush counts, refusal reasons,
  flush makespans -> printed + results/crossjob_summary.json

No changes to jobs/runtime.py: phase is a delay wrapper around runtime.run().

Usage:
  python3 run_crossjob.py --config configs/crossjob_smoke.toml
  python3 run_crossjob.py --config ... --policy policy.json
  python3 run_crossjob.py --config ... --random-phases 7   # seed; uncoordinated baseline
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import simpy

from checkpointing.capacity import CapacityRegistry, NodeCaps
from checkpointing.crossjob import STORE_NODE, CrossJobPeerStrategy, DonorRegistry
from checkpointing.group_policy import normalize_group_policy
from jobs import load_config
from jobs.failures import FailureController
from jobs.runtime import DistributedJobRuntime
from nodes.node import EventLogger
from simulation import SimPyBackend

RESULTS_DIR = Path("results")


class FlushSpanAccumulator:
    """Incrementally maintain mean realized persist duration per job.

    EventLogger is append-only. Keeping the previous scan position and adjusting
    a group's contribution when a later event extends its span produces the same
    answer as rescanning the complete trace at every controller epoch.
    """

    def __init__(self) -> None:
        self.scan_index = 0
        self.spans: dict[tuple[str | None, object], tuple[float, float]] = {}
        self.duration_totals: dict[str | None, float] = defaultdict(float)
        self.group_counts: dict[str | None, int] = defaultdict(int)

    def add(self, event) -> None:
        details = event.details or {}
        if (
            details.get("checkpoint_strategy") != "crossjob_peer"
            or "path" not in details
        ):
            return
        key = (event.job_id, details.get("checkpoint_group"))
        previous = self.spans.get(key)
        if previous is None:
            start, end = event.start, event.end
            self.group_counts[event.job_id] += 1
            self.duration_totals[event.job_id] += end - start
        else:
            old_start, old_end = previous
            start = min(old_start, event.start)
            end = max(old_end, event.end)
            self.duration_totals[event.job_id] += (
                (end - start) - (old_end - old_start)
            )
        self.spans[key] = (start, end)

    def means(self) -> dict[str, float]:
        return {
            str(job): self.duration_totals[job] / count
            for job, count in self.group_counts.items()
            if job is not None and count > 0
        }

    def update(self, events) -> dict[str, float]:
        event_count = len(events)
        for index in range(self.scan_index, event_count):
            self.add(events[index])
        self.scan_index = event_count
        return self.means()


def controller_epochs(backend, logger, epoch_s: float, period: float, actions: list):
    """Closed-loop Level-1 controller (sim form): every epoch, measure each job's
    REALIZED flush duration from the event log and re-pack slot offsets greedily
    (longest flusher first). One-shot slots pack by estimates; estimates are wrong
    (contention, size classes) and workloads churn — the closed loop corrects both."""
    accumulator = FlushSpanAccumulator()
    while True:
        yield backend.timeout(epoch_s)
        provider = getattr(logger, "realized_flush_durations", None)
        dur = provider() if provider is not None else accumulator.update(logger.events)
        if not dur:
            continue
        cursor = 0.0
        new_slots = {}
        for job in sorted(dur, key=lambda j: -dur[j]):
            new_slots[job] = round(cursor % period, 2)
            cursor += dur[job]
        CrossJobPeerStrategy.slot_by_job = new_slots
        actions.append({"t": round(backend.now, 1), "measured_s": {j: round(d, 2) for j, d in dur.items()},
                        "slots": new_slots, "overload_x": round(cursor / period, 2)})


def preemption_process(backend, controller, runtimes, *, every_s, job_prefix, rng):
    """Whole-job preemption (the failure mode DP replicas cannot survive): every
    `every_s`, evict ALL workers of one random matching job simultaneously with spot
    semantics (local copies lost). Uses the FailureController's own application path,
    so restart/recovery flows are identical to organic failures.

    Grace rule (livelock guard): a job is only eligible if every worker is healthy AND
    it has been healthy for at least half the preemption interval — otherwise repeated
    evictions can outpace recovery and the job never finishes (observed: sim time ran
    away past 7e5 s on a 2-job test with every_s < restart time)."""
    victims = [rt for rt in runtimes if rt.config.job_id.startswith(job_prefix)]
    if not victims:
        return
    grace = every_s / 2.0
    while True:
        yield backend.timeout(every_s)
        eligible = []
        for rt in victims:
            ws = list(rt.workers.values())
            if all(w.active and not w.failed for w in ws) and                all(backend.now - w.failed_until >= grace for w in ws):
                eligible.append(rt)
        if not eligible:
            continue
        rt = rng.choice(eligible)
        workers = [w for w in rt.workers.values() if w.active and not w.failed]
        for w in workers:
            controller._apply_failure(rt.config, w, failure_type="spot",
                                      batch_size=len(workers))


def run(config_path: Path, phases: dict[str, float], kpeers: dict[str, int],
        quiet: bool = True, async_persist: bool = True,
        slot_period: float | None = None,
        slot_by_job: dict[str, float] | None = None,
        controller_epoch: float | None = None,
        rates_by_job: dict[str, dict] | None = None,
        capacity: bool = False,
        store_mode: bool = False,
        store_caps: dict | None = None,
        preemption: dict | None = None,
        donor_groups: dict | None = None) -> dict:
    config = load_config(config_path)
    CrossJobPeerStrategy.kpeers_by_job = dict(kpeers)
    CrossJobPeerStrategy.async_persist = async_persist
    CrossJobPeerStrategy.slot_period = slot_period
    CrossJobPeerStrategy.slot_by_job = dict(slot_by_job or {})
    CrossJobPeerStrategy.rates_by_job = dict(rates_by_job or {})
    CrossJobPeerStrategy.capacity_mode = capacity
    CrossJobPeerStrategy.store_mode = store_mode
    CrossJobPeerStrategy.matching = "registry"      # default board; no leak from
    CrossJobPeerStrategy.spread_jobs = False        # a prior run_scenario arm
    CrossJobPeerStrategy.window_bias = False
    DonorRegistry.reset(0)
    registry = DonorRegistry.for_run(0)
    registry.allowed_donors_by_job = normalize_group_policy(
        donor_groups or {}
    )
    CapacityRegistry.reset(0)
    if store_caps:
        CrossJobPeerStrategy.store_stream_gbps = float(store_caps.get("stream", 999.0))
        CapacityRegistry.for_run(0).set_node(STORE_NODE, NodeCaps(
            nic_in=float(store_caps.get("in", 999.0)),
            nic_out=float(store_caps.get("out", 999.0)),
            disk_w=float(store_caps.get("disk", 999.0))))

    env = simpy.Environment()
    backend = SimPyBackend(env)
    logger = EventLogger()
    object_store = backend.resource(config.cluster.object_store_concurrency)

    def delayed(runtime, delay):
        if delay > 0:
            yield backend.timeout(delay)
        yield from runtime.run()

    node_offset = 0
    processes = []
    strategies = []
    runtimes = []
    for job in config.jobs:
        runtime = DistributedJobRuntime(
            run_id=0, config=job, simulator_config=config, logger=logger,
            node_offset=node_offset, object_store=object_store,
            verbose=not quiet, backend=backend,
        )
        runtimes.append(runtime)
        # register ALL workers as donors up front — jobs that rarely checkpoint must
        # still donate (lazy registration inside the strategy misses idle jobs)
        strat = runtime.checkpoint_strategy
        if isinstance(strat, CrossJobPeerStrategy):
            strategies.append(strat)
            for w in runtime.workers.values():
                strat.register_worker(w, job.job_id)
        processes.append(backend.process(delayed(runtime, phases.get(job.job_id, 0.0))))
        node_offset += job.rank_count
    controller_actions: list = []
    if controller_epoch and slot_period:
        backend.process(controller_epochs(backend, logger, controller_epoch,
                                          slot_period, controller_actions))
    # failure machinery — mirrors run_configured_jobs (jobs/runtime.py:932); without
    # this the driver silently runs failure-free regardless of [failures] settings
    if config.failures.probability_per_second > 0 or preemption:
        failure_controller = FailureController(
            run_id=0, settings=config.failures, logger=logger,
            rng=random.Random(config.simulation.seed),
            recovery_handlers={rt.config.job_id: rt.checkpoint_strategy
                               for rt in runtimes},
            backend=backend,
        )
        if config.failures.probability_per_second > 0:
            backend.process(failure_controller.monitor(
                ((rt.config, worker) for rt in runtimes
                 for worker in rt.workers.values())))
        if preemption:
            backend.process(preemption_process(
                backend, failure_controller, runtimes,
                every_s=float(preemption.get("every_s", 120.0)),
                job_prefix=str(preemption.get("job_prefix", "be")),
                rng=random.Random(int(preemption.get("seed", 1)) + config.simulation.seed)))
    env.run(until=backend.all_of(processes))
    # drain background flushes still in flight when training ends (async persist)
    pending = [p for s in strategies for p in s.background_flushes]
    if pending:
        env.run(until=backend.all_of(pending))

    RESULTS_DIR.mkdir(exist_ok=True)
    logger.write_jsonl(RESULTS_DIR / "crossjob_log.jsonl")

    # ---- metrics from the event log ----
    rank_paths = {}                                       # (job, grp, rank) -> path
    group_spans = defaultdict(list)                       # (job, grp) -> all chunk spans
    for e in logger.events:
        det = e.details or {}
        if det.get("checkpoint_strategy") != "crossjob_peer" or "path" not in det:
            continue
        job = e.job_id or (det.get("checkpoint_group") or "?").split("-iteration-")[0]
        grp = det.get("checkpoint_group")
        rank_paths[(job, grp, det.get("checkpoint_owner_rank"))] = det["path"]
        group_spans[(job, grp)].append((e.start, e.end))
    flush_stats = defaultdict(lambda: {"peer": 0, "fallback": 0, "makespans": []})
    for (job, grp, _rank), path in rank_paths.items():
        durable = path.startswith("peer") or path == "store"
        flush_stats[job]["peer" if durable else "fallback"] += 1
    for (job, grp), spans in group_spans.items():
        flush_stats[job]["makespans"].append(
            max(s[1] for s in spans) - min(s[0] for s in spans))

    reg = DonorRegistry.for_run(0)
    summary = {"config": str(config_path), "phases": phases, "kpeers": kpeers,
               "slots": {"period_s": slot_period, "by_job": dict(slot_by_job or {})},
               "controller_actions": controller_actions,
               "refusals": dict(reg.refusals), "sim_seconds": env.now, "jobs": {}}
    for job, st in sorted(flush_stats.items()):
        tot = st["peer"] + st["fallback"]
        mk = st["makespans"]
        summary["jobs"][job] = {
            "rank_flushes": tot, "peer": st["peer"], "fallback": st["fallback"],
            "peer_rate": round(st["peer"] / tot, 3) if tot else None,
            "flush_makespan_mean_s": round(sum(mk) / len(mk), 2) if mk else None,
            "flush_makespan_max_s": round(max(mk), 2) if mk else None,
        }
    (RESULTS_DIR / "crossjob_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--policy", type=Path, help="controller policy JSON (job_id -> kpeers/phase_s)")
    ap.add_argument("--random-phases", type=int, metavar="SEED",
                    help="uncoordinated baseline: uniform-random phases")
    ap.add_argument("--max-phase", type=float, default=60.0)
    ap.add_argument("--controller-epoch", type=float, default=0.0,
                    help="closed-loop: re-pack slots from measured flush durations every E sim-s")
    ap.add_argument("--capacity", action="store_true",
                    help="per-node duplex NIC + disk capacity model (max-min fair)")
    ap.add_argument("--store-mode", action="store_true",
                    help="persist to the shared store node instead of donors")
    args = ap.parse_args()

    phases: dict[str, float] = {}
    kpeers: dict[str, int] = {}
    slot_period = None
    slot_by_job: dict[str, float] = {}
    rates_by_job: dict[str, dict] = {}
    store_caps = None
    preemption = None
    donor_groups = {}
    if args.policy:
        pol = json.loads(args.policy.read_text())
        store_caps = pol.get("_store")
        preemption = pol.get("_preemption")
        donor_groups = pol.get("_groups", {})
        slots = pol.get("_slots") or {}
        slot_period = slots.get("period_s")
        slot_by_job = {k: float(v) for k, v in (slots.get("by_job") or {}).items()}
        for job_id, p in pol.items():
            if job_id.startswith("_") or job_id == "jobs":
                continue
            if isinstance(p, dict):
                phases[job_id] = float(p.get("phase_s", 0.0))
                if "kpeers" in p:
                    kpeers[job_id] = int(p["kpeers"])
                if "class_rates" in p:
                    rates_by_job[job_id] = dict(p["class_rates"])
    if args.random_phases is not None:
        rng = random.Random(args.random_phases)
        cfg = load_config(args.config)
        phases = {j.job_id: rng.uniform(0, args.max_phase) for j in cfg.jobs}

    summary = run(args.config, phases, kpeers,
                  slot_period=slot_period, slot_by_job=slot_by_job,
                  controller_epoch=args.controller_epoch or None,
                  rates_by_job=rates_by_job,
                  capacity=args.capacity, store_mode=args.store_mode,
                  store_caps=store_caps, preemption=preemption,
                  donor_groups=donor_groups)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
