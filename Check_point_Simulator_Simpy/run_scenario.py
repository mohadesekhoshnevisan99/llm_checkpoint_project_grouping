"""YAML-scenario driver: real-width multi-tenant checkpoint simulation.

The whole experiment is defined in ONE reviewable YAML file — job classes at their
REAL 2026 widths (ranks per job), per-rank checkpoint shards, measured per-node rates,
failure rates, controller settings, store provisioning, arms — so the workload being
simulated is explicit and auditable BEFORE anything runs (scenarios/*.yaml).

Wide jobs run at COHORT granularity: a job of R ranks is represented by C cohort
workers, each standing for m = R/C ranks (worker.represents). The unchanged
CrossJobPeerStrategy interface handles multiplicity (crossjob.py: m-fold capacity
caps, aggregate bytes, donor grants in shards), so per-rank transfer physics stay
identical to the hardware-anchored runs while cross-job contention carries the
physically-correct aggregate demand. Per-rank simulation of these widths is
intractable (measured: 623 s wall for 9.5 k ranks, superlinear).

Failure semantics (DDP): one node failure stalls the WHOLE job; recovery restores
that node's per-rank shard from newest of donor/store/local (process failures keep
DRAM; node/spot lose local copies). Whole-job preemption refetches every cohort's
full state (size_scale=m). Losses are counted per restore against the restored
checkpoint's iteration, mirroring run_final73.analyze().

Usage:
  python3 run_scenario.py --scenario scenarios/realwidth73.yaml            # all arms
  python3 run_scenario.py --scenario ... --arm ours_full --seed 7
Author: Sam's agent — new module only; engine and student runtime untouched.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import gzip
import heapq
import json
import math
import random
import statistics
import sys
import threading
import time
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import simpy
import yaml

from checkpointing import baselines
from checkpointing import parallelism    # parallelism-aware shard calculator (D2)
from checkpointing.base import train_fabric_of
from checkpointing.capacity import CapacityRegistry, NodeCaps
from checkpointing.crossjob import STORE_NODE, CrossJobPeerStrategy, DonorRegistry
from jobs.config import ClusterConfig, FailureSettings, JobConfig
from jobs.runtime import JobWorker
from nodes.node import EventLogger, SimulationEvent
from run_crossjob import FlushSpanAccumulator, controller_epochs
from simulation import SimPyBackend
from simulation.progress import Slowdown, advance_work

RESULTS_DIR = Path("results")

# Stage 4 hybrid placement: in-job DRAM replication degree (Gemini's paper value;
# a replica on ONE other node of the job survives a single-node loss, which is the
# common failure — cross-job striping's k is a different, donor-supply quantity).
INJOB_DEGREE = 2


class BatchedScenarioEventLogger(EventLogger):
    """Bounded-memory scenario logger with externally merged trace chunks."""

    def __init__(
        self,
        *,
        trace_path: Path | None,
        batch_size: int = 50_000,
    ) -> None:
        super().__init__()
        self.trace_path = trace_path
        self.batch_size = max(1, int(batch_size))
        self.total_event_count = 0
        self.checkpoint_paths: Counter = Counter()
        self._flush_spans = FlushSpanAccumulator()
        self._temporary = (
            tempfile.TemporaryDirectory(prefix="scenario-trace-")
            if trace_path is not None
            else None
        )
        self._chunks: list[Path] = []

    def _event_recorded(self, event: SimulationEvent) -> None:
        self.total_event_count += 1
        self._flush_spans.add(event)
        details = event.details or {}
        if (
            details.get("checkpoint_strategy") in baselines.STRATEGY_NAMES
            and "path" in details
            and details.get("path") not in ("l3_drain", "dram_demote")
        ):
            self.checkpoint_paths[str(details["path"])] += details.get(
                "represents", 1
            )
        if len(self.events) >= self.batch_size:
            self._flush_batch()

    def realized_flush_durations(self) -> dict[str, float]:
        return self._flush_spans.means()

    def _flush_batch(self) -> None:
        if not self.events:
            return
        if self._temporary is not None:
            rows = sorted(
                self.events,
                key=lambda event: (event.start, event.end, event.event_id),
            )
            chunk = Path(self._temporary.name) / f"chunk-{len(self._chunks):06d}.jsonl"
            with chunk.open("w", encoding="utf-8") as handle:
                for event in rows:
                    handle.write(json.dumps(dataclasses.asdict(event), sort_keys=True))
                    handle.write("\n")
            self._chunks.append(chunk)
        self.events.clear()

    @staticmethod
    def _chunk_rows(path: Path):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield (
                    float(row["start"]),
                    float(row["end"]),
                    int(row["event_id"]),
                    line,
                )

    def finalize_trace(self) -> Path | None:
        self._flush_batch()
        if self.trace_path is None:
            return None
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if self.trace_path.suffix == ".gz" else open
        iterators = [self._chunk_rows(path) for path in self._chunks]
        with opener(self.trace_path, "wt", encoding="utf-8") as output:
            for _start, _end, _event_id, line in heapq.merge(*iterators):
                output.write(line)
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        return self.trace_path


def run_until_complete_or_cap(env, backend, processes, sim_cap: float) -> bool:
    """Run until every job completes or the absolute simulated-time cap wins."""

    completion = backend.all_of(processes)
    cap_event = backend.timeout(max(0.0, sim_cap - backend.now))
    env.run(until=(completion | cap_event))
    return completion.triggered


@dataclass
class CohortWorker(JobWorker):
    represents: int = 1          # ranks this worker stands for

    @property
    def name(self) -> str:
        return f"{self.job_id}-cohort-{self.rank}"


@dataclass(frozen=True)
class ScheduledFailure:
    """One deterministic, progress-aligned failure from a scenario file."""

    event_id: str
    job_id: str
    cohort: int
    after_iteration: int
    failure_type: str


@dataclass(frozen=True)
class JobArrival:
    """One concrete job admitted to a scenario at an absolute simulated time."""

    job_id: str
    class_name: str
    arrival_s: float
    source: str
    batch: int


def normalize_scheduled_failures(
    scenario: dict,
    jobs: list[tuple[str, dict]],
) -> dict[str, list[ScheduledFailure]]:
    """Validate and group deterministic failures by exact generated job ID."""
    grouped = {job_id: [] for job_id, _spec in jobs}
    job_specs = dict(jobs)
    raw_events = scenario.get("scheduled_failures", [])
    if not isinstance(raw_events, list):
        raise ValueError("scheduled_failures must be a list")

    seen_ids: set[str] = set()
    seen_boundaries: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_events, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"scheduled_failures[{index}] must be a mapping")
        event_id = str(raw.get("id", f"scheduled-{index}"))
        job_id = str(raw.get("job_id", ""))
        failure_type = str(raw.get("failure_type", ""))
        cohort = raw.get("cohort")
        after_iteration = raw.get("after_iteration")
        if event_id in seen_ids:
            raise ValueError(f"duplicate scheduled failure id {event_id!r}")
        if job_id not in job_specs:
            raise ValueError(
                f"scheduled failure {event_id!r} names unknown job {job_id!r}"
            )
        if not isinstance(cohort, int) or isinstance(cohort, bool):
            raise ValueError(f"scheduled failure {event_id!r} cohort must be an int")
        spec = job_specs[job_id]
        cohort_count = min(int(spec.get("cohorts", 64)), int(spec["ranks"]))
        if not 0 <= cohort < cohort_count:
            raise ValueError(
                f"scheduled failure {event_id!r} cohort {cohort} is outside "
                f"[0, {cohort_count}) for {job_id}"
            )
        if not isinstance(after_iteration, int) or isinstance(after_iteration, bool):
            raise ValueError(
                f"scheduled failure {event_id!r} after_iteration must be an int"
            )
        if not 1 <= after_iteration <= int(spec["iterations"]):
            raise ValueError(
                f"scheduled failure {event_id!r} iteration {after_iteration} "
                f"is outside [1, {spec['iterations']}] for {job_id}"
            )
        if failure_type not in {"process", "reboot", "node", "spot"}:
            raise ValueError(
                f"scheduled failure {event_id!r} has invalid failure_type "
                f"{failure_type!r}"
            )
        boundary = (job_id, after_iteration)
        if boundary in seen_boundaries:
            raise ValueError(
                f"multiple scheduled failures target {job_id} immediately after "
                f"iteration {after_iteration}"
            )
        event = ScheduledFailure(
            event_id=event_id,
            job_id=job_id,
            cohort=cohort,
            after_iteration=after_iteration,
            failure_type=failure_type,
        )
        seen_ids.add(event_id)
        seen_boundaries.add(boundary)
        grouped[job_id].append(event)

    for events in grouped.values():
        events.sort(key=lambda event: event.after_iteration)
    return grouped


# ------------- deterministic failure-trace replay (EXP2_SPEC.md, CL-018) -------------
# FLAGGED, ABSENT BY DEFAULT: the machinery below exists only when the scenario
# declares `failures.trace_file`. Absent => normalize_failure_trace returns None,
# no process is spawned, no RNG stream is touched, no results key appears — the
# Poisson failure model is byte-identical to the pre-CL-018 code (the same hard
# regression gate as CL-014/CL-016/CL-017).


@dataclass(frozen=True)
class TraceReplayEvent:
    """One job-level failure from the trace: every raw event that shares this
    exact t_s and targets a node of the SAME job, coalesced (EXP2_SPEC §5 —
    the 448.68 s two-node burst is ONE job failure, not two)."""

    t_s: float
    job_id: str
    targets: tuple[str, ...]
    types: tuple[str, ...]


@dataclass(frozen=True)
class FailureTraceReplay:
    """The replayed trace + the frozen pre-run recovery constants (EXP2_SPEC §6:
    the events list is consumed verbatim from the SAME YAML the hardware harness
    replays — never regenerated or resampled — and every constant is a measured,
    frozen number; none is a free parameter and none has a default)."""

    path: str
    window_s: float                    # goodput horizon W (EXP2_SPEC §4)
    detect_s: float                    # §7: t(teardown initiated) − t_kill
    reprovision_delay_s: float         # §3.2: relaunch cost before ckpt load
    restore_s: float                   # §3.3: checkpoint load (single-class legacy)
    events_by_job: dict[str, tuple[TraceReplayEvent, ...]]
    event_count: int
    # CL-019: per-class restore durations for class-mixed traces (recovery
    # ladder F1-F4: process_kill/node_reboot/node_loss/node_and_peer_loss).
    # None => every event uses restore_s (byte-identical legacy behavior).
    restore_by_class: dict | None = None

    def restore_for(self, types: tuple) -> float:
        """Restore duration for one (possibly coalesced) event: the SEVEREST
        class in the burst governs (max duration) — a burst restarts once and
        must reach the deepest surviving tier."""
        if not self.restore_by_class:
            return self.restore_s
        return max(self.restore_by_class.get(ty, self.restore_s)
                   for ty in types)


class _TraceChain:
    """One down-interval of one job: first kill -> (resets) -> resumed."""

    __slots__ = ("t_kill", "it_at_kill", "end", "events")

    def __init__(self, t_kill: float, it_at_kill: int, end: float) -> None:
        self.t_kill = t_kill           # first event's t_s (never moves)
        self.it_at_kill = it_at_kill   # iteration index standing at t_kill
        self.end = end                 # recovery completes here (resets move it)
        self.events = 1


def normalize_failure_trace(sc: dict, jobs: list[tuple[str, dict]]):
    """Parse + validate `failures.trace_file` replay config; None when absent.

    Scenario keys (all under `failures:`, live only when trace_file is set):
      trace_file       path to the exp2 trace YAML (events replayed verbatim)
      trace_node_map   {trace target node -> job_id}: which sim job occupies
                       that physical node (EXP2_SPEC blast radius = the JOB)
      trace_constants  mapping OR path to exp2_constants.yaml; REQUIRED keys
                       detect_s / reprovision_delay_s / restore_s — measured
                       pre-run and frozen, so there are NO defaults here
      trace_window_s   goodput horizon W; defaults to the trace file's own
                       provenance.window_s (one file is the authority)
    """
    failures_cfg = sc.get("failures", {}) or {}
    trace_file = failures_cfg.get("trace_file")
    if trace_file is None:
        return None                    # flag absent => byte-identical Poisson
    path = Path(str(trace_file))
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get("events"), list) \
            or not doc["events"]:
        raise ValueError(f"failure trace {path} must carry a nonempty events list")

    node_map = failures_cfg.get("trace_node_map")
    if not isinstance(node_map, dict) or not node_map:
        raise ValueError(
            "failures.trace_node_map must map trace target nodes to job ids")
    job_ids = {job_id for job_id, _spec in jobs}
    for target, job_id in node_map.items():
        if job_id not in job_ids:
            raise ValueError(
                f"trace_node_map target {target!r} names unknown job {job_id!r}")

    constants = failures_cfg.get("trace_constants")
    if isinstance(constants, str):     # exp2_constants.yaml path
        constants = yaml.safe_load(Path(constants).read_text())
    if not isinstance(constants, dict):
        raise ValueError(
            "failures.trace_constants must be a mapping or a YAML path; the "
            "recovery constants are measured pre-run and frozen (EXP2_SPEC §6)")
    values = {}
    restore_by_class = constants.get("restore_by_class") if isinstance(constants, dict) else None
    if restore_by_class is not None:
        if not isinstance(restore_by_class, dict) or not restore_by_class:
            raise ValueError("trace_constants.restore_by_class must be a non-empty mapping")
        for cls, v in restore_by_class.items():
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"restore_by_class.{cls} must be finite and >= 0")
    for key in ("detect_s", "reprovision_delay_s", "restore_s"):
        if key not in constants:
            raise ValueError(
                f"trace_constants.{key} is required — it is a measured frozen "
                "constant, not a free parameter, and has NO default")
        value = float(constants[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"trace_constants.{key} must be finite and >= 0")
        values[key] = value

    window = failures_cfg.get(
        "trace_window_s", (doc.get("provenance") or {}).get("window_s"))
    if window is None:
        raise ValueError(
            "goodput window: set failures.trace_window_s or provide "
            "provenance.window_s in the trace file")
    window_s = float(window)
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("trace window_s must be finite and positive")

    building: dict[str, list] = {}
    previous_t = 0.0
    count = 0
    for index, raw in enumerate(doc["events"], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"trace events[{index}] must be a mapping")
        t_s = float(raw.get("t_s", -1.0))
        if not math.isfinite(t_s) or t_s < 0:
            raise ValueError(f"trace events[{index}].t_s must be finite and >= 0")
        if t_s < previous_t:
            raise ValueError(
                f"trace events[{index}] is out of order (t_s {t_s} < "
                f"{previous_t}); the trace is replayed verbatim and must be "
                "time-ordered")
        previous_t = t_s
        etype = str(raw.get("type", ""))
        if etype not in ("node_loss", "process_kill"):
            raise ValueError(
                f"trace events[{index}].type {etype!r} unknown "
                "(node_loss | process_kill)")
        target = str(raw.get("target", ""))
        if target not in node_map:
            raise ValueError(
                f"trace events[{index}].target {target!r} is not in "
                "failures.trace_node_map")
        job_id = str(node_map[target])
        count += 1
        per_job = building.setdefault(job_id, [])
        if per_job and per_job[-1]["t_s"] == t_s:
            # EXP2_SPEC §5: equal t_s on the SAME job coalesce to ONE failure.
            # Equal t_s on DIFFERENT jobs stay separate (independent, parallel).
            per_job[-1]["targets"].append(target)
            per_job[-1]["types"].append(etype)
        else:
            per_job.append({"t_s": t_s, "targets": [target], "types": [etype]})

    events_by_job = {
        job_id: tuple(
            TraceReplayEvent(
                t_s=item["t_s"], job_id=job_id,
                targets=tuple(item["targets"]), types=tuple(item["types"]))
            for item in items)
        for job_id, items in building.items()
    }
    return FailureTraceReplay(
        path=str(path), window_s=window_s,
        detect_s=values["detect_s"],
        reprovision_delay_s=values["reprovision_delay_s"],
        restore_s=values["restore_s"],
        events_by_job=events_by_job, event_count=count,
        restore_by_class=restore_by_class)


def trace_replay_driver(backend, rt: "CohortJobRuntime",
                        events: tuple[TraceReplayEvent, ...]):
    """Deliver one job's trace events at their exact t_s (zero RNG draws)."""
    for event in events:
        delay = event.t_s - backend.now
        if delay > 0:
            yield backend.timeout(delay)
        rt.trace_event_now(event)


def trace_window_monitor(backend, runtimes: list, trace: FailureTraceReplay,
                         out: dict) -> None:
    """Snapshot every job's goodput accounting at t = W (EXP2_SPEC §4: the
    metric is defined AT the horizon, not at job completion)."""
    yield backend.timeout(trace.window_s)
    for rt in runtimes:
        out[rt.config.job_id] = rt.trace_snapshot_at_window(trace.window_s)


class CohortJobRuntime:
    """Lumped training clock + cohort-granular checkpoints for one wide job."""

    # FABRIC-SCOPED CHECKPOINT<->COLLECTIVE COUPLING.
    # sim_validation_protocol.md CL-012.D S1/S2, implemented by CL-014.
    # OFF (default): _allreduce_slowdown reads the fabric-blind aggregate count,
    #   i.e. ANY in-flight checkpoint transfer on this job's workers stretches
    #   its collective, whichever wire it rides. Every committed scenario,
    #   staircase, rack table and parallelism table depends on this and is
    #   byte-identical with the flag absent — that is the hard regression gate.
    # ON  (`model.fabric_aware_coupling: true`): a transfer counts against this
    #   job's collective ONLY if it rides the fabric the collective rides
    #   (class `rates.nic_fabric` vs the transfer's `rates.ckpt_fabric`). A
    #   physically separate card contributes exactly zero. The MAGNITUDE is
    #   deliberately unchanged — it is still 1 + tasks, not an intensity term;
    #   intensity is CL-012.D S3 and is a separate C3 with its own justification.
    # Set per run by run_arm(); a class attribute so arms cannot leak into
    # each other and so a test can flip it without rebuilding a scenario.
    fabric_aware_coupling = False

    # INTENSITY-BASED CHECKPOINT<->COLLECTIVE COUPLING.
    # sim_validation_protocol.md CL-012.D S3, implemented by CL-016.
    # OFF (default): whichever of the two paths above the fabric flag selects —
    #   every committed scenario, staircase, rack table and parallelism table is
    #   byte-identical with the flag absent (the hard regression gate, same as
    #   CL-014).
    # ON  (`model.intensity_coupling: true`): implies fabric-scoped semantics
    #   (only same-fabric transfers are visible at all), and the MAGNITUDE stops
    #   being a count. A transfer sharing the collective's fabric charges
    #   slowdown proportional to actual bandwidth contention:
    #       offered = collective demand + in-flight transfer demand   (GB/s)
    #       factor  = offered / link capacity   if offered > capacity, else 1.0
    #   An unsaturated link charges NOTHING (factor exactly 1.0). Every input is
    #   a measured constant the scenario already declares — ZERO free
    #   parameters (see _allreduce_slowdown for per-input provenance).
    # Set per run by run_arm(), exactly parallel to fabric_aware_coupling.
    intensity_coupling = False

    def __init__(self, *, config: JobConfig, class_name: str, spec: dict,
                 cluster: ClusterConfig,
                 failures: FailureSettings, per_node_fail_rate: float,
                 logger: EventLogger, backend, object_store, rng: random.Random,
                 strategy_cls: type = CrossJobPeerStrategy,
                 baseline: str | None = None,
                 scheduled_failures: list[ScheduledFailure] | None = None):
        self.config = config
        self.class_name = class_name
        self.backend = backend
        self.logger = logger
        self.failures = failures
        self.rng = rng
        self.baseline = baseline    # driver mode: None (default) or e.g. "jit"
        # parallelism config (D2): None for legacy classes (no `model:` block),
        # which keep every original recovery semantic. Present -> drives the
        # baseline-fairness audit (e.g. JIT survivor-DRAM recovery is possible
        # only when a DP peer holds an intact replica: dp>1 AND zero_stage==0).
        self.model = (parallelism.parse_model(spec["model"])
                      if spec.get("model") else None)
        # jit_pc_fallback (Gupta et al. EuroSys'24 Sec. 5.2, fairness fix
        # 2026-07-28): set by run_arm ONLY for jit-arm classes whose ZeRO stage
        # leaves no replicated state (the ones that previously fell to scratch).
        # When True, single-node failures take the GENERIC recover() path —
        # newest store copy laid down by the periodic c* checkpoints — instead
        # of the survivor-drain-else-scratch jit branch. False (default)
        # reproduces the archived behavior exactly.
        self.jit_pc_fallback = False
        self.ranks = config.data_parallel_replicas
        self.iter_s = float(spec["iteration_seconds"])
        self.iterations = int(spec["iterations"])
        # all-reduce timing: the detailed engine's exact ring formula, on top of the
        # compute time — and the same count-based checkpoint<->training coupling
        # (network_slowdown = 1 + max in-flight checkpoint transfers; this is what
        # stretches training while flushes congest NICs, e.g. the store arm's 2x
        # all-reduce measured in final73 traces)
        nic = float(spec.get("rates", {}).get("nic", cluster.network_bandwidth_gbps))
        self.ar_base = (2 * (self.ranks - 1) / self.ranks
                        * float(spec.get("gradient_gb_per_rank", 4.0)) / nic)
        # CL-016 (CL-012.D S3) inputs, both PRE-EXISTING scenario constants:
        # * collective demand = `rates.nic` — the measured NCCL busbw for the
        #   collective's fabric (e.g. exp1's nccl_allreduce_roce_gbps = 4.03,
        #   probed by nccl-tests), i.e. the SAME whitelisted constant that sets
        #   ar_base two lines up. Nothing new is declared.
        # * link capacity = the class's declared node NIC capacity on that
        #   fabric, min(rates.nic_in, rates.nic_out) — the nameplate line rate
        #   per CL-012.D S4 (`ethtool` link speed), and exactly the caps
        #   CrossJobPeerStrategy._node_caps() already feeds the CapacityRegistry
        #   (same fallback chain: nic_in/nic_out -> nic -> cluster default).
        self.collective_busbw_gbps = nic
        _r = spec.get("rates", {}) or {}
        self.train_link_capacity_gbps = min(float(_r.get("nic_in", nic)),
                                            float(_r.get("nic_out", nic)))
        # which fabric that collective rides (CL-014). `rates.nic_fabric`, or the
        # single default fabric when the scenario does not say — see
        # checkpointing/base.py. Read even when the flag is off (it is inert then)
        # so the declaration is always visible to tests and to the trace.
        self.train_fabric = train_fabric_of(spec.get("rates"))
        self.fail_rate = self.ranks * per_node_fail_rate   # job-level failures/s
        n_cohorts = min(int(spec.get("cohorts", 64)), self.ranks)
        base, extra = divmod(self.ranks, n_cohorts)
        self.workers: dict[int, CohortWorker] = {}
        for i in range(n_cohorts):
            self.workers[i] = CohortWorker(
                job_id=config.job_id, rank=i, data_parallel_rank=i, pipeline_stage=0,
                physical_node=f"{config.job_id}-cohort-{i}",
                gpu=backend.resource(1), cpu=backend.priority_resource(1),
                represents=base + (1 if i < extra else 0))
        self.checkpoint_strategy = strategy_cls(
            run_id=0, cluster=cluster, logger=logger, object_store=object_store,
            workers=self.workers, backend=backend, paired=False, persist_to_ssd=False)
        self.it = 0
        self.capture_every: int | None = None   # f1 enactment (policy-set)
        self.store_every: int | None = None     # f3 backstop cadence (policy-set)
        self.last_store_it: int = 0             # last iteration a backstop fired
        self.phase = 0.0                  # start offset (excluded from completion)
        self.arrival_s = 0.0              # Stage 3: job arrives (starts) at this t
        self.arrival_source = "initial"
        self.start_time: float | None = None
        self.state = "pending"
        self.all_strategies: list = []    # set by run_arm (cross-strategy wipes)
        self.end_time: float | None = None
        self.restores: Counter = Counter()
        self.lost_iters: list[int] = []
        self.lost_by_tier: dict[str, list[int]] = {}   # tier -> [iters rolled back]
        self.failure_types: Counter = Counter()
        self.reboot_local_restores: int = 0   # reboots served by own surviving SSD
        self._preempt_requested = False
        # rack failure domains (rack_failure_spec.md): the coordinator queues
        # (worker, dark_node_count, event_time) here at EVENT time; the run loop
        # processes the batch at the next iteration boundary (_rack_failure),
        # like every other failure class. _rack_rng is the dedicated seeded
        # `rack` stream (None while the feature is off => zero draws).
        self._rack_pending: list = []
        self._rack_rng: random.Random | None = None
        self.scheduled_failures_planned = tuple(scheduled_failures or ())
        self._scheduled_failures = {
            event.after_iteration: event
            for event in self.scheduled_failures_planned
        }
        self.scheduled_failures_fired: list[ScheduledFailure] = []
        # DETERMINISTIC FAILURE-TRACE REPLAY (EXP2_SPEC.md, CL-018). None unless
        # the scenario declares `failures.trace_file` (run_arm wires it): then
        # trace_replay carries the frozen recovery constants, trace_replay_driver
        # calls trace_event_now at each event's exact t_s, and the run loop
        # handles the failure at the next iteration boundary (_trace_failure),
        # boundary-aligned like every other failure class here. While None,
        # every attribute below is inert — no draws, no timeouts, no events.
        self.trace_replay: FailureTraceReplay | None = None
        self._trace_chain: _TraceChain | None = None
        self._trace_reset_evt = None
        self.trace_restarts = 0            # completed down-chains
        self.trace_resets = 0              # events landing mid-recovery (§3 reset)
        self.trace_events_seen = 0         # raw trace events delivered to this job
        self.trace_events_ignored = 0      # events while pending/finished
        self.trace_lost_iters = 0          # Σ (it at first kill − it restored)
        self.trace_recovery_s = 0.0        # Σ (resume − first kill) per chain
        self.trace_ckpt_stall_s = 0.0      # Σ capture-induced GPU-wait (§4 ckpt_s)
        # PER-ITERATION EMISSION (sim-vs-real validation, exp1 mirror 2026-07-31).
        # OFF by default: when False this class behaves exactly as before (the
        # coalesced `training_iterations` bars are the only Training events, and
        # no extra event ids are consumed), so every existing scenario/trace is
        # byte-identical. When True, run() additionally records ONE
        # `training_iteration` event per cohort per iteration carrying the
        # measured wall split (gpu-wait stall + compute + all-reduce), which is
        # what METRICS.md §1 `dt_s` is defined on. Purely observational: it adds
        # logger records only — no RNG draws, no timeouts, no state changes.
        self.emit_iterations = False

    def _record_iteration(self, it_start: float, gpu_ready: float) -> None:
        """One METRICS.md §1-shaped record per cohort for the iteration that just
        finished. dt_s = the whole wall interval the cohort spent on it, i.e.
        (waiting for its GPU because a capture holds it) + compute + all-reduce."""
        now = self.backend.now
        stall_s = round(gpu_ready - it_start, 9)
        allreduce_s = round(now - gpu_ready - self.iter_s, 9)
        for w in self.workers.values():
            self.logger.record(
                start=it_start, end=now, run_id=0, job_id=w.job_id,
                rank=w.rank, node=w.name, physical_node=w.physical_node,
                pipeline_stage=w.pipeline_stage,
                data_parallel_rank=w.data_parallel_rank,
                category="Training", operation="training_iteration",
                resources=["GPU"], iteration=self.it,
                details={"represents": w.represents,
                         "dt_s": round(now - it_start, 9),
                         "stall_s": stall_s,
                         "compute_s": self.iter_s,
                         "allreduce_s": allreduce_s,
                         "class": self.class_name})

    def _set_iteration(self, iteration: int, *, rollback: bool = False) -> None:
        """Move the synchronized DDP clock and invalidate abandoned state."""
        target = max(0, min(int(iteration), self.iterations))
        if rollback:
            for worker in self.workers.values():
                worker.current_iteration = target
            discard = getattr(
                self.checkpoint_strategy, "discard_checkpoints_after", None
            )
            if discard is not None:
                discard(self.config.job_id, target)
        else:
            for worker in self.workers.values():
                worker.current_iteration = target
        self.it = target
        self.last_store_it = min(self.last_store_it, self.it)

    def _invalidate_checkpoint_timeline(self) -> None:
        """Cancel queued work from the pre-failure checkpoint timeline."""
        for worker in self.workers.values():
            worker.checkpoint_epoch += 1
            worker.failure_generation = getattr(worker, "failure_generation", 0) + 1
            cap = getattr(self.checkpoint_strategy, "capacity", None)
            if cap is not None and getattr(
                self.checkpoint_strategy, "engine", None
            ) == "event":
                cap.ev_notify_worker_change(worker)

    def _discard_future_checkpoints(self, iteration: int | None = None) -> None:
        discard = getattr(
            self.checkpoint_strategy, "discard_checkpoints_after", None
        )
        if discard is not None:
            discard(
                self.config.job_id,
                self.it if iteration is None else max(0, int(iteration)),
            )

    def _common_recovery_plan(self) -> tuple[int, dict[int, tuple[int, str]]]:
        """Find one iteration every cohort can restore without time travel."""
        target = self.it
        while True:
            for worker in self.workers.values():
                worker.current_iteration = target
            peeks = {
                rank: self._peek_restore(worker)
                for rank, worker in self.workers.items()
            }
            next_target = min(
                max(restored_it, 0) for restored_it, _tier in peeks.values()
            )
            if next_target == target:
                return target, peeks
            if next_target > target:
                raise AssertionError(
                    f"future checkpoint selected for {self.config.job_id}: "
                    f"target={target} selected={next_target}"
                )
            target = next_target

    # ---- external API (driver preemption process) -----------------------------
    @property
    def healthy(self) -> bool:
        return self.state == "active" and all(
            not w.failed for w in self.workers.values()
        )

    @property
    def last_failed_until(self) -> float:
        # audit fix #11: grace must measure from the MOST RECENT recovery,
        # otherwise one old timestamp makes the grace period a no-op
        return max((w.failed_until for w in self.workers.values()), default=0.0)

    def request_preemption(self) -> None:
        self._preempt_requested = True

    def _book_loss(self, tier: str, lost: int) -> None:
        """Record one restore's rolled-back iterations, split by recovery tier
        (feeds the drain-vs-push staleness table: L3 tiers vs the L2 peer tier)."""
        if lost < 0:
            raise AssertionError(
                f"future checkpoint selected for {self.config.job_id}: "
                f"tier={tier} lost={lost} current_iteration={self.it}"
            )
        self.lost_iters.append(lost)
        self.lost_by_tier.setdefault(tier, []).append(lost)

    def _flush_training_block(self, block_start: float) -> float:
        """Emit one coalesced Training bar per cohort for [block_start, now] —
        the cohort engine advances training as timeouts, so without this the
        trace (and every report) shows checkpoints floating in blank space."""
        now = self.backend.now
        if now - block_start > 1e-9:
            for w in self.workers.values():
                self.logger.record(
                    start=block_start, end=now, run_id=0, job_id=w.job_id,
                    rank=w.rank, node=w.name, physical_node=w.physical_node,
                    pipeline_stage=w.pipeline_stage,
                    data_parallel_rank=w.data_parallel_rank,
                    category="Training", operation="training_iterations",
                    resources=["GPU"], iteration=self.it,
                    details={"represents": w.represents})
        return now

    # ---- training loop ---------------------------------------------------------
    def run(self):
        strat = self.checkpoint_strategy
        block_start = self.backend.now
        # audit fix #12: arrivals are an absolute-time Poisson process; each
        # next arrival extends from the PREVIOUS ARRIVAL, never from recovery
        # end (resampling from recovery end thinned the process ~40% for jobs
        # whose iteration+recovery spans exceeded the inter-arrival time)
        # failure clock starts when the runtime actually begins (Stage 3: a job
        # that ARRIVES at t=arrival_s only starts failing then; its nodes did not
        # exist before). backend.now is 0 for t0 jobs -> byte-identical there.
        t_fail = (self.backend.now + self.rng.expovariate(self.fail_rate)
                  if self.fail_rate > 0 else float("inf"))
        while self.it < self.iterations:
            # compute + all-reduce holds every cohort GPU, exactly like the detailed
            # engine's per-op gpu.request(): an in-flight capture (itself blocked on
            # the flush-held CPU core) stalls the next iteration — the backpressure
            # that produces the store arm's measured collapse.
            it_start = self.backend.now
            requests = [w.gpu.request() for w in self.workers.values()]
            yield self.backend.all_of(requests)
            gpu_ready = self.backend.now
            if self.trace_replay is not None:
                # EXP2_SPEC §4 checkpoint_s: time stalled in capture. In this
                # engine a capture holds the GPU, so the stall is exactly the
                # GPU-wait before the iteration. Pure accounting, gated on
                # trace mode so non-trace runs are untouched.
                self.trace_ckpt_stall_s += gpu_ready - it_start
            try:
                yield self.backend.timeout(self.iter_s)      # fwd + bwd + optimizer
                yield from advance_work(
                    self.backend, base_duration=self.ar_base,
                    slowdown=self._allreduce_slowdown, aborted=lambda: False,
                    quantum=0.5)
            finally:
                for w, req in zip(self.workers.values(), requests):
                    w.gpu.release(req)
            self.it += 1
            if self.emit_iterations:
                self._record_iteration(it_start, gpu_ready)
            for worker in self.workers.values():
                worker.current_iteration = self.it
            scheduled = self._scheduled_failures.pop(self.it, None)
            if scheduled is not None:
                block_start = self._flush_training_block(block_start)
                self.scheduled_failures_fired.append(scheduled)
                yield from self._single_node_failure(
                    scheduled.failure_type,
                    cohort=scheduled.cohort,
                )
                block_start = self.backend.now
                continue
            if self._trace_chain is not None:
                # trace replay (CL-018): the driver recorded the kill at its
                # exact t_s (and already aborted every in-flight stripe); the
                # job-wide teardown + recovery runs here at the boundary,
                # exactly like every other failure class in this engine.
                block_start = self._flush_training_block(block_start)
                yield from self._trace_failure()
                block_start = self.backend.now
                continue
            persist_wave = (self.it % self.config.checkpoint_every == 0
                            and self.it < self.iterations)
            if persist_wave:
                block_start = self._flush_training_block(block_start)
                for w in self.workers.values():
                    self.backend.process(
                        strat.checkpoint(
                            w,
                            self.config,
                            iteration=self.it,
                            checkpoint_epoch=w.checkpoint_epoch,
                        )
                    )
            elif (self.capture_every and self.it % self.capture_every == 0
                  and self.it < self.iterations):
                # f1: cheap DRAM snapshot between persists (process-failure
                # freshness at the SOLVED f1 cadence, not f2's)
                block_start = self._flush_training_block(block_start)
                for w in self.workers.values():
                    self.backend.process(
                        strat.snapshot(
                            w,
                            self.config,
                            iteration=self.it,
                            checkpoint_epoch=w.checkpoint_epoch,
                        )
                    )
                self.last_capture_it = self.it
            if (self.store_every and persist_wave and self.it < self.iterations
                    and self.it - self.last_store_it >= self.store_every
                    and getattr(strat, "backstop", "owner_push") == "owner_push"):
                # f3 OWNER-PUSH backstop: the owner uploads the whole shard from
                # DRAM to the store on the store_every cadence. Under donor_drain
                # the donors instead forward pieces to L3 every wave (inside
                # _persist_flow), so this owner-push path is skipped entirely.
                # f3: store backstop rides the FIRST persist wave at or after
                # each store interval. (The old `it % store_every == 0 and
                # persist_wave` needed the two cadences to share a multiple —
                # for hand-set ratios the LCM could exceed the horizon and the
                # backstop never fired; even solved policies fired rarer than
                # intended when store_every wasn't a multiple of the persist
                # cadence.)
                self.last_store_it = self.it
                for w in self.workers.values():
                    strat.background_flushes.append(self.backend.process(
                        strat.backstop_upload(
                            w,
                            self.config,
                            iteration=self.it,
                            checkpoint_epoch=w.checkpoint_epoch,
                        )
                    ))
            if self._preempt_requested:
                self._preempt_requested = False
                self._flush_training_block(block_start)
                yield from self._whole_job_eviction()
                block_start = self.backend.now
            if self._rack_pending:
                # rack event (rack_failure_spec.md): boundary-aligned like every
                # other failure class — the coordinator already destroyed the
                # dark copies and bumped generations at event time; here the
                # affected cohorts restart on REPLACEMENT nodes and restore.
                pending, self._rack_pending = self._rack_pending, []
                block_start = self._flush_training_block(block_start)
                yield from self._rack_failure(pending)
                block_start = self.backend.now
            if self.fail_rate > 0 and self.backend.now >= t_fail:
                block_start = self._flush_training_block(block_start)
            while self.backend.now >= t_fail:
                ftype = self.rng.choices(
                    ("process", "reboot", "node", "spot"),
                    weights=(self.failures.process_weight,
                             getattr(self.failures, "reboot_weight", 0.0),
                             self.failures.node_weight,
                             self.failures.spot_weight))[0]
                yield from self._single_node_failure(ftype)
                t_fail += self.rng.expovariate(self.fail_rate)
        self._flush_training_block(block_start)
        self.end_time = self.backend.now

    def _allreduce_slowdown(self) -> Slowdown:
        """How much concurrent checkpoint traffic stretches this job's collective.

        FABRIC-BLIND (default, `fabric_aware_coupling` off): every in-flight
        transfer on one of this job's workers counts, whichever wire it rides.
        FABRIC-SCOPED (flag on, CL-014): only transfers on the fabric this job's
        collective actually rides count, so a checkpoint stream on a physically
        separate card contributes NOTHING and the factor is exactly 1.0. The
        magnitude of a same-fabric transfer is unchanged (1 + tasks).
        INTENSITY (`intensity_coupling` on, CL-016 = CL-012.D S3; implies the
        fabric scoping): the magnitude stops being a count and becomes actual
        bandwidth contention on the collective's link —
            offered = collective busbw demand + in-flight transfer demand
            factor  = offered / capacity   if offered > capacity, else 1.0.
        An unsaturated link charges NOTHING; a separate fabric contributes no
        demand and so charges nothing at any rate. ZERO free parameters:
        * collective demand — measured NCCL busbw (`rates.nic`, the ar_base
          constant; whitelisted, probe-measured);
        * transfer demand — the in-flight streams' offered rates as bump-ed by
          the transfer paths (shaped tc rate / measured TCP ceiling / path NIC
          caps: all declared scenario constants), max over this job's workers,
          mirroring the count path's max-over-workers reduction;
        * capacity — the declared node NIC line rate min(nic_in, nic_out),
          CL-012.D S4's nameplate `ethtool` link speed.
        getattr on the flags keeps pre-CL-016 SimpleNamespace test mocks valid;
        real instances always carry the class attributes."""
        if getattr(self, "intensity_coupling", False):
            fabric = self.train_fabric
            demand = max(
                (w.checkpoint_network_demand_by_fabric.get(fabric, 0.0)
                 for w in self.workers.values()), default=0.0)
            if demand <= 0.0:
                return Slowdown(factor=1.0, reasons=frozenset())
            offered = self.collective_busbw_gbps + demand
            capacity = self.train_link_capacity_gbps
            if capacity <= 0.0 or offered <= capacity:
                # unsaturated shared link: the checkpoint stream fits in the
                # headroom and charges the collective NOTHING
                return Slowdown(factor=1.0, reasons=frozenset())
            return Slowdown(factor=offered / capacity,
                            reasons=frozenset({"checkpoint_network"}))
        if self.fabric_aware_coupling:
            fabric = self.train_fabric
            tasks = max(w.checkpoint_network_tasks_by_fabric.get(fabric, 0)
                        for w in self.workers.values())
        else:
            tasks = max(w.checkpoint_network_tasks for w in self.workers.values())
        if tasks <= 0:
            return Slowdown(factor=1.0, reasons=frozenset())
        return Slowdown(factor=1.0 + tasks,
                        reasons=frozenset({"checkpoint_network"}))

    def _peek_restore(self, w: CohortWorker) -> tuple[int, str]:
        """Which checkpoint recover() will pick (newest of store/peer/local) —
        mirrors CrossJobPeerStrategy.recover()'s preference order for accounting.
        A baseline strategy that recovers differently supplies its own
        peek_restore hook (e.g. gemini's surviving-replica tier)."""
        s = self.checkpoint_strategy
        peek = getattr(s, "peek_restore", None)
        if peek is not None:
            return peek(w, self.config, self.it)
        local = s._latest_available_copy(w)
        peer = s._best_peer_set(self.config.job_id, w.rank)
        # store copies count regardless of store_mode: backstop uploads
        # (f3) also land there, and recover() consults them unconditionally —
        # this gate predated the backstop and made L3 invisible to rollback
        # accounting for non-store arms (caught by the ablation batch: no-peer
        # arms showed 0 store recoveries with backstops firing, 2026-07-17)
        store = s.store_copies.get((self.config.job_id, w.rank))
        if store is not None and store["iteration"] > w.current_iteration:
            store = None
        local_it = local.iteration if local is not None else -1
        peer_it = peer[0] if peer is not None else -1
        store_it = store["iteration"] if store is not None else -1
        # piece-level L3 stitch (mirrors recover()'s order): only when it beats
        # every complete source. _best_stitch_set is a no-op for baselines/strats
        # without it, so guard by attribute.
        stitch_fn = getattr(s, "_best_stitch_set", None)
        stitch = stitch_fn(self.config.job_id, w.rank) if stitch_fn else None
        stitch_it = stitch[0] if stitch is not None else -1
        parity_fn = getattr(s, "_best_parity_set", None)
        parity = parity_fn(self.config.job_id, w.rank) if parity_fn else None
        parity_it = parity[0] if parity is not None else -1
        if parity_it > max(local_it, peer_it, store_it, stitch_it):
            return parity_it, "crossjob_peer_parity"
        if stitch_it > max(local_it, peer_it, store_it):
            return stitch_it, "crossjob_peer_l3_stitch"
        if store_it > max(local_it, peer_it):
            return store_it, "store"
        if peer is not None and peer_it > local_it:
            # injob_ssd (rack_failure_spec.md): the ring-neighbor SSD replica
            # carries its own recovery-source label, mirroring recover().
            if any(m.get("injob_ssd") for _d, m in peer[1]):
                return peer_it, "injob_ssd_replica"
            # L1.5: mirror recover()'s tier labels — pieces still resident in
            # donor DRAM recover as crossjob_peer_dram; mixed waves (some pieces
            # demoted, some not) are the dram stitch variant.
            piece_tiers = [m.get("tier", "ssd") for _d, m in peer[1]]
            if all(t == "dram" for t in piece_tiers):
                return peer_it, "crossjob_peer_dram"
            if "dram" in piece_tiers:
                return peer_it, "crossjob_peer_dram_stitch"
            return peer_it, "crossjob_peer"
        if local is not None:
            return local_it, local.tier
        return -1, "initial_state"

    def _restart_seconds(self, ftype: str) -> float:
        # getattr default keeps SimpleNamespace test mocks (no reboot field)
        # working; a real FailureSettings always carries reboot_restart_seconds.
        # rack: replacement-node provisioning; defaults to the node value (a
        # rack restart IS a node replacement, for every node of the rack).
        rack_restart = getattr(self.failures, "rack_restart_seconds", None)
        return {"process": self.failures.process_restart_seconds,
                "reboot": getattr(self.failures, "reboot_restart_seconds", 180.0),
                "node": self.failures.node_restart_seconds,
                "spot": self.failures.spot_restart_seconds,
                "rack": (rack_restart if rack_restart is not None
                         else self.failures.node_restart_seconds)}[ftype]

    def _fail_worker(self, w: CohortWorker, ftype: str, wipe_local: bool,
                     full_wipe: bool = False, wipe_tier: str | None = None) -> None:
        w.failure_generation += 1
        w.failed = True
        # event engine: abort in-flight capacity transfers that watch this worker
        # (as a flusher or as a donor) now that its generation changed — the same
        # abort the polling loop detects on its next quantum. No-op in polling mode.
        cap = getattr(self.checkpoint_strategy, "capacity", None)
        if cap is not None and getattr(self.checkpoint_strategy, "engine", None) == "event":
            cap.ev_notify_worker_change(w)
        w.failure_type = ftype
        w.recovered_event = self.backend.event()
        if wipe_local:
            # eviction kills ALL m nodes (full wipe); a single-node failure
            # kills each located copy with p=1/m (fairness: no wholesale
            # cohort wipes — they over-punished in-job replication baselines)
            frac = 1.0 if full_wipe else 1.0 / max(w.represents, 1)
            # reboot (2026-07-21): the host restarts on the SAME node, so only the
            # DRAM-tier copies drop (wipe_tier='dram'); the local SSD survives.
            # node/spot pass wipe_tier=None -> every located tier on that node dies.
            self.checkpoint_strategy.drop_local_copies(
                w.rank, fraction=frac, rng=self.rng, tier=wipe_tier)

    def _finish_worker(self, w: CohortWorker) -> None:
        w.failed = False
        w.failed_until = self.backend.now
        ev, w.recovered_event = w.recovered_event, None
        if ev is not None:
            ev.succeed()

    def _single_node_failure(self, ftype: str, *, cohort: int | None = None):
        """One node of one cohort dies -> whole job stalls (DDP barrier) for
        restart + that node's per-rank shard refetch."""
        w = (
            self.workers[cohort]
            if cohort is not None
            else self.rng.choice(list(self.workers.values()))
        )
        self.failure_types[ftype] += 1
        self._invalidate_checkpoint_timeline()
        # reboot (2026-07-21): wipe DRAM only (host restarts on the SAME node, so
        # the local SSD survives); process wipes nothing; node/spot wipe every
        # located tier. wipe_local drives whether anything drops; wipe_tier which.
        self._fail_worker(w, ftype, wipe_local=(ftype != "process"),
                          wipe_tier=("dram" if ftype == "reboot" else None))
        if ftype not in ("process", "reboot"):
            # audit fix #13: ONE node of this m-node cohort died — shards it
            # HOSTED for other jobs die with it (sampled 1/m per entry).
            # reboot EXEMPT (2026-07-21): a host restart leaves the donor SSD
            # intact, so the hosted donor pieces on w's node SURVIVE (do NOT
            # drop_hosted_copies) — reboot's whole point vs node/spot.
            for s in self.all_strategies:
                drop = getattr(s, "drop_hosted_piece", None)
                for key in [k for k in s.peer_copies if k[2] == w.name
                            and self.rng.random() < 1.0 / max(w.represents, 1)]:
                    if drop is not None:      # frees L1.5 DRAM ledger if dram-tier
                        drop(key, lost=True)
                    else:
                        del s.peer_copies[key]
        elif ftype == "reboot":
            # L1.5 (l15_peer_dram_spec.md): a host REBOOT wipes the daemon's DRAM
            # buffers — hosted DRAM-TIER pieces on w's node are lost (sampled 1/m
            # like node/spot), while hosted SSD-tier pieces keep the exemption
            # above EXACTLY as before. No dram pieces => no RNG draws => every
            # pre-L1.5 scenario is byte-identical.
            for s in self.all_strategies:
                drop = getattr(s, "drop_hosted_piece", None)
                if drop is None:
                    continue
                for key in [k for k, m2 in s.peer_copies.items()
                            if k[2] == w.name
                            and isinstance(m2, dict)
                            and m2.get("tier") == "dram"
                            and self.rng.random() < 1.0 / max(w.represents, 1)]:
                    # disk_survives: a reboot wipes DRAM only — if the donor
                    # still held the previous demoted piece's SSD file, it is
                    # restored (write-then-rename; see drop_hosted_piece)
                    drop(key, lost=True, disk_survives=True)
        yield self.backend.timeout(self._restart_seconds(ftype))
        if self.baseline == "jit" and not self.jit_pc_fallback:
            # JIT (EuroSys'24): recovers a lost rank by draining a surviving DP
            # replica's DRAM snapshot over the NIC; exactly one iteration lost.
            # FAIRNESS AUDIT (D2): this only works when a DP peer holds an INTACT
            # replica of the failed rank's full training state — dp>1 AND
            # zero_stage==0 (full DP replication). Under ZeRO-1 the failed rank's
            # optimizer slice is sharded (unique to that rank), so no peer has it;
            # under ZeRO-3 the whole shard is unique. JIT keeps no periodic
            # checkpoint, so in those regimes a lost rank is unrecoverable from
            # survivors and honestly falls to initial_state (total loss). Legacy
            # classes (self.model is None) keep the original survivor-drain path.
            # (Whole-job eviction takes the generic path below regardless.)
            #
            # jit_pc_fallback (fairness fix 2026-07-28): the scratch arm above
            # was NOT what the paper prescribes for sharded state. Gupta et al.
            # Sec. 5.2 degrade to PERIODIC checkpointing at the optimal frequency
            # c* = sqrt(N*f/(2*o)) (their eq. 3) when JIT's transparent mechanism
            # cannot cover a failure class. run_arm therefore sets
            # jit_pc_fallback=True on exactly the classes that used to take the
            # scratch route (no-replica ZeRO stages) and gives them
            # checkpoint_every = c* with persists routed to the L3 store
            # (baselines.jit_pc_fallback_rule); those classes SKIP this branch
            # and recover on the generic path below (newest store copy).
            # Replica-holding classes never get the flag, so this branch — and
            # every RNG draw on it — is byte-identical to the archived arm.
            restored_it = max(self.it - 1, 0)
            if parallelism.jit_dram_recoverable(self.model):
                ok = yield from self.checkpoint_strategy.jit_drain(
                    w, self.config, iteration=restored_it)
            else:
                ok = False              # no intact DP replica (ZeRO-sharded state)
            self._finish_worker(w)
            if ok:
                self.restores["jit_peer_dram"] += 1
                self._book_loss("jit_peer_dram", self.it - restored_it)
                self._set_iteration(restored_it, rollback=True)
            else:                       # no survivor (1-node job): total loss
                self.restores["initial_state"] += 1
                self._book_loss("initial_state", self.it)
                self._set_iteration(0, rollback=True)
            return
        self._discard_future_checkpoints()
        restored_it, tier = self._peek_restore(w)   # fix #9: what recover() sees NOW
        ok = yield from self.checkpoint_strategy.recover(w, self.config)
        self._finish_worker(w)
        if not ok:
            tier, restored_it = "initial_state", -1
        self.restores[tier] += 1
        # reboot physics made visible (2026-07-21): a reboot that recovers from
        # the SAME node's surviving local copy (only the SSD tier survives a
        # reboot; DRAM was wiped) is a LEGITIMATE own-SSD recovery at the class's
        # durable cadence — NOT a stale/scratch fallback. Count it apart so the
        # no-peer arms' new coverage niche is auditable.
        if ftype == "reboot" and tier in ("ssd", "dram"):
            self.reboot_local_restores += 1
        self._book_loss(tier, self.it - max(restored_it, 0))
        # rollback semantics (Sam, 2026-07-14): destroyed work must be
        # RE-EXECUTED, not just counted — the job rewinds to the restored
        # iteration. Without this, completion time was blind to losses and
        # maximally flattered rare checkpointing.
        self._set_iteration(max(restored_it, 0), rollback=True)
        # rollback interaction with the interval backstop: last_store_it must
        # rewind with the iteration counter, or `it - last_store_it` goes
        # negative and the backstop never fires again for a job that keeps
        # being rolled back (exactly the evicted best-effort jobs that need
        # L3 most — caught in the first ablation batch, 2026-07-17)

    # ---- deterministic failure-trace replay (EXP2_SPEC.md, CL-018) -------------
    def trace_event_now(self, event: TraceReplayEvent) -> None:
        """Called by trace_replay_driver at the event's exact t_s.

        EXP2_SPEC semantics enforced HERE, at event time (not at the boundary):
        - blast radius is the JOB (§2): one event on any of this job's nodes
          dooms the whole job — the boundary handler restarts every cohort;
        - a stripe in flight at t_kill gets NO partial credit (§3.3): the
          generation bump aborts every in-flight transfer NOW, so nothing can
          commit at or after t_kill;
        - the killed nodes' local state is unreadable from here on (§1): the
          local spool is wiped for every rank, deterministically (fraction 1.0
          => zero RNG draws). Hosted donor pieces on these nodes SURVIVE — the
          failure is a process kill; host, SSD and daemon stay up;
        - an event landing while the job is already down RESETS recovery (§3):
          the clock restarts at this event's t_s + reprovision + restore. One
          rule, no special cases. Rolled-back work is never counted twice: the
          chain keeps the FIRST kill's iteration index.
        """
        now = self.backend.now
        self.trace_events_seen += len(event.targets)
        replay = self.trace_replay
        active = self.state == "active" and self.end_time is None
        chain = self._trace_chain
        action = ("ignored" if not active
                  else "reset" if chain is not None else "failure")
        self.logger.record(
            start=now, end=now, run_id=0, job_id=self.config.job_id,
            rank=None, node=f"{self.config.job_id}-trace", category="Failure",
            operation="trace_node_loss", failure_type="trace_node_loss",
            details={"t_s": round(event.t_s, 6),
                     "targets": list(event.targets),
                     "types": list(event.types),
                     "coalesced": len(event.targets) > 1,
                     "action": action})
        if not active:
            self.trace_events_ignored += len(event.targets)
            return
        if chain is not None:                       # §3 reset — one rule
            chain.end = (now + replay.reprovision_delay_s
                         + replay.restore_for(event.types))
            chain.events += 1
            self.trace_resets += 1
            reset_evt = self._trace_reset_evt
            if reset_evt is not None and not reset_evt.triggered:
                reset_evt.succeed()
            return
        self._trace_chain = _TraceChain(
            t_kill=now, it_at_kill=self.it,
            end=now + replay.detect_s + replay.reprovision_delay_s
                + replay.restore_for(event.types))
        for _ty in set(event.types):
            self.failure_types[f"trace_{_ty}"] += 1
        self._invalidate_checkpoint_timeline()      # no partial credit (§3.3)
        for w in self.workers.values():             # §1: local spool unreadable
            self.checkpoint_strategy.drop_local_copies(w.rank, fraction=1.0)

    def _trace_failure(self):
        """Boundary handler for the pending trace chain: teardown -> wait out
        detect + reprovision + restore (frozen measured constants, §3/§7; a
        reset moves the deadline) -> restore the latest complete checkpoint
        committed strictly before t_kill -> resume. Recovery durations are the
        frozen constants, NOT simulated transfers — the sim mirrors the
        harness's measured relaunch/restore cost 1:1 (§6)."""
        chain = self._trace_chain
        for w in self.workers.values():
            # whole-job teardown (§2/§3.1): the supervisor SIGKILLs the hung
            # peers; local copies were already wiped at event time.
            self._fail_worker(w, "trace_node_loss", wipe_local=False)
        while True:
            remaining = chain.end - self.backend.now
            if remaining <= 1e-12:
                break
            self._trace_reset_evt = reset_evt = self.backend.event()
            yield self.backend.timeout(remaining) | reset_evt
            self._trace_reset_evt = None
        # §3.3: every copy still in the books committed strictly before the
        # first kill (in-flight transfers were aborted at event time; the job
        # wrote nothing while down), so the newest surviving copy IS the
        # latest complete checkpoint. One common iteration for all cohorts.
        self._discard_future_checkpoints()
        target, peeks = self._common_recovery_plan()
        self._set_iteration(target, rollback=True)
        for w in self.workers.values():
            self._finish_worker(w)
        for _rank, (restored_it, tier) in peeks.items():
            booked = tier if restored_it >= 0 else "initial_state"
            self.restores[booked] += 1
            self._book_loss(booked, chain.it_at_kill - max(restored_it, 0))
        # §4 job-level accounting: a rolled-back iteration is lost work ONCE.
        self.trace_restarts += 1
        self.trace_lost_iters += chain.it_at_kill - max(target, 0)
        self.trace_recovery_s += self.backend.now - chain.t_kill
        self._trace_chain = None

    def trace_snapshot_at_window(self, window_s: float) -> dict:
        """EXP2_SPEC §4 accounting at t = W: useful_iters is the iteration
        index standing at the horizon — the live counter if the job is
        training, else the latest-complete-checkpoint index (a job mid-
        recovery at W gets credit only for checkpointed work, e.g. the
        3589.4 s event). goodput = useful_iters / W. Nothing here mutates
        durable state: the peek restores the live counters it touches."""
        chain = self._trace_chain
        if chain is None:
            useful = min(self.it, self.iterations)
        else:
            target, _peeks = self._common_recovery_plan()
            for w in self.workers.values():        # undo the plan's peeking
                w.current_iteration = self.it
            useful = max(target, 0)
        recovery_s = self.trace_recovery_s + (
            (window_s - chain.t_kill) if chain is not None else 0.0)
        lost = self.trace_lost_iters + (
            (chain.it_at_kill - useful) if chain is not None else 0)
        return {
            "useful_iters": int(useful),
            "lost_work_iters": int(lost),
            "recovery_s": round(recovery_s, 6),
            "checkpoint_s": round(self.trace_ckpt_stall_s, 6),
            "goodput_iters_per_s": round(useful / window_s, 9),
            "trace_events": int(self.trace_events_seen),
            "trace_events_ignored": int(self.trace_events_ignored),
            "restarts": int(self.trace_restarts),
            "resets": int(self.trace_resets),
            "recovering_at_window": chain is not None,
        }

    def _whole_job_eviction(self):
        """Spot semantics: every node of the job evicted at once; each cohort
        refetches its FULL m-rank state (size_scale=m) in parallel."""
        self.failure_types["preemption"] += 1
        failure_iteration = self.it
        self._invalidate_checkpoint_timeline()
        for w in self.workers.values():
            self._fail_worker(w, "spot", wipe_local=True, full_wipe=True)
            for s in self.all_strategies:          # fix #13: hosted shards die
                s.drop_hosted_copies(w.name)
        yield self.backend.timeout(self.failures.spot_restart_seconds)
        target, peeks = self._common_recovery_plan()
        self._set_iteration(target, rollback=True)

        def recover_one(w: CohortWorker):
            ok = yield from self.checkpoint_strategy.recover(
                w, self.config, size_scale=float(w.represents))
            if ok:
                self._finish_worker(w)
            return ok

        recoveries = [
            self.backend.process(recover_one(w)) for w in self.workers.values()
        ]
        outcomes = yield self.backend.all_of(recoveries)
        failed = [
            worker.rank
            for worker, recovery in zip(self.workers.values(), recoveries)
            if not outcomes[recovery]
        ]
        if failed:
            raise RuntimeError(
                f"whole-job recovery failed for {self.config.job_id} "
                f"cohorts={failed}"
            )
        # audit fix #10: one restore + one loss entry PER COHORT, mirroring
        # run_final73.analyze (which counts per restored worker)
        for i, (restored_it, tier) in peeks.items():
            t = tier if restored_it >= 0 else "initial_state"
            self.restores[t] += 1
            self._book_loss(t, failure_iteration - max(restored_it, 0))

    def _rack_failure(self, pending):
        """Rack event hit this job (rack_failure_spec.md + Sam's post-spec
        overrides): the dark nodes are gone for the REST OF THE RUN — no
        waiting branch — so the affected cohorts restart on REPLACEMENT nodes
        after the rack restart window and restore from the newest AVAILABLE
        copy (best off-rack copy; L3 if that is all; scratch only if nothing).
        The coordinator already destroyed/darkened everything the rack held at
        EVENT time; this is the boundary-aligned restart+restore, mirroring
        _whole_job_eviction's parallel-recovery shape for the affected subset."""
        affected: list[tuple[CohortWorker, int, float]] = []
        seen: set[int] = set()
        for w, n_dark, event_t in pending:
            if w.rank in seen:          # merged events on the same cohort
                continue
            seen.add(w.rank)
            affected.append((w, n_dark, event_t))
        self.failure_types["rack"] += 1
        failure_iteration = self.it
        self._invalidate_checkpoint_timeline()
        strat = self.checkpoint_strategy
        for w, n_dark, event_t in affected:
            if not w.failed:
                # local copies were already wiped at event time; replacements
                # start empty, so no wipe here (wipe_local=False).
                self._fail_worker(w, "rack", wipe_local=False)
            # anything captured/restored in the event->boundary window still
            # sits on the dark nodes — re-drop the late arrivals (same 1/m
            # located-copy sampling as the event-time wipe; no draws when the
            # cohort was fully dark, which keeps the handcheck deterministic).
            frac = n_dark / max(w.represents, 1)
            for key in [key for key, copy in strat.copies.items()
                        if key[1] == w.rank and copy.completed_at > event_t]:
                if frac >= 1.0 or self._rack_rng is None \
                        or self._rack_rng.random() < frac:
                    del strat.copies[key]
        yield self.backend.timeout(self._restart_seconds("rack"))
        self._discard_future_checkpoints()
        # subset "no time travel" plan (see _common_recovery_plan): every
        # affected cohort must restore ONE common iteration.
        target = self.it
        while True:
            for w, _n, _t in affected:
                w.current_iteration = target
            peeks = {w.rank: self._peek_restore(w) for w, _n, _t in affected}
            next_target = min(max(it, 0) for it, _tier in peeks.values())
            if next_target == target:
                break
            if next_target > target:
                raise AssertionError(
                    f"future checkpoint selected for {self.config.job_id}: "
                    f"target={target} selected={next_target}")
            target = next_target
        self._set_iteration(target, rollback=True)

        def recover_one(w, scale):
            ok = yield from self.checkpoint_strategy.recover(
                w, self.config, size_scale=scale)
            if ok:
                self._finish_worker(w)
            return ok

        recoveries = [
            self.backend.process(recover_one(w, float(max(n_dark, 1))))
            for w, n_dark, _t in affected
        ]
        outcomes = yield self.backend.all_of(recoveries)
        failed = [w.rank for (w, _n, _t), rec in zip(affected, recoveries)
                  if not outcomes[rec]]
        if failed:
            raise RuntimeError(
                f"rack recovery failed for {self.config.job_id} "
                f"cohorts={failed}")
        registry = getattr(strat, "registry", None)
        for w, _n, _t in affected:
            restored_it, tier = peeks[w.rank]
            t = tier if restored_it >= 0 else "initial_state"
            self.restores[t] += 1
            self._book_loss(t, failure_iteration - max(restored_it, 0))
            # the REPLACEMENTS are live donors again (the OLD nodes stay dark
            # for the rest of the run — they are no longer this slot); the
            # worker was relocated to rack-less at event time.
            slot = registry.slots.get(w.name) if registry is not None else None
            if slot is not None:
                slot.available = True


def preemption_process(backend, runtimes: list[CohortJobRuntime], *,
                       every_s: float, prefix: str, rng: random.Random):
    """model: one_of_class (LEGACY, kept for the sensitivity note).

    ONE healthy job of the victim class is evicted every `every_s`. The cluster
    rate is fixed, so it REDISTRIBUTES onto the survivors as jobs complete: with
    one job left that job is evicted every `every_s`. Real preemptible capacity
    does not work this way (see job_hazard_process); this model traps the last
    stragglers and manufactures "never finishes" outcomes."""
    victims = [rt for rt in runtimes if rt.config.job_id.startswith(prefix)]
    if not victims:
        return
    grace = every_s / 2.0
    while True:
        yield backend.timeout(every_s)
        eligible = [rt for rt in victims
                    if rt.end_time is None and rt.healthy
                    and backend.now - rt.last_failed_until >= grace]
        if eligible:
            rng.choice(eligible).request_preemption()


def job_hazard_process(
    backend,
    rt: CohortJobRuntime,
    *,
    mean_s: float,
    rng: random.Random,
    grace_s: float = 60.0,
):
    """model: per_job_hazard (DEFAULT) — ONE job's independent reclaim hazard.

    Real preemptible/spot capacity gives each VM an independent reclaim hazard:
    this job draws its next eviction from Exp(1/mean_s) regardless of how many
    siblings are still running, so the rate a survivor faces NEVER rises as its
    class completes. Grace (livelock guard only): a job must have been healthy
    for `grace_s` before it can be evicted again. The default is an absolute
    60 seconds; deliberately time-compressed stress scenarios may set a smaller
    value so the guard does not silently thin their requested hazard.
    A draw landing inside the grace window (or while the job is already down) is
    DROPPED, not queued — a reclaim notice for a VM that is already gone is not
    redelivered."""
    grace = min(max(0.0, grace_s), mean_s / 2.0)
    if rt.arrival_s > backend.now:
        yield backend.timeout(rt.arrival_s - backend.now)
    while rt.end_time is None:
        yield backend.timeout(rng.expovariate(1.0 / mean_s))
        if rt.end_time is not None:
            return
        if rt.healthy and backend.now - rt.last_failed_until >= grace:
            rt.request_preemption()


GRACE_S = 60.0     # livelock guard; see per-job hazard docstring


def start_preemption(backend, runtimes: list[CohortJobRuntime], *, pre: dict,
                     seed: int, configured_victim_count: int | None = None) -> None:
    """Wire the scenario's `preemption:` block onto the victim class.

    `preemption.model` selects the eviction model:
      per_job_hazard (default)  each victim independently draws Exp(mean) with
                                mean = every_s * count_of_class, so the AGGREGATE
                                cluster rate starts at 1/every_s (unchanged) but
                                does not concentrate on survivors;
      one_of_class              legacy redistributed cluster rate (above).
    Determinism: each victim's hazard RNG is seeded from (run seed, preemption
    seed, job_id) — the same scheme as the per-job failure RNGs, so a job's
    eviction history is reproducible and independent of its siblings."""
    every_s = float(pre["every_s"])
    prefix = str(pre["class_prefix"])
    pre_seed = int(pre.get("seed", 3))
    model = str(pre.get("model", "per_job_hazard"))
    victims = [rt for rt in runtimes if rt.config.job_id.startswith(prefix)]
    if not victims:
        return
    if model == "one_of_class":
        backend.process(preemption_process(
            backend, runtimes, every_s=every_s, prefix=prefix,
            rng=random.Random(pre_seed + seed)))
        return
    if model != "per_job_hazard":
        raise ValueError(f"unknown preemption.model {model!r}; "
                         "known: per_job_hazard | one_of_class")
    mean_s = every_s * (configured_victim_count or len(victims))
    grace_s = float(pre.get("grace_s", GRACE_S))
    for rt in victims:
        backend.process(job_hazard_process(
            backend, rt, mean_s=mean_s,
            rng=random.Random(f"{seed}:{pre_seed}:preempt:{rt.config.job_id}"),
            grace_s=grace_s))


# ------------------- rack failure domains (rack_failure_spec.md) ---------------

class RackTopology:
    """rack_id = node_index // rack_size over the CONTIGUOUS job packing.

    Node indices are assigned job-by-job in manifest order (declaration order
    for legacy scenarios), each cohort worker owning the same contiguous slice
    the divmod cohort split gives it — so a job's nodes span adjacent racks and
    a Gemini-style in-job ring neighbor is usually SAME rack (the bet the rack
    2x2 prices). Failure domain ONLY: no bandwidth/topology change. A worker
    hit by a rack event RELOCATES to replacement nodes and becomes rack-less
    (slice None): a repeat event on the old rack must not chase the moved job,
    and rack-less donors never conflict with anti-affinity."""

    def __init__(self, rack_size: int, node_count: int) -> None:
        self.rack_size = int(rack_size)
        self.node_count = int(node_count)
        self.n_racks = (self.node_count + self.rack_size - 1) // self.rack_size
        self.slices: dict[str, tuple[int, int] | None] = {}

    def assign(self, runtimes: list) -> None:
        base = 0
        for rt in runtimes:
            for index in sorted(rt.workers):
                worker = rt.workers[index]
                m = max(getattr(worker, "represents", 1), 1)
                self.slices[worker.name] = (base, base + m)
                base += m
        if base > self.node_count:
            raise ValueError(
                f"rack topology: {base} job nodes exceed cluster.node_count="
                f"{self.node_count}")

    def racks_of(self, name: str) -> frozenset:
        sl = self.slices.get(name)
        if not sl:
            return frozenset()
        lo, hi = sl
        rs = self.rack_size
        return frozenset(range(lo // rs, (hi - 1) // rs + 1))

    def overlap(self, name: str, lo: int, hi: int) -> int:
        """How many of this worker's CURRENT nodes lie in [lo, hi)."""
        sl = self.slices.get(name)
        if not sl:
            return 0
        wlo, whi = sl
        return max(0, min(whi, hi) - max(wlo, lo))

    def overlap_nodes(self, name: str, nodes: list[int]) -> int:
        sl = self.slices.get(name)
        if not sl:
            return 0
        wlo, whi = sl
        return sum(1 for n in nodes if wlo <= n < whi)

    def relocate(self, name: str) -> None:
        self.slices[name] = None


class RackFailureCoordinator:
    """Cluster-wide rack events (rack_failure_spec.md + Sam's post-spec
    overrides, 2026-07-27). On an event ONE rack's nodes go dark for the REST
    OF THE RUN (no waiting branch anywhere):

      - DRAM state on the rack is DESTROYED: L1 snapshots and in-job DRAM
        replicas (drop_local_copies over the located-copy sample), hosted
        L1.5/peerdram pieces (drop_hosted_piece, DRAM ledger freed);
      - SSD state (local copies + hosted stripe pieces) becomes UNAVAILABLE to
        every recovery-availability check and to the AvailabilityBoard for the
        rest of the run — removed from the live books, tallied as `dark`
        (or `destroyed` on a destructive-residue node: each node of the rack
        independently with p = rack_p_destroy is a permanent node loss);
      - affected ACTIVE jobs get a boundary-aligned rack restart on REPLACEMENT
        nodes (CohortJobRuntime._rack_failure) and restore from the newest
        AVAILABLE copy; their workers relocate to rack-less;
      - finished (idle-donor) jobs never re-provision: fully-dark cohorts leave
        the board for good, partially-dark cohorts keep donating from their
        surviving nodes;
      - a return timer (rack_outage_s, U(3600, 21600) default) is drawn and
        RECORDED per event for the outage bookkeeping, but typically fires
        beyond the horizon and nothing waits on it — jobs never wait.

    Event rate: rack_weight x the cluster-wide node-level ORGANIC rate (sum of
    per-node lambda over all configured job nodes) — independent of rack_size
    (bigger racks = bigger blast radius, same event count). Rack chosen
    uniformly at random per event over ceil(node_count / rack_size) racks.

    DETERMINISM (hard gate): every draw — event times, rack choice, outage,
    destructive residue, partial-cohort located-copy sampling — comes from the
    dedicated stream random.Random(f"{seed}:rack"). rack_weight == 0 with no
    scheduled events => ZERO draws; feature absent => this object is never
    built, so rack-less scenarios are byte-identical to the pre-rack code."""

    def __init__(self, *, backend, logger, registry, strategies, runtimes,
                 topology: RackTopology, rng: random.Random,
                 p_destroy: float, outage_s: tuple[float, float]) -> None:
        self.backend = backend
        self.logger = logger
        self.registry = registry
        self.strategies = strategies
        self.runtimes = runtimes
        self.topology = topology
        self.rng = rng
        self.p_destroy = float(p_destroy)
        self.outage_lo, self.outage_hi = float(outage_s[0]), float(outage_s[1])
        self.counters: Counter = Counter()

    def _relocate(self, name: str) -> None:
        self.topology.relocate(name)
        self.registry.racks_by_slot[name] = frozenset()

    def poisson_process(self, rate: float):
        while True:
            yield self.backend.timeout(self.rng.expovariate(rate))
            self.fire(self.rng.randrange(self.topology.n_racks),
                      source="poisson")

    def scheduled_process(self, events: list[dict]):
        """Deterministic forced rack events (`rack_scheduled`, the handcheck
        hook): explicit rack + destroy list + optional outage_s => zero draws."""
        now = 0.0
        for ev in events:
            at_s = float(ev["at_s"])
            yield self.backend.timeout(at_s - now)
            now = at_s
            outage = ev.get("outage_s")
            self.fire(int(ev["rack"]),
                      destroyed=[int(n) for n in ev.get("destroy", [])],
                      outage_end=(now + float(outage)
                                  if outage is not None else None),
                      source=str(ev.get("id", "scheduled")))

    def fire(self, rack_id: int, *, destroyed: list[int] | None = None,
             outage_end: float | None = None, source: str = "poisson") -> None:
        topo = self.topology
        rs = topo.rack_size
        lo, hi = rack_id * rs, min((rack_id + 1) * rs, topo.node_count)
        now = self.backend.now
        if outage_end is None:
            outage_end = now + self.rng.uniform(self.outage_lo, self.outage_hi)
        if destroyed is None:
            destroyed = [n for n in range(lo, hi)
                         if self.rng.random() < self.p_destroy]
        # membership snapshot BEFORE any relocation
        affected = []                    # (rt, worker, n_dark, n_destroyed)
        for rt in self.runtimes:
            for index in sorted(rt.workers):
                w = rt.workers[index]
                n_dark = topo.overlap(w.name, lo, hi)
                if n_dark <= 0:
                    continue
                affected.append(
                    (rt, w, n_dark, topo.overlap_nodes(w.name, destroyed)))
        self.counters["rack_events"] += 1
        self.counters["rack_nodes_dark"] += sum(n for _r, _w, n, _d in affected)
        self.counters["rack_nodes_destroyed"] += len(destroyed)
        for rt, w, n_dark, n_destroyed in affected:
            m = max(getattr(w, "represents", 1), 1)
            frac = n_dark / m
            self._relocate(w.name)       # replacements are rack-less hereafter
            slot = self.registry.slots.get(w.name)
            if rt.state == "pending":
                continue                 # not provisioned yet: it will arrive
                                         # on replacement nodes
            # local state on the dark nodes: DRAM destroyed; the SSD is dark
            # for the rest of the run and the job never re-attaches it
            # (survival ruling 2026-06-12: PD re-attach is out) — both drop.
            before = len(rt.checkpoint_strategy.copies)
            rt.checkpoint_strategy.drop_local_copies(
                w.rank, fraction=frac, rng=self.rng, tier=None)
            self.counters["rack_local_copies_lost"] += (
                before - len(rt.checkpoint_strategy.copies))
            # hosted pieces (every strategy) physically on the dark nodes:
            # DRAM-tier destroyed; SSD-tier dark (destroyed on residue nodes).
            # Same located-entry sampling as single-node loss (p = n_dark/m).
            for s in self.strategies:
                drop = getattr(s, "drop_hosted_piece", None)
                for key in [k for k in s.peer_copies if k[2] == w.name]:
                    if frac < 1.0 and self.rng.random() >= frac:
                        continue
                    meta = s.peer_copies.get(key)
                    tier = (meta.get("tier", "ssd")
                            if isinstance(meta, dict) else "ssd")
                    if tier == "dram":
                        self.counters["rack_dram_pieces_lost"] += 1
                    elif n_destroyed >= n_dark or (
                            n_destroyed > 0
                            and self.rng.random() < n_destroyed / n_dark):
                        self.counters["rack_ssd_pieces_destroyed"] += 1
                    else:
                        self.counters["rack_ssd_pieces_dark"] += 1
                    if drop is not None:  # frees the L1.5 DRAM ledger; the
                        drop(key, lost=True)   # dark disk never returns, so no
                    else:                      # shadow survives either way
                        del s.peer_copies[key]
            if rt.end_time is not None or rt.state != "active":
                # finished/censored job: nobody re-provisions it — fully-dark
                # cohorts leave the board for good (failed excludes them from
                # every availability check; the generation bump aborts any
                # in-flight transfer touching them, e.g. a piece mid-landing or
                # a drain reading the now-dark SSD); partial cohorts keep
                # donating from their surviving nodes.
                if n_dark >= m:
                    w.failed = True
                    w.failure_generation += 1
                    cap = getattr(rt.checkpoint_strategy, "capacity", None)
                    if cap is not None and getattr(
                            rt.checkpoint_strategy, "engine", None) == "event":
                        cap.ev_notify_worker_change(w)
                    if slot is not None:
                        slot.available = False
                continue
            # ACTIVE job: exclude the slot until the boundary restart brings
            # the replacements back; bump the generation NOW so every in-flight
            # transfer touching this worker (either direction) aborts. A worker
            # already down mid-recovery keeps its generation — the boundary
            # rack failure re-restores it on replacements anyway.
            if slot is not None:
                slot.available = False
            if not w.failed:
                w.failure_generation += 1
                cap = getattr(rt.checkpoint_strategy, "capacity", None)
                if cap is not None and getattr(
                        rt.checkpoint_strategy, "engine", None) == "event":
                    cap.ev_notify_worker_change(w)
            rt._rack_pending.append((w, n_dark, now))
        self.logger.record(
            start=now, end=now, run_id=0, job_id=None, rank=None,
            node=f"rack-{rack_id}", category="Failure",
            operation="rack_failure", failure_type="rack",
            details={
                "rack": rack_id, "node_lo": lo, "node_hi": hi,
                "source": source,
                "outage_end_s": round(float(outage_end), 3),
                "destroyed_nodes": sorted(destroyed),
                "affected": {w.name: n for _rt, w, n, _d in affected},
                "destroyed_by_worker": {w.name: d
                                        for _rt, w, _n, d in affected if d},
            })

    def summary(self) -> dict:
        return {
            "rack_events": int(self.counters["rack_events"]),
            "rack_nodes_dark": int(self.counters["rack_nodes_dark"]),
            "rack_nodes_destroyed": int(self.counters["rack_nodes_destroyed"]),
            "rack_local_copies_lost": int(
                self.counters["rack_local_copies_lost"]),
            "rack_dram_pieces_lost": int(
                self.counters["rack_dram_pieces_lost"]),
            "rack_ssd_pieces_dark": int(self.counters["rack_ssd_pieces_dark"]),
            "rack_ssd_pieces_destroyed": int(
                self.counters["rack_ssd_pieces_destroyed"]),
        }


def normalize_rack_scheduled(sc: dict) -> list[dict]:
    """Validate the deterministic `rack_scheduled` list (handcheck hook)."""
    raw = sc.get("rack_scheduled", [])
    if not isinstance(raw, list):
        raise ValueError("rack_scheduled must be a list")
    rack_size = int(sc["rack_size"])
    node_count = int(sc["cluster"]["node_count"])
    n_racks = (node_count + rack_size - 1) // rack_size
    sim_cap = float(sc.get("simulation", {}).get("max_sim_s", 144000.0))
    out, prev = [], 0.0
    for i, ev in enumerate(raw, start=1):
        if not isinstance(ev, dict):
            raise ValueError(f"rack_scheduled[{i}] must be a mapping")
        at_s = float(ev.get("at_s", -1.0))
        if not (prev <= at_s < sim_cap):
            raise ValueError(
                f"rack_scheduled[{i}].at_s must be ordered and inside "
                f"[0, max_sim_s)")
        prev = at_s
        rack = ev.get("rack")
        if not isinstance(rack, int) or not 0 <= rack < n_racks:
            raise ValueError(
                f"rack_scheduled[{i}].rack must be an int in [0, {n_racks})")
        destroy = ev.get("destroy", [])
        lo, hi = rack * rack_size, min((rack + 1) * rack_size, node_count)
        if not all(isinstance(n, int) and lo <= n < hi for n in destroy):
            raise ValueError(
                f"rack_scheduled[{i}].destroy must name nodes of rack {rack} "
                f"([{lo}, {hi}))")
        out.append(ev)
    return out


# ------------------------------ scenario assembly ------------------------------

def expand_jobs(sc: dict) -> list[tuple[str, dict]]:
    """[(job_id, class_spec)] in declaration order."""
    out = []
    for cname, spec in sc["classes"].items():
        for i in range(int(spec["count"])):
            out.append((f"{cname}{i}", spec))
    return out


def _arrival_count(value, *, label: str, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _arrival_time(value, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return result


def build_arrival_manifest(sc: dict, seed: int) -> list[JobArrival]:
    """Resolve legacy, scheduled, or weighted-random jobs for one scenario seed.

    The manifest is generated before an arm starts. Its RNG deliberately excludes
    the arm name, so every arm sees the same workload for a given scenario seed.
    """

    classes = sc.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("classes must be a nonempty mapping")
    caps = {
        cname: _arrival_count(
            spec.get("count"), label=f"classes.{cname}.count", positive=True
        )
        for cname, spec in classes.items()
    }
    block = sc.get("job_arrivals")
    if block is None:
        manifest: list[JobArrival] = []
        for cname, spec in classes.items():
            at_s = _arrival_time(
                spec.get("arrival_s", 0.0),
                label=f"classes.{cname}.arrival_s",
            )
            source = "legacy_delayed" if at_s > 0 else "initial"
            for index in range(caps[cname]):
                manifest.append(JobArrival(
                    job_id=f"{cname}{index}", class_name=cname,
                    arrival_s=at_s, source=source, batch=index,
                ))
        return manifest

    if not isinstance(block, dict):
        raise ValueError("job_arrivals must be a mapping")
    for cname, spec in classes.items():
        if "arrival_s" in spec:
            raise ValueError(
                f"classes.{cname}.arrival_s cannot be combined with job_arrivals"
            )

    mode = str(block.get("mode", "all_at_start")).strip().lower()
    if mode not in {"all_at_start", "scheduled", "weighted_random"}:
        raise ValueError(
            "job_arrivals.mode must be all_at_start, scheduled, or weighted_random"
        )
    if mode == "all_at_start":
        return [
            JobArrival(
                job_id=f"{cname}{index}", class_name=cname,
                arrival_s=0.0, source="initial", batch=0,
            )
            for cname in classes
            for index in range(caps[cname])
        ]

    sim_cap = float(sc.get("simulation", {}).get("max_sim_s", 144000.0))
    used = Counter()
    manifest: list[JobArrival] = []

    def add_jobs(cname: str, count, *, at_s: float, source: str, batch: int) -> None:
        if cname not in classes:
            raise ValueError(f"job_arrivals names unknown class {cname!r}")
        amount = _arrival_count(
            count, label=f"job_arrivals count for {cname!r}"
        )
        if used[cname] + amount > caps[cname]:
            raise ValueError(
                f"job_arrivals creates {used[cname] + amount} {cname!r} jobs, "
                f"exceeding classes.{cname}.count={caps[cname]}"
            )
        for _ in range(amount):
            index = used[cname]
            used[cname] += 1
            manifest.append(JobArrival(
                job_id=f"{cname}{index}", class_name=cname,
                arrival_s=at_s, source=source, batch=batch,
            ))

    initial = block.get("initial", {})
    if not isinstance(initial, dict):
        raise ValueError("job_arrivals.initial must be a mapping")
    for cname, count in initial.items():
        add_jobs(cname, count, at_s=0.0, source="initial", batch=0)

    if mode == "scheduled":
        events = block.get("events", [])
        if not isinstance(events, list):
            raise ValueError("job_arrivals.events must be a list")
        previous_at = 0.0
        for event_index, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                raise ValueError(
                    f"job_arrivals.events[{event_index}] must be a mapping"
                )
            at_s = _arrival_time(
                event.get("at_s"),
                label=f"job_arrivals.events[{event_index}].at_s",
            )
            if at_s < previous_at:
                raise ValueError("job_arrivals.events must be ordered by at_s")
            if at_s >= sim_cap:
                raise ValueError(
                    f"job_arrivals.events[{event_index}].at_s must be below "
                    f"simulation.max_sim_s={sim_cap}"
                )
            previous_at = at_s
            additions = event.get("add")
            if not isinstance(additions, dict) or not additions:
                raise ValueError(
                    f"job_arrivals.events[{event_index}].add must be a "
                    "nonempty mapping"
                )
            for cname, count in additions.items():
                add_jobs(
                    cname, count, at_s=at_s, source="scheduled",
                    batch=event_index,
                )
    else:
        random_cfg = block.get("random")
        if not isinstance(random_cfg, dict):
            raise ValueError("job_arrivals.random must be a mapping")
        start_s = _arrival_time(
            random_cfg.get("start_s"), label="job_arrivals.random.start_s"
        )
        end_s = _arrival_time(
            random_cfg.get("end_s"), label="job_arrivals.random.end_s"
        )
        if end_s <= start_s:
            raise ValueError("job_arrivals.random.end_s must be greater than start_s")
        if end_s >= sim_cap:
            raise ValueError(
                "job_arrivals.random.end_s must be below "
                f"simulation.max_sim_s={sim_cap}"
            )
        pool = random_cfg.get("pool")
        if not isinstance(pool, dict) or not pool:
            raise ValueError("job_arrivals.random.pool must be a nonempty mapping")
        remaining: dict[str, int] = {}
        weights: dict[str, float] = {}
        for cname, raw in pool.items():
            if cname not in classes:
                raise ValueError(f"job_arrivals names unknown class {cname!r}")
            if not isinstance(raw, dict):
                raise ValueError(
                    f"job_arrivals.random.pool.{cname} must be a mapping"
                )
            limit = _arrival_count(
                raw.get("limit"),
                label=f"job_arrivals.random.pool.{cname}.limit",
                positive=True,
            )
            if used[cname] + limit > caps[cname]:
                raise ValueError(
                    f"initial jobs plus random limit for {cname!r} exceed "
                    f"classes.{cname}.count={caps[cname]}"
                )
            try:
                weight = float(raw.get("weight", 1.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"job_arrivals.random.pool.{cname}.weight must be positive"
                ) from exc
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"job_arrivals.random.pool.{cname}.weight must be positive"
                )
            remaining[cname] = limit
            weights[cname] = weight
        capacity = sum(remaining.values())
        total = _arrival_count(
            random_cfg.get("total", capacity),
            label="job_arrivals.random.total",
            positive=True,
        )
        if total > capacity:
            raise ValueError(
                f"job_arrivals.random.total={total} exceeds pool capacity={capacity}"
            )
        arrival_seed = random_cfg.get("seed", 0)
        rng = random.Random(f"{seed}:{arrival_seed}:job-arrivals")
        times = sorted(rng.uniform(start_s, end_s) for _ in range(total))
        for event_index, at_s in enumerate(times, start=1):
            eligible = [cname for cname in pool if remaining[cname] > 0]
            cname = rng.choices(
                eligible, weights=[weights[name] for name in eligible], k=1
            )[0]
            remaining[cname] -= 1
            add_jobs(
                cname, 1, at_s=at_s, source="weighted_random",
                batch=event_index,
            )

    if not manifest:
        raise ValueError("job_arrivals must create at least one job")
    return sorted(manifest, key=lambda item: (item.arrival_s, item.batch))


def build_config(job_id: str, spec: dict) -> JobConfig:
    it_s = float(spec["iteration_seconds"])
    return JobConfig(
        job_id=job_id, strategy="data_parallel",
        data_parallel_replicas=int(spec["ranks"]), pipeline_stages=1, microbatches=1,
        model_weights_gb=float(spec.get("weights_gb_per_rank", 4.0)),
        gradient_gb=float(spec.get("weights_gb_per_rank", 4.0)),
        activation_gb=1.0,
        checkpoint_gb=float(spec["checkpoint_gb_per_rank"]),
        forward_seconds=round(0.4 * it_s, 3), backward_seconds=round(0.55 * it_s, 3),
        optimizer_seconds=round(0.05 * it_s, 3),
        checkpoint_every=int(spec["checkpoint_every"]),
        checkpoint_strategy="crossjob_peer",
        checkpoint_mode="asynchronous",
        checkpoint_chunk_gb=4.0, checkpoint_upload_gpu_slowdown=1.0,
        failure_rate=0.0)


def _scenario_work_progress(runtimes: list) -> tuple[float, int, int]:
    """Cohort-weighted work fraction, total jobs, and unfinished jobs."""

    total = 0
    completed = 0
    unfinished = 0
    for runtime in runtimes:
        cohorts = max(1, len(runtime.workers))
        total += runtime.iterations * cohorts
        completed += min(max(runtime.it, 0), runtime.iterations) * cohorts
        if runtime.end_time is None:
            unfinished += 1
    fraction = completed / total if total else 1.0
    return fraction, len(runtimes), unfinished


def _format_wall_time(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    whole = int(round(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_heartbeat(
    backend,
    runtimes: list,
    path: Path,
    stop: threading.Event,
    *,
    label: str,
    interval: float,
) -> None:
    """Print and persist a live progress bar with a wall-clock ETA."""

    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "wall_s sim_s work_pct active_jobs sim_rate eta_s\n",
        encoding="utf-8",
    )
    while not stop.wait(interval):
        elapsed = max(time.monotonic() - started, 1e-9)
        fraction, total_jobs, unfinished = _scenario_work_progress(runtimes)
        pending = sum(
            getattr(runtime, "state", "active") == "pending"
            for runtime in runtimes
        )
        active = max(0, unfinished - pending)
        sim_now = float(backend.now)
        sim_rate = sim_now / elapsed
        eta = (
            elapsed * (1.0 - fraction) / fraction
            if 0.0 < fraction < 1.0
            else (0.0 if fraction >= 1.0 else None)
        )
        width = 28
        filled = min(width, max(0, round(width * fraction)))
        bar = "#" * filled + "-" * (width - filled)
        line = (
            f"[{label}] [{bar}] {100.0 * fraction:5.1f}% "
            f"jobs {total_jobs - unfinished}/{total_jobs} "
            f"(active {active}, pending {pending}) "
            f"sim {sim_now:,.0f}s ({sim_rate:,.1f}x) "
            f"elapsed {_format_wall_time(elapsed)} "
            f"ETA ~{_format_wall_time(eta)}"
        )
        print(line, file=sys.stderr, flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{elapsed:.1f} {sim_now:.3f} {100.0 * fraction:.3f} "
                f"{unfinished} {sim_rate:.6f} "
                f"{eta if eta is not None else -1.0:.1f}\n"
            )


# module-level cache: cost-model solve is deterministic per cluster composition,
# so re-solves for a composition seen before (any seed/arm in this process) reuse
# the result. Keyed by the sorted (class, active_count) signature.
_RESOLVE_CACHE: dict = {}


class ChurnController:
    """Stage 3 (Mark D7 dynamic half): on every job ARRIVAL or DEPARTURE, RE-SOLVE
    the cost-model policy for the CURRENT active cluster composition and enact the
    new per-class cadence/k on the RUNNING jobs (taking effect at their next wave).
    This is the "dynamically recalculates" controller: a new important job arriving
    shrinks everyone else's donor share; a job finishing frees it back up. The
    frozen-policy baseline runs NO controller and keeps its one static solve."""

    def __init__(self, backend, sc: dict, runtimes: list, strategy_cls, logger):
        self.backend = backend
        self.sc = sc
        self.runtimes = runtimes
        self.strategy_cls = strategy_cls
        self.classes = list(sc["classes"])
        self.class_by_job = {
            rt.config.job_id: rt.class_name for rt in runtimes
        }
        self.active = {rt for rt in runtimes if rt.arrival_s <= 0}
        self.actions: list = []

    def _class_of(self, job_id: str) -> str:
        return self.class_by_job[job_id]

    def on_arrival(self, rt) -> None:
        self.active.add(rt)
        self.resolve(f"arrival:{rt.config.job_id}")

    def on_departure(self, rt) -> None:
        self.active.discard(rt)
        self.resolve(f"departure:{rt.config.job_id}")

    def resolve(self, reason: str) -> None:
        import gp_policy
        running = [rt for rt in self.active if rt.end_time is None]
        counts = Counter(self._class_of(rt.config.job_id) for rt in running)
        if not counts:
            return
        sig = tuple(sorted(counts.items()))
        pol = _RESOLVE_CACHE.get(sig)
        if pol is None:
            sub = {**self.sc, "classes": {
                c: {**self.sc["classes"][c], "count": n, "arrival_s": 0}
                for c, n in counts.items()}}
            pol = gp_policy.solve_dict(sub, name="resolve")
            _RESOLVE_CACHE[sig] = pol
        by_class: dict = {}
        for _jid, pj in pol["jobs"].items():
            by_class.setdefault(pj["class"], pj)
        for rt in running:
            pj = by_class.get(self._class_of(rt.config.job_id))
            if not pj:
                continue
            rt.config = dataclasses.replace(
                rt.config, checkpoint_every=int(pj["checkpoint_every"]))
            self.strategy_cls.kpeers_by_job[rt.config.job_id] = int(pj["kpeers"])
            rt.capture_every = pj.get("capture_every")
            rt.store_every = pj.get("store_every")
            if hasattr(rt.checkpoint_strategy, "drain_every_waves"):
                rt.checkpoint_strategy.drain_every_waves = max(
                    1,
                    round(
                        (pj.get("store_every") or 1)
                        / max(pj.get("checkpoint_every") or 1, 1)
                    ),
                )
        self.actions.append({
            "t": round(self.backend.now, 1), "reason": reason,
            "sig": [list(x) for x in sig],
            "k": {c: by_class[c]["kpeers"] for c in by_class},
            "every": {c: by_class[c]["checkpoint_every"] for c in by_class}})


def run_arm(sc: dict, arm: str, seed: int, trace_path: Path | None,
            policy: dict | None = None, progress_path: Path | None = None,
            backstop: str | None = None,
            progress_interval: float = 10.0,
            event_batch_size: int = 50_000) -> dict:
    """policy: gp_policy.py output — the cost model's open-loop decisions
    (per-job checkpoint_every / kpeers / slot). Overrides the YAML's hand-set
    cadences; the closed loop still refines slots from measured durations."""
    # D2 (Mark 2026-07-23): resolve parallelism-configured classes to their
    # no-dedup per-GPU shard (checkpoint_gb_per_rank), in place + idempotent.
    # Classes without a `model:` block are untouched (legacy scenarios exact).
    parallelism.apply_model_shards(sc)
    arm_cfg = dict(policy["flags"] if policy else sc["arms"][arm])
    requested_arm_cfg = dict(arm_cfg)
    ideal_mode = bool(arm_cfg.get("ideal", False))
    # DP-aware checkpointing (OURS ONLY): a policy/arm flag `dp_dedup: true`
    # replaces each parallelism-configured class's per-rank flush/capture/recovery
    # bytes with the effective shard (unique + a rotating replicated/dp slice).
    dp_dedup = bool(arm_cfg.get("dp_dedup", False))
    # named-baseline dispatch: an arm `{baseline: <name>}` selects a strategy
    # class, forces class attributes, merges in the baseline's arm flags, and
    # supplies a per-job frequency rule. Non-baseline arms are unaffected.
    bl = baselines.resolve(arm_cfg)
    strategy_cls: type = CrossJobPeerStrategy
    baseline_every_rule = None
    baseline_store_rule = None
    baseline_driver = None
    if bl is not None:
        strategy_cls = bl.strategy_cls
        baseline_every_rule = bl.every_rule
        baseline_store_rule = bl.store_every_rule
        baseline_driver = bl.driver
        for _k, _v in bl.class_attrs.items():
            setattr(CrossJobPeerStrategy, _k, _v)
        arm_cfg = {**bl.arm_flags,
                   **{k: v for k, v in arm_cfg.items() if k != "baseline"}}
    # jit_pc_fallback (fairness fix 2026-07-28, ON by default for the jit arm;
    # `jit_pc_fallback: false` in the arm cfg reproduces the archived rows):
    # JIT's paper (Gupta et al. EuroSys'24 Sec. 5.2) degrades to PERIODIC
    # checkpointing at c* = sqrt(N*f/(2*o)) (eq. 3) where its transparent
    # mechanism cannot reconstruct state. Swap the arm's every_rule so classes
    # with no replicated state get the c* cadence (replicated/legacy classes
    # keep _never — replica path byte-untouched). f = the scenario's
    # ACCELERATED per-node organic rate, the same one every arm sees.
    _jit_pc_rate = float(sc["failures"]["per_node_per_second"])
    jit_pcfb = (bl is not None and bl.name == "jit"
                and bool(arm_cfg.get("jit_pc_fallback", True)))
    if jit_pcfb:
        baseline_every_rule = baselines.jit_pc_fallback_rule(_jit_pc_rate)
    jit_pc_by_class: dict[str, dict] = {}   # documented in the results row
    # Stage 3 churn: a `resolve` arm flag makes the driver solve the cost model
    # in-process. Both churn arms start from the FULL-cluster solve (the static
    # plan); resolve:true additionally re-solves per active composition at each
    # arrival/departure, resolve:false freezes it. arm_cfg keeps the arm's own
    # flags (resolve/slots/...), unlike the --policy path which overrides them.
    churn_mode = "resolve" in arm_cfg and policy is None
    do_resolve = bool(arm_cfg.get("resolve"))
    if churn_mode:
        import gp_policy
        full_sig = ("__full__", tuple(sorted(
            (c, int(s["count"])) for c, s in sc["classes"].items())))
        policy = _RESOLVE_CACHE.get(full_sig)
        if policy is None:
            policy = gp_policy.solve_dict(sc, name="churn_init")
            _RESOLVE_CACHE[full_sig] = policy
    period = float(sc["controller"]["slot_period_s"])
    CrossJobPeerStrategy.kpeers_by_job = {}
    CrossJobPeerStrategy.async_persist = True
    CrossJobPeerStrategy.capacity_mode = True
    # capacity engine: scenario `simulation.engine` (event|polling, default event);
    # env var SIM_ENGINE still overrides per resolve_engine() for quick A/B.
    CrossJobPeerStrategy.engine = sc.get("simulation", {}).get("engine")
    CrossJobPeerStrategy.store_mode = bool(arm_cfg.get("store_mode", False))
    CrossJobPeerStrategy.slot_period = period if arm_cfg.get("slots") else None
    CrossJobPeerStrategy.slot_by_job = {}
    CrossJobPeerStrategy.rates_by_job = {}
    # centralized peer matching (Mark D8/D9): arm knobs -> AvailabilityBoard mode
    CrossJobPeerStrategy.matching = str(arm_cfg.get("matching", "registry"))
    CrossJobPeerStrategy.spread_jobs = bool(arm_cfg.get("spread_jobs", False))
    CrossJobPeerStrategy.window_bias = bool(arm_cfg.get("window_bias", False))
    # durability backstop mechanism: donor_drain L3 (default for ours arms;
    # scenario-level `backstop:` sets the default, an arm's `backstop:` wins) vs
    # legacy owner_push. Named baselines always use owner_push (their remote-store
    # backstop rides backstop_upload on the store_every cadence).
    CrossJobPeerStrategy.backstop = (
        backstop if backstop                       # explicit CLI/caller override
        else "owner_push" if bl is not None
        # NO-PEER arms cannot drain (no donor pieces exist): donor_drain would
        # silently disable L3 entirely — the mini-C ablation caught every
        # kpeers:false arm with identical be+694%/300-scratch fingerprints
        # regardless of L3 cadence (2026-07-20). They fall back to owner_push.
        else "owner_push" if not arm_cfg.get("kpeers", True)
        else str(arm_cfg.get("backstop", sc.get("backstop", "donor_drain"))))
    DonorRegistry.reset(0)
    CapacityRegistry.reset(0)
    DonorRegistry.for_run(0).rng = random.Random(f"{seed}:probe")

    # L1.5 peer-DRAM capacity (l15_peer_dram_spec.md): per-node daemon-owned DRAM
    # budget — model.dram_host_slots shard-pieces (default 1) and
    # model.dram_host_gb bytes (default 2x the largest class piece at its
    # configured stripe width). Enforced at reservation; inert unless some job
    # enacts placement `peerdram` (dram_by_job below).
    CrossJobPeerStrategy.dram_by_job = {}
    # injob_ssd placement (rack_failure_spec.md): per-run enactment map — reset
    # so one arm's placements never leak into the next (dict empty => inert).
    CrossJobPeerStrategy.injob_ssd_by_job = {}
    _mdl = sc.get("model") or {}
    # FABRIC-AWARE COUPLING (sim_validation_protocol.md CL-012.D S1/S2, CL-014).
    # DEFAULT OFF: absent from BOTH the arm flags and the scenario `model:` block
    # => the fabric-blind count-based coupling every committed result was
    # produced under, on the identical code path. The arm-level key exists so one
    # scenario can carry the A/B (see scenarios/handcheck_fabric.yaml).
    CohortJobRuntime.fabric_aware_coupling = bool(
        arm_cfg.get("fabric_aware_coupling",
                    _mdl.get("fabric_aware_coupling", False)))
    # INTENSITY COUPLING (sim_validation_protocol.md CL-012.D S3, CL-016).
    # DEFAULT OFF, exactly parallel to fabric_aware_coupling above: absent from
    # BOTH the arm flags and the scenario `model:` block => the pre-CL-016
    # coupling on the identical code path. ON implies fabric-scoped semantics
    # (the intensity branch reads only the collective's own fabric).
    CohortJobRuntime.intensity_coupling = bool(
        arm_cfg.get("intensity_coupling",
                    _mdl.get("intensity_coupling", False)))
    # NON-BLOCKING CAPTURE (sim_validation_protocol.md CL-017). DEFAULT OFF,
    # exactly parallel to the two coupling flags above: absent from BOTH the
    # arm flags and the scenario `model:` block => the capture-as-GPU-lock
    # model every committed result was produced under, on the identical code
    # path. ON: the D2H capture takes no GPU lock (exp1: capture 413 ms vs
    # 485 ms iteration, p99 stretch 1.002 — the capture does not stall
    # training), while its DURATION still gates the persist pipeline. The
    # lock lives on the strategy, not the runtime: tiered.checkpoint() and
    # crossjob.snapshot() read this class attribute.
    CrossJobPeerStrategy.nonblocking_capture = bool(
        arm_cfg.get("nonblocking_capture",
                    _mdl.get("nonblocking_capture", False)))
    _reg = DonorRegistry.for_run(0)
    _reg.dram_host_slots = int(_mdl.get("dram_host_slots", 1))
    _largest_piece = max(
        float(spec["checkpoint_gb_per_rank"])
        / max(int(spec.get("kpeers", 2)) or 1, 1)
        for spec in sc["classes"].values())
    _reg.dram_host_gb = float(_mdl.get("dram_host_gb", 2.0 * _largest_piece))

    # Route C — per-node persistent background front-end traffic. Set on the
    # registry BEFORE any node is registered, so every node (store below + every
    # worker in register_worker) gets its standing ingress demand installed.
    # cluster.background_nic_gbps defaults to 0.0 => nothing installed, existing
    # scenarios byte-identical (workload_sources_dossier.md "Dual-fabric evidence";
    # Mark meeting D3). --background-nic-gbps CLI overrides it into sc["cluster"].
    CapacityRegistry.for_run(0).background_nic_gbps = float(
        sc.get("cluster", {}).get("background_nic_gbps", 0.0))

    store = sc.get("store", {})
    CrossJobPeerStrategy.store_stream_gbps = float(store.get("stream_gbps", 999.0))
    CapacityRegistry.for_run(0).set_node(STORE_NODE, NodeCaps(
        nic_in=float(store.get("in_gbps", 999.0)),
        nic_out=float(store.get("out_gbps", 999.0)),
        disk_w=float(store.get("disk_gbps", 999.0))))

    cl = sc["cluster"]
    cluster = ClusterConfig(
        node_count=int(cl["node_count"]), cpu_cores_per_node=1,
        cpu_memory_gb=float(cl.get("cpu_memory_gb", 3.75)), gpus_per_node=1,
        gpu_memory_gb=80.0,
        gpu_cpu_bandwidth_gbps=float(cl["gpu_cpu_bandwidth_gbps"]),
        network_bandwidth_gbps=float(cl["network_bandwidth_gbps"]),
        communication_launch_seconds=0.01,
        local_ssd_bandwidth_gbps=float(cl["local_ssd_bandwidth_gbps"]),
        object_store_bandwidth_gbps=float(cl.get("object_store_bandwidth_gbps", 8.0)),
        object_store_concurrency=4)
    f = sc["failures"]
    failures = FailureSettings(
        probability_per_second=0.0,
        process_weight=float(f["weights"]["process"]),
        node_weight=float(f["weights"]["node"]),
        spot_weight=float(f["weights"]["spot"]),
        # reboot (2026-07-21): optional 4th organic class; absent => weight 0, so
        # every pre-reboot scenario is byte-identical.
        reboot_weight=float(f["weights"].get("reboot", 0.0)),
        process_restart_seconds=float(f["restart_seconds"]["process"]),
        node_restart_seconds=float(f["restart_seconds"]["node"]),
        spot_restart_seconds=float(f["restart_seconds"]["spot"]),
        reboot_restart_seconds=float(f["restart_seconds"].get("reboot", 180.0)),
        # rack (rack_failure_spec.md): replacement provisioning after a rack
        # event; absent => node_restart_seconds (see _restart_seconds).
        rack_restart_seconds=(
            float(f["restart_seconds"]["rack"])
            if "rack" in f["restart_seconds"] else None))
    inject_random_failures = bool(f.get("inject_random", True))
    per_node_rate = (
        0.0
        if ideal_mode or not inject_random_failures
        else float(f["per_node_per_second"])
    )

    env = simpy.Environment()
    backend = SimPyBackend(env)
    logger = BatchedScenarioEventLogger(
        trace_path=trace_path,
        batch_size=event_batch_size,
    )
    object_store = backend.resource(cluster.object_store_concurrency)
    rng_phase = random.Random(seed + 1000)

    # Stage 4 (Mark D5): per-class L2 PLACEMENT. crossjob (default) = cross-job
    # donor striping; injob = Gemini-style in-job DRAM replication; auto = the
    # solver's choice, here an eviction-hazard rule -- in-job copies DIE on
    # whole-job loss, so injob is chosen only for GUARANTEED classes (not the
    # preemption victim) whose replicas survive the common single-node failure;
    # the preempted class stays crossjob (its donors live in OTHER jobs, which
    # survive its eviction). Only for non-baseline arms; default leaves every
    # existing scenario on crossjob (GeminiStrategy never selected => unchanged).
    placement = str(arm_cfg.get("placement", "crossjob")) if bl is None else "crossjob"
    victim = str((sc.get("preemption") or {}).get("class_prefix", "\x00"))

    arrival_manifest = build_arrival_manifest(sc, seed)
    arrivals_by_job = {arrival.job_id: arrival for arrival in arrival_manifest}
    jobs = [
        (arrival.job_id, sc["classes"][arrival.class_name])
        for arrival in arrival_manifest
    ]
    dynamic_arrivals = "job_arrivals" in sc
    scheduled_by_job = normalize_scheduled_failures(sc, jobs)
    # deterministic failure-trace replay (EXP2_SPEC.md, CL-018): None unless the
    # scenario declares failures.trace_file — validated always, wired only for
    # non-ideal runs (ideal disables every failure class, trace included).
    failure_trace = normalize_failure_trace(sc, jobs)
    trace_active = failure_trace is not None and not ideal_mode
    runtimes: list[CohortJobRuntime] = []
    for job_id, spec in jobs:
        cfg = build_config(job_id, spec)
        # D2 DP-aware checkpointing: for a parallelism-configured class under an
        # ours+dedup policy, the controller rotates a 1/dp slice of the DP-
        # REPLICATED bytes to each rank, so per-rank persisted bytes drop from the
        # full shard to unique + replicated/dp. Reducing checkpoint_gb here prices
        # the saving in EVERY byte path — capture (GPU->DRAM), peer/store flush,
        # backstop, and recovery reads (the lost rank restores its slice from the
        # durable tier; the replicated remainder is gathered from a surviving DP
        # peer's DRAM or the DP group's other slices, off the critical restore).
        # Deterministic (pure function of the config -> byte-identical reruns).
        if dp_dedup and "model" in spec:
            eff = parallelism.effective_shard_gb(
                parallelism.parse_model(spec["model"]), dedup=True)
            cfg = dataclasses.replace(cfg, checkpoint_gb=eff)
        kp = (int(arm_cfg.get("replication_degree", spec.get("kpeers", 2)))
              if arm_cfg.get("kpeers", True) else 0)
        # arm-level frequency override: arms like {checkpoint_every: 100}
        # sweep NAIVE fixed cadences against the controller-solved policy. An
        # explicit arm value wins; else a baseline's per-job frequency rule
        # (checkfreq profiling, checknrun 30-min, megascale 300 s) applies.
        naive_all_tiers = None
        naive_tiers = None
        if ideal_mode:
            # Analytic comparator executed through the real runtime: retain
            # compute + all-reduce, but disable periodic checkpoints and all
            # failure/preemption injection. `it < iterations` means this
            # cadence can never fire.
            kp = 0
            cfg = dataclasses.replace(
                cfg, checkpoint_every=int(spec["iterations"]) + 1
            )
        elif arm_cfg.get("cadence_l2_s") and not arm_cfg.get("baseline"):
            # SEMI-NAIVE hand-set per-tier cadences (arms like
            # {cadence_l1_s: 300, cadence_l2_s: 1000, cadence_l3_s: 3000}):
            # tiered like ours, but numbers chosen by hand, not solved.
            from checkpointing.baselines import iteration_wall_seconds
            iw = iteration_wall_seconds(spec, cl)

            def it(seconds):
                return max(1, round(float(seconds) / iw))

            naive_tiers = (it(arm_cfg.get("cadence_l1_s", arm_cfg["cadence_l2_s"])),
                           it(arm_cfg["cadence_l2_s"]),
                           it(arm_cfg.get("cadence_l3_s", arm_cfg["cadence_l2_s"])))
            cfg = dataclasses.replace(cfg, checkpoint_every=naive_tiers[1])
        elif arm_cfg.get("cadence_s") and not arm_cfg.get("baseline"):
            # NAIVE fixed wall-clock cadence hitting ALL THREE tiers at once
            # (arms like {cadence_s: 100}): the "one number for everything"
            # policy an operator would hand-configure — snapshot, peer flush
            # and store backstop all fire on the same interval, for every job.
            from checkpointing.baselines import iteration_wall_seconds
            naive_all_tiers = max(1, round(float(arm_cfg["cadence_s"])
                                           / iteration_wall_seconds(spec, cl)))
            cfg = dataclasses.replace(cfg, checkpoint_every=naive_all_tiers)
        elif arm_cfg.get("checkpoint_every"):
            cfg = dataclasses.replace(
                cfg, checkpoint_every=int(arm_cfg["checkpoint_every"]))
        elif baseline_every_rule is not None:
            cfg = dataclasses.replace(
                cfg, checkpoint_every=int(baseline_every_rule(spec, cl)))
        if policy:
            pj = policy["jobs"][job_id]
            kp = int(pj["kpeers"])
            cfg = dataclasses.replace(
                cfg, checkpoint_every=int(pj["checkpoint_every"]))
        CrossJobPeerStrategy.kpeers_by_job[job_id] = kp
        if "rates" in spec:
            CrossJobPeerStrategy.rates_by_job[job_id] = dict(spec["rates"])
        # PER-CLASS L2 PLACEMENT (priced placement 2026-07-21). A placement-solved
        # policy carries the priced per-job choice (crossjob|injob|local); it wins
        # over the arm-global `placement` knob + eviction heuristic (which stays
        # for non-policy arms: ours_injob / ours_auto). crossjob = donor stripe;
        # injob = Gemini in-job DRAM replica (+ owner-push store backstop, its
        # tier-3 that survives eviction); local = own SSD, k=0 (backstop switches
        # to owner-push per-CLASS -- no donor pieces exist to drain).
        job_strategy_cls = strategy_cls
        arrival = arrivals_by_job[job_id]
        cname = arrival.class_name
        # AUDIT FIX (2026-07-25): the k=0 guard was dead code — `int(k or 1)`
        # coerces numeric 0 to 1, and JSON `"placement": null` took the first
        # branch (job_placement=None -> crossjob route). Correct rule: any job
        # whose EFFECTIVE kpeers is 0 and whose placement is absent/null/
        # crossjob MUST take the local route (a zero-width cross-job stripe is
        # not a checkpoint path, and donor_drain would be a silently dead L3).
        _pj0 = policy["jobs"].get(job_id, {}) if policy else {}
        _pol_placement = _pj0.get("placement")        # JSON null -> None
        _pol_k = _pj0.get("kpeers")
        _pol_k = int(_pol_k) if _pol_k is not None else None
        if policy and _pol_k == 0 and _pol_placement in (None, "crossjob"):
            job_placement = "local"
        elif policy and _pol_placement is not None:
            job_placement = _pol_placement
        elif placement == "auto":
            job_placement = "injob" if not cname.startswith(victim) else "crossjob"
        else:
            job_placement = placement          # crossjob (default) or injob arm
        job_backstop = None                    # per-job override (None = arm class)
        if bl is None and job_placement == "injob":
            job_strategy_cls = baselines.GeminiStrategy
            CrossJobPeerStrategy.kpeers_by_job[job_id] = min(kp, INJOB_DEGREE)
            job_backstop = "owner_push"        # Gemini tier-3 (survives eviction)
        elif bl is None and job_placement == "local":
            CrossJobPeerStrategy.kpeers_by_job[job_id] = 0
            job_backstop = "owner_push"        # k=0: no pieces to drain -> push
        elif bl is None and job_placement == "injob_ssd":
            # in-job SSD replica (rack_failure_spec.md): ring-neighbor node of
            # the SAME job, k=1, on that node's SSD. No donor duty/slots, no
            # DRAM rent; survives owner process failure and host reboot (disk),
            # dies with the host node, goes dark with the host rack. Backstop
            # owner-push: no cross-job pieces exist to drain.
            CrossJobPeerStrategy.injob_ssd_by_job[job_id] = True
            CrossJobPeerStrategy.kpeers_by_job[job_id] = 1
            job_backstop = "owner_push"
        elif bl is None and job_placement == "peerdram":
            # L1.5 peer-DRAM (l15_peer_dram_spec.md): the SAME cross-job stripe
            # machinery, but pieces land in donor daemon DRAM at NIC_in and then
            # demote to the host SSD in the background (becoming ordinary L2).
            # kpeers stays the solved k; the donor-drain L3 backstop chains
            # after demotion inside the strategy, so the arm backstop is kept.
            CrossJobPeerStrategy.dram_by_job[job_id] = True
        rt = CohortJobRuntime(
            config=cfg, class_name=cname, spec=spec, cluster=cluster,
            failures=failures,
            per_node_fail_rate=per_node_rate, logger=logger, backend=backend,
            object_store=object_store, rng=random.Random(f"{seed}:{job_id}"),
            strategy_cls=job_strategy_cls, baseline=baseline_driver,
            scheduled_failures=(
                [] if ideal_mode else scheduled_by_job[job_id]
            ))
        if job_backstop is not None:
            # per-CLASS backstop override (instance attr shadows the arm-global
            # class attr): local/injob push whole shards to the store; crossjob
            # keeps the arm default (donor_drain). Fires only when store_every is
            # set (below), so non-policy injob/local arms are unaffected.
            rt.checkpoint_strategy.backstop = job_backstop
        if jit_pcfb and baselines.jit_pc_engaged(spec, _jit_pc_rate):
            # jit_pc_fallback engaged for THIS class (no replicated state for
            # JIT's survivor drain): periodic checkpoints at the c* cadence
            # (already in cfg.checkpoint_every via jit_pc_fallback_rule) land on
            # the shared L3 store — instance store_mode shadows the arm-global
            # False, exactly the megascale/checknrun persist mechanics. Recovery
            # = newest store copy on the existing generic path (the runtime flag
            # skips the survivor-drain-else-scratch jit branch). Classes NOT
            # engaged never reach here: no flag, no store_mode, no new RNG draws
            # — the replica path stays byte-identical to the archived arm.
            rt.jit_pc_fallback = True
            rt.checkpoint_strategy.store_mode = True
            jit_pc_by_class.setdefault(
                cname, baselines.jit_pc_cstar(spec, cl, _jit_pc_rate))
        if policy:
            pj = policy["jobs"][job_id]
            rt.capture_every = pj.get("capture_every")
            rt.store_every = pj.get("store_every")
            # drain gating: donors forward pieces at the SOLVED f3 cadence,
            # not every wave (store ingest honesty; see _schedule_wave_drain)
            if hasattr(rt.checkpoint_strategy, "drain_every_waves"):
                rt.checkpoint_strategy.drain_every_waves = max(
                    1,
                    round(
                        (pj.get("store_every") or 1)
                        / max(pj.get("checkpoint_every") or 1, 1)
                    ),
                )
        elif naive_all_tiers is not None:
            rt.capture_every = naive_all_tiers      # same cadence, all tiers
            rt.store_every = naive_all_tiers
        elif naive_tiers is not None:
            rt.capture_every = min(naive_tiers[0], naive_tiers[1])
            rt.store_every = max(naive_tiers[2], naive_tiers[1])
        elif baseline_store_rule is not None:
            # baseline's own periodic store backstop (gemini's remote persistent
            # store). Rides after a persist wave like our f3, which is exact here
            # because such baselines flush every iteration (every_rule=_every_one)
            # -> every iteration is a wave. An arm-level `checkpoint_every`
            # override would break that alignment; none of ours sets one.
            rt.store_every = int(baseline_store_rule(spec, cl))
        # sim-vs-real validation hook: `simulation.emit_iterations: true` makes
        # every runtime log one METRICS.md §1 record per iteration. Absent =>
        # False => existing scenarios byte-identical (see _record_iteration).
        rt.emit_iterations = bool(
            sc.get("simulation", {}).get("emit_iterations", False))
        rt.arrival_s = arrival.arrival_s
        rt.arrival_source = arrival.source
        if trace_active:
            # every job accrues §4 accounting (job B's goodput is the
            # cross-check even though N=2 traces never target it)
            rt.trace_replay = failure_trace
        for w in rt.workers.values():
            rt.checkpoint_strategy.register_worker(w, job_id)
        if rt.arrival_s > 0:            # Stage 3: nodes exist but can't donate yet
            rt.checkpoint_strategy.registry.set_available(job_id, False)
        runtimes.append(rt)

    strategies = [rt.checkpoint_strategy for rt in runtimes]
    for rt in runtimes:
        rt.all_strategies = strategies

    # trace replay processes (CL-018): one driver per targeted job — so equal
    # t_s on DIFFERENT jobs fail independently in parallel (EXP2_SPEC §5) —
    # plus one window monitor snapshotting §4 accounting at t = W. Spawned
    # ONLY when failures.trace_file is present: absent => zero new processes.
    trace_snapshot: dict[str, dict] = {}
    if trace_active:
        by_id = {rt.config.job_id: rt for rt in runtimes}
        for job_id, events in failure_trace.events_by_job.items():
            backend.process(trace_replay_driver(backend, by_id[job_id], events))
        backend.process(trace_window_monitor(
            backend, runtimes, failure_trace, trace_snapshot))

    # ---- rack failure domains (rack_failure_spec.md, 2026-07-27) --------------
    # Feature keys on `rack_size` presence. Knobs (exact names): rack_size
    # (default 4 in the spec, but the feature is OFF unless the key exists),
    # rack_weight (0.02), rack_p_destroy (0.01), rack_outage_s ([3600, 21600]
    # uniform). ZERO rack-stream RNG draws unless events actually fire, so
    # every rack-less scenario (and any rack_weight=0 run without scheduled
    # events) is byte-identical to the pre-rack code — hard regression gate.
    rack_coord = None
    if sc.get("rack_size") is not None and not ideal_mode:
        rack_size = int(sc["rack_size"])
        if rack_size <= 0:
            raise ValueError("rack_size must be a positive integer")
        rack_weight = float(sc.get("rack_weight", 0.02))
        rack_p_destroy = float(sc.get("rack_p_destroy", 0.01))
        rack_outage = sc.get("rack_outage_s", [3600.0, 21600.0])
        if (not isinstance(rack_outage, (list, tuple)) or len(rack_outage) != 2
                or float(rack_outage[0]) > float(rack_outage[1])):
            raise ValueError("rack_outage_s must be [lo, hi] with lo <= hi")
        topo = RackTopology(rack_size, int(cl["node_count"]))
        topo.assign(runtimes)
        _rack_reg = DonorRegistry.for_run(0)
        # donor matcher anti-affinity: default ON for crossjob/peerdram waves
        # whenever rack_size is present (spec) — a deterministic matcher
        # change, no RNG involved.
        _rack_reg.rack_anti_affinity = True
        _rack_reg.racks_by_slot = {
            name: topo.racks_of(name) for name in topo.slices}
        rack_rng = random.Random(f"{seed}:rack")   # the dedicated rack stream
        for rt in runtimes:
            rt._rack_rng = rack_rng
        rack_coord = RackFailureCoordinator(
            backend=backend, logger=logger, registry=_rack_reg,
            strategies=strategies, runtimes=runtimes, topology=topo,
            rng=rack_rng, p_destroy=rack_p_destroy,
            outage_s=(float(rack_outage[0]), float(rack_outage[1])))
        logger.record(
            start=0.0, end=0.0, run_id=0, job_id=None, rank=None,
            node="__rack_topology__", category="Topology",
            operation="rack_topology",
            details={"rack_size": rack_size, "node_count": int(cl["node_count"]),
                     "n_racks": topo.n_racks,
                     "slices": {n: list(s) for n, s in topo.slices.items()
                                if s is not None}})
        # cluster-wide rack-event rate = rack_weight x SUM over job nodes of
        # their organic lambda — independent of rack_size (same event count,
        # bigger blast). inject_random=false / ideal => rate 0 => no process.
        organic_rate = per_node_rate * sum(
            int(spec["ranks"]) for _job_id, spec in jobs)
        rack_rate = rack_weight * organic_rate
        if rack_rate > 0:
            backend.process(rack_coord.poisson_process(rack_rate))
        rack_sched = normalize_rack_scheduled(sc)
        if rack_sched:
            backend.process(rack_coord.scheduled_process(rack_sched))

    if arm_cfg.get("slots"):
        if policy:
            # cost-model phases (#8.1 stagger, packed by predicted durations)
            CrossJobPeerStrategy.slot_by_job = {
                job_id: float(p["slot_s"]) for job_id, p in policy["jobs"].items()}
        else:
            # heuristic: even spread; the closed loop re-packs from measurements
            CrossJobPeerStrategy.slot_by_job = {
                job_id: round(i * period / len(jobs), 2)
                for i, (job_id, _s) in enumerate(jobs)}
    churn_ctrl = (ChurnController(backend, sc, runtimes, CrossJobPeerStrategy, logger)
                  if do_resolve else None)

    def record_lifecycle(rt, operation: str, start: float, end: float) -> None:
        if not dynamic_arrivals:
            return
        node_state = {
            "job_pending": "pending",
            "job_arrival": "training",
            "job_idle_checkpoint": "idle_checkpoint_donor",
        }[operation]
        logger.record(
            start=start, end=end, run_id=0, job_id=rt.config.job_id,
            rank=None, node=f"{rt.config.job_id}-lifecycle",
            category="Lifecycle", operation=operation,
            details={
                "class": rt.class_name,
                "arrival_source": rt.arrival_source,
                "arrival_s": round(rt.arrival_s, 9),
                "job_nodes": rt.ranks,
                "node_state": node_state,
            },
        )

    procs = []
    for rt in runtimes:
        phase = (rng_phase.uniform(0, period)
                 if arm_cfg.get("random_phases") else 0.0)

        rt.phase = phase
        def start(rt=rt, phase=phase):
            if rt.arrival_s > 0:               # Stage 3: delayed job arrival
                record_lifecycle(rt, "job_pending", 0.0, rt.arrival_s)
                yield backend.timeout(rt.arrival_s)
            rt.state = "active"
            rt.checkpoint_strategy.registry.set_available(rt.config.job_id, True)
            record_lifecycle(rt, "job_arrival", backend.now, backend.now)
            if rt.arrival_s > 0:
                if churn_ctrl is not None:
                    churn_ctrl.on_arrival(rt)
            if phase > 0:
                yield backend.timeout(phase)
            rt.start_time = backend.now
            yield from rt.run()
            # Training is complete, but the fixed job allocation remains alive as
            # checkpoint-only donor capacity. Dynamic arrivals always add a NEW
            # immutable-width job; they never resize an existing runtime.
            rt.state = "idle_checkpoint"
            record_lifecycle(
                rt, "job_idle_checkpoint", backend.now, backend.now
            )
            if churn_ctrl is not None:
                churn_ctrl.on_departure(rt)
        procs.append(backend.process(start()))

    if churn_ctrl is not None:
        churn_ctrl.resolve("t0")               # solve for the active-at-t0 set
    if arm_cfg.get("controller_epoch_s"):
        backend.process(controller_epochs(
            backend, logger, float(arm_cfg["controller_epoch_s"]), period, []))
    pre = None if ideal_mode else sc.get("preemption")
    if pre:
        victim_prefix = str(pre["class_prefix"])
        configured_victim_count = sum(
            int(spec["count"])
            for cname, spec in sc["classes"].items()
            if cname.startswith(victim_prefix)
        )
        start_preemption(
            backend, runtimes, pre=pre, seed=seed,
            configured_victim_count=configured_victim_count,
        )

    # censoring cap: with rollback semantics, arms without durable copies can
    # loop forever under preemption (evict -> restart from zero -> evict) —
    # the honest real-world outcome. Cap at max_sim_s (default 20x the 2 h
    # horizon) and mark censored completions.
    sim_cap = float(sc.get("simulation", {}).get("max_sim_s", 144000.0))
    _hb_stop = threading.Event()
    _hb_thread = None
    if progress_path is not None:
        _hb_thread = threading.Thread(
            target=_progress_heartbeat,
            args=(backend, runtimes, progress_path, _hb_stop),
            kwargs={
                "label": f"{arm}/s{seed}",
                "interval": max(1.0, float(progress_interval)),
            },
            daemon=True,
        )
        _hb_thread.start()
    try:
        # Controller and preemption monitors are intentionally long-lived, so
        # waiting for the queue to drain always runs to max_sim_s. Stop as soon
        # as every job process completes, with max_sim_s retained as a censoring
        # guard for policies that repeatedly roll back and never finish.
        run_until_complete_or_cap(env, backend, procs, sim_cap)
        censored = [rt.config.job_id for rt in runtimes if rt.end_time is None]
        pending_at_cap = [
            rt.config.job_id for rt in runtimes if rt.state == "pending"
        ]
        active_at_cap = [
            rt.config.job_id for rt in runtimes if rt.state == "active"
        ]
        for rt in runtimes:
            if rt.end_time is None:
                rt.end_time = env.now              # censored at the cap
                if rt.state == "active":
                    rt.state = "censored"
        # A persist process may schedule donor-drain or backstop work while it
        # finishes. Re-snapshot until no live background process remains; one
        # snapshot can miss children appended after the initial list is built.
        while True:
            pending = [
                process
                for rt in runtimes
                for process in rt.checkpoint_strategy.background_flushes
                if not process.triggered
            ]
            if not pending:
                break
            env.run(until=backend.all_of(pending))
    finally:
        _hb_stop.set()
        if _hb_thread is not None:
            _hb_thread.join(timeout=2.0)

    # L1.5: close the hosted-DRAM residency ledger (pieces still resident at the
    # horizon bill their GB-seconds up to now; nothing is released — run is over)
    for rt in runtimes:
        fin = getattr(rt.checkpoint_strategy, "finalize_dram_accounting", None)
        if fin is not None:
            fin()
    # resource bill (2026-07-28): close the checkpoint residency integrals at
    # the horizon (L1/local/replica copies + hosted ssd pieces), same contract
    # as the L1.5 dram ledger above. Pure accounting.
    for rt in runtimes:
        fin = getattr(rt.checkpoint_strategy, "finalize_bill_accounting", None)
        if fin is not None:
            fin()

    logger.finalize_trace()

    # ---- metrics (schema mirrors run_final73.analyze; counts are RANK-weighted) ----
    # Maintained as events are recorded so finalized batches need not be loaded
    # back into memory. L3 drain events remain excluded from the flush-path
    # denominator exactly as in the former full-history scan.
    paths = logger.checkpoint_paths
    aborted = sum(int(rt.checkpoint_strategy.stats.get("peer_flush_aborted", 0))
                  for rt in runtimes)
    total = sum(paths.values()) + aborted
    # durable = an OFF-NODE copy survives at least single-node loss: cross-job
    # peer stripe (peer*), the store (owner-push / drained backstop), and the
    # in-job DRAM replica (gemini_replica) chosen by priced-placement injob -- a
    # replica on a PEER node survives that node's loss, exactly like a peer copy
    # (it dies only on whole-job eviction, same as any in-memory tier). local's
    # own-SSD local_fallback is NOT node-loss durable; its store_backstop is.
    durable = sum(v for p, v in paths.items()
                  if p.startswith("peer")
                  or p in ("store", "store_backstop", "gemini_replica",
                           # injob_ssd (rack_failure_spec.md): the ring-neighbor
                           # SSD replica survives single OWNER-node loss exactly
                           # like a peer copy (it dies with the HOST node/rack —
                           # that exposure is the 2x2's point, priced separately)
                           "injob_ssd"))
    restores: Counter = Counter()
    lost: list[int] = []
    ftypes: Counter = Counter()
    sstats: Counter = Counter()          # strategy-level flush outcome counters
    lost_by_tier: dict[str, list] = {}
    reboot_local_restores = 0            # reboots served by own surviving local SSD
    for rt in runtimes:
        restores.update(rt.restores)
        lost.extend(rt.lost_iters)
        ftypes.update(rt.failure_types)
        sstats.update(rt.checkpoint_strategy.stats)
        reboot_local_restores += rt.reboot_local_restores
        for tier, vals in rt.lost_by_tier.items():
            lost_by_tier.setdefault(tier, []).extend(vals)
    lost.sort()
    # donor-drain L3 accounting (goal point 5c): mean % of donor NIC-seconds
    # spent draining pieces to L3. Denominator = physical donor NIC-seconds over
    # the run = (total nodes) x makespan; numerator = booked drain NIC-seconds.
    total_nodes = sum(getattr(w, "represents", 1)
                      for rt in runtimes for w in rt.workers.values())
    makespan = max((rt.end_time or 0.0) for rt in runtimes)
    drain_nic_s = sum(float(rt.checkpoint_strategy.stats.get("drain_nic_seconds", 0.0))
                      for rt in runtimes)
    drain_bytes = sum(float(rt.checkpoint_strategy.stats.get("drain_bytes_gb", 0.0))
                      for rt in runtimes)
    donor_drain_nic_pct = (round(100.0 * drain_nic_s / (total_nodes * makespan), 4)
                           if total_nodes and makespan else 0.0)
    def _mean(xs): return round(statistics.mean(xs), 2) if xs else None
    # L3 tiers = any recovery served by the store backstop (owner-push whole
    # copy, drained pieces, or a peer+L3 stitch); L2 = the durable peer tier.
    l3_tiers = ("store", "l3", "crossjob_peer_l3_stitch")
    l3_lost = [x for t in l3_tiers for x in lost_by_tier.get(t, [])]
    l2_lost = lost_by_tier.get("crossjob_peer", [])
    # "scratch/stale" recoveries (goal point 5d): scratch = total loss to
    # initial_state; stale = fell back to a NON-durable on-node copy (dram/ssd),
    # which under node/donor loss means an older iteration than the durable tiers.
    # reboot (2026-07-21): a reboot recovering from its OWN surviving local SSD is
    # a legitimate insured recovery at the class's durable cadence, NOT stale —
    # exclude it (reported separately as reboot_local_ssd_recoveries).
    scratch_stale = (restores.get("initial_state", 0) + restores.get("dram", 0)
                     + restores.get("ssd", 0) - reboot_local_restores)
    policy_by_class = {}
    if policy:
        for class_name in sc["classes"]:
            class_jobs = [
                job_policy
                for job_policy in policy["jobs"].values()
                if job_policy.get("class") == class_name
            ]
            policy_by_class[class_name] = {}
            for key in ("capture_every", "checkpoint_every", "store_every", "kpeers"):
                values = sorted({job_policy.get(key) for job_policy in class_jobs},
                                key=lambda value: (value is None, value))
                policy_by_class[class_name][key] = (
                    values[0] if len(values) == 1 else values
                )
    scheduled_planned = [
        dataclasses.asdict(event)
        for job_id, _spec in jobs
        for event in scheduled_by_job[job_id]
    ]
    scheduled_fired = [
        dataclasses.asdict(event)
        for runtime in runtimes
        for event in runtime.scheduled_failures_fired
    ]
    turnarounds = [
        (rt.end_time or 0.0) - rt.start_time
        for rt in runtimes
        if rt.start_time is not None
    ]
    turnaround_by_class = {
        cname: [
            (rt.end_time or 0.0) - rt.start_time
            for rt in runtimes
            if rt.class_name == cname and rt.start_time is not None
        ]
        for cname in sc["classes"]
    }
    end_by_class = {
        cname: [
            (rt.end_time or 0.0) - rt.phase
            for rt in runtimes
            if rt.class_name == cname
        ]
        for cname in sc["classes"]
    }
    arrival_mode = (
        str(sc["job_arrivals"].get("mode", "all_at_start"))
        if dynamic_arrivals else "legacy"
    )
    row = {"arm": (policy["name"] if policy else arm), "seed": seed,
            "arrival_mode": arrival_mode,
            "arrival_manifest": [{
                "job_id": arrival.job_id,
                "class": arrival.class_name,
                "arrival_s": round(arrival.arrival_s, 9),
                "source": arrival.source,
                "batch": arrival.batch,
                "nodes": int(sc["classes"][arrival.class_name]["ranks"]),
            } for arrival in arrival_manifest],
            "jobs_initial": sum(
                arrival.arrival_s == 0 for arrival in arrival_manifest
            ),
            "jobs_arrived": sum(rt.state != "pending" for rt in runtimes),
            "max_turnaround_s": (
                round(max(turnarounds), 1) if turnarounds else None
            ),
            "per_class_turnaround_s": {
                cname: (round(max(values), 1) if values else None)
                for cname, values in turnaround_by_class.items()
            },
            "pending_at_cap": pending_at_cap,
            "active_at_cap": active_at_cap,
            "policy_kind": ("gp" if policy else "ideal" if ideal_mode
                            else "baseline" if bl is not None else "configured"),
            "policy_name": (policy["name"] if policy else bl.name if bl else arm),
            "policy_flags": requested_arm_cfg,
            "policy_by_class": policy_by_class,
            "random_failures_injected": (
                inject_random_failures and not ideal_mode
            ),
            "scheduled_failures_planned": scheduled_planned,
            "scheduled_failures_fired": scheduled_fired,
            # audit fix #14: completion is measured from each job's OWN start
            # (random_phases arms start jobs up to a period late; that offset
            # is scheduling, not checkpoint cost). makespan_s keeps the old
            # cluster-clock number for reference.
            "train_end_s": round(max((rt.end_time or 0.0) - rt.phase
                                     for rt in runtimes), 1),
            "makespan_s": round(max(rt.end_time or 0.0 for rt in runtimes), 1),
            "censored_jobs": censored,
            "per_class_end_s": {
                cname: (round(max(values), 1) if values else None)
                for cname, values in end_by_class.items()},
            "rank_flushes": total,
            "aborted_flushes": aborted,
            "stale_flushes_skipped": sum(
                s.stats.get("stale_flush_skipped", 0)
                for rt in runtimes for s in rt.all_strategies) if False else
                sum(getattr(rt.checkpoint_strategy, "stats", {}).get(
                    "stale_flush_skipped", 0) for rt in runtimes),
            "durable_frac": round(durable / total, 4) if total else None,
            "failures": dict(ftypes),
            "restores": sum(restores.values()),
            "recovery_source_tiers": dict(restores),
            "total_loss": restores.get("initial_state", 0),
            "lost_iters_p50": lost[len(lost) // 2] if lost else None,
            "lost_iters_max": lost[-1] if lost else None,
            # donor-drain L3 backstop (goal 2026-07-19)
            "backstop": CrossJobPeerStrategy.backstop,
            "donor_drain_nic_pct": donor_drain_nic_pct,
            "drain_bytes_gb": round(drain_bytes, 1),
            "l3_drains": int(sstats.get("l3_drains", 0)),
            "l3_drain_aborted": int(sstats.get("l3_drain_aborted", 0)),
            "stitch_recoveries": int(restores.get("crossjob_peer_l3_stitch", 0)),
            "parity_recoveries": int(restores.get("crossjob_peer_parity", 0)),
            "parity_flushes": int(sstats.get("parity_flushes", 0)),
            # L1.5 peer-DRAM tier (l15_peer_dram_spec.md): the memory bill +
            # volatility exposure. hosted_dram_gb_h integrates piece_gb x
            # residency (land -> demote/loss/horizon) across all donors.
            "dram_waves": int(sstats.get("dram_peer_flushes", 0)),
            "dram_demotions": int(sstats.get("dram_demotions", 0)),
            "dram_demote_aborted": int(sstats.get("dram_demote_aborted", 0)),
            "dram_piece_losses": int(sstats.get("dram_pieces_lost", 0)),
            "dram_superseded": int(sstats.get("dram_superseded", 0)),
            "hosted_dram_gb_h": round(
                float(sstats.get("dram_gb_seconds", 0.0)) / 3600.0, 4),
            "dram_recoveries": int(
                restores.get("crossjob_peer_dram", 0)
                + restores.get("crossjob_peer_dram_stitch", 0)),
            "scratch_stale_recoveries": int(scratch_stale),
            # reboot-class physics: recoveries served by the rebooted node's OWN
            # surviving local SSD (the coverage niche `local` placement gains).
            "reboot_local_ssd_recoveries": int(reboot_local_restores),
            "l2_staleness_mean": _mean(l2_lost),
            "l3_staleness_mean": _mean(l3_lost),
            "staleness_by_tier": {t: _mean(v) for t, v in lost_by_tier.items()},
            # matching A/B outcome metrics (Mark D8): fallback = a wave that got
            # ZERO donors -> local-only (not node-loss durable); partial = a wave
            # matched to < k donors. Both rise under a worse matcher.
            "peer_flushes": int(sstats.get("peer_flushes", 0)),
            "fallback_flushes": int(sstats.get("fallback_flushes", 0)),
            "partial_grants": int(sstats.get("partial_grants", 0)),
            "matching": CrossJobPeerStrategy.matching,
            "spread_jobs": CrossJobPeerStrategy.spread_jobs,
            "window_bias": CrossJobPeerStrategy.window_bias,
            "placement": placement,           # Stage 4 L2 placement (auto/crossjob/injob)
            # priced placement (2026-07-21): the ENACTED per-class placement mix
            # (from the policy's per-job choice; empty for arm-global runs).
            "placement_by_class": ({
                pj["class"]: pj["placement"] for pj in policy["jobs"].values()
                if "placement" in pj} if policy else {}),
            # Stage 3: the dynamic re-solves (composition -> new per-class k/every)
            "resolve_count": len(churn_ctrl.actions) if churn_ctrl else 0,
            "resolve_actions": (churn_ctrl.actions if churn_ctrl else []),
            "refusals": dict(DonorRegistry.for_run(0).refusals),
            # resource bill (2026-07-28): checkpoint-resident capacity-time +
            # store write volume, priced downstream (bill_gammas.md). Pure
            # accounting — dram = L1 snapshots + in-job DRAM replicas + restored
            # copies (cluster-wide bytes, cohort-scaled) + L1.5 hosted pieces;
            # ssd = own local-SSD copies + hosted stripe/injob_ssd pieces (incl.
            # write-then-rename shadows); store = GB successfully written to L3
            # (store flushes + owner-push backstops + donor drains).
            "ckpt_dram_gbh": round(
                (float(sstats.get("ckpt_dram_gb_seconds", 0.0))
                 + float(sstats.get("dram_gb_seconds", 0.0))) / 3600.0, 4),
            "ckpt_ssd_gbh": round(
                (float(sstats.get("ckpt_ssd_gb_seconds", 0.0))
                 + float(sstats.get("ckpt_hosted_ssd_gb_seconds", 0.0)))
                / 3600.0, 4),
            "store_gb_written": round(
                float(sstats.get("store_gb_written", 0.0)), 2)}
    # jit_pc_fallback documentation (fairness fix 2026-07-28): the c* arithmetic
    # per engaged class — N, f (the scenario's accelerated per-node rate), o
    # (predicted GPU->DRAM capture stall), c*, interval, enacted checkpoint_every.
    # Key appears ONLY when at least one class engaged, so jit runs where every
    # class holds a DP replica (e.g. the zero_stage-0 grid) keep the archived
    # row schema byte-identical.
    if jit_pc_by_class:
        row["jit_pc_fallback"] = {
            "formula": "c* = sqrt(N*f/(2*o))  [JIT EuroSys'24 Sec.5.2 eq.3]",
            "per_class": jit_pc_by_class,
        }
    # rack failure domains + injob_ssd (rack_failure_spec.md): keys appear ONLY
    # when the feature/placement is live, so every pre-rack scenario's results
    # JSON stays byte-identical (mod wall_s) — the hard regression gate.
    if sstats.get("injob_ssd_flushes"):
        row["injob_ssd_flushes"] = int(sstats["injob_ssd_flushes"])
    if rack_coord is not None:
        row.update(rack_coord.summary())
        # anti-affinity fallback counter: surfaced even at 0 — never silent.
        row["rack_affinity_fallbacks"] = int(
            DonorRegistry.for_run(0).rack_fallbacks)
    # fabric-aware coupling (CL-014): provenance for a run whose alpha mechanism
    # is fabric-scoped, plus the fabric each class's collective was declared on.
    # Key appears ONLY when the flag is on => every flag-off row byte-identical.
    if CohortJobRuntime.fabric_aware_coupling:
        row["fabric_aware_coupling"] = True
    # intensity coupling (CL-016): same provenance rule — the key appears ONLY
    # when the flag is on, so every flag-off row stays byte-identical.
    if CohortJobRuntime.intensity_coupling:
        row["intensity_coupling"] = True
    # non-blocking capture (CL-017): same provenance rule — the key appears
    # ONLY when the flag is on, so every flag-off row stays byte-identical.
    if CrossJobPeerStrategy.nonblocking_capture:
        row["nonblocking_capture"] = True
    if CohortJobRuntime.fabric_aware_coupling or CohortJobRuntime.intensity_coupling:
        row["train_fabric_by_class"] = {
            cname: train_fabric_of(cspec.get("rates"))
            for cname, cspec in sc["classes"].items()}
    # failure-trace replay (CL-018): key appears ONLY when failures.trace_file
    # is present, so every trace-less row stays byte-identical — the same
    # provenance rule as CL-014/CL-016/CL-017. Jobs the window monitor did not
    # reach (every job finished, or max_sim_s < W) are snapshotted post-run
    # with the same §4 semantics.
    if trace_active:
        for rt in runtimes:
            if rt.config.job_id not in trace_snapshot:
                trace_snapshot[rt.config.job_id] = rt.trace_snapshot_at_window(
                    failure_trace.window_s)
        row["trace_replay"] = {
            "trace_file": failure_trace.path,
            "window_s": failure_trace.window_s,
            "constants": {
                "detect_s": failure_trace.detect_s,
                "reprovision_delay_s": failure_trace.reprovision_delay_s,
                "restore_s": failure_trace.restore_s,
            },
            "events_total": failure_trace.event_count,
            "per_job": {job_id: trace_snapshot[job_id]
                        for job_id, _spec in jobs},
        }
    return row


def _execute_run_task(task):
    (
        sc,
        arm,
        seed,
        trace,
        policy,
        progress_path,
        backstop,
        progress_interval,
        event_batch_size,
    ) = task
    wall_start = time.monotonic()
    row = run_arm(
        sc,
        arm,
        seed,
        trace,
        policy=policy,
        progress_path=progress_path,
        backstop=backstop,
        progress_interval=progress_interval,
        event_batch_size=event_batch_size,
    )
    # wall_s: end-to-end wall-clock for THIS arm/seed run (Sim-Wall column,
    # Table-15 parity). Per-run so it is meaningful under --workers>1.
    row["wall_s"] = round(time.monotonic() - wall_start, 3)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--arm", help="single arm (default: every arm in the YAML)")
    ap.add_argument("--seed", type=int, help="single seed (default: YAML seed list)")
    ap.add_argument("--out", type=Path, help="results JSON path")
    ap.add_argument("--trace-dir", type=Path,
                    help="write per-run gzipped event traces here")
    ap.add_argument("--policy", type=Path,
                    help="gp_policy.py JSON: cost-model frequencies/k/slots")
    ap.add_argument("--backstop",
                    choices=("donor_drain", "owner_push", "parity",
                             "parity_cutoff"),
                    help="override the durability backstop mechanism for every "
                         "arm (donor-drain L3 | legacy owner-push | XOR parity "
                         "stretch | parity+straggler-cutoff); lets one solved "
                         "policy be run every way")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "independent arm/seed processes to run concurrently (default 1; "
            "a single arm/seed remains one ordered event simulation)"
        ),
    )
    ap.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="wall-clock seconds between progress/ETA updates (default 10)",
    )
    ap.add_argument(
        "--event-batch-size",
        type=int,
        default=50_000,
        help="events retained before a sorted trace chunk is spooled (default 50000)",
    )
    ap.add_argument(
        "--background-nic-gbps",
        type=float,
        default=None,
        help="Route C: per-node persistent background front-end (ingress) traffic "
             "in GB/s (data-loading + other-tenant storage I/O). Overrides "
             "cluster.background_nic_gbps for every arm/seed in this run; omit to "
             "use the scenario value (default 0.0 = off).",
    )
    args = ap.parse_args()

    sc = yaml.safe_load(args.scenario.read_text())
    if args.background_nic_gbps is not None:
        sc.setdefault("cluster", {})["background_nic_gbps"] = args.background_nic_gbps
    policy = json.loads(args.policy.read_text()) if args.policy else None
    arms = ([policy["name"]] if policy
            else [args.arm] if args.arm else list(sc["arms"]))
    seeds = [args.seed] if args.seed is not None else list(sc["simulation"]["seeds"])
    out = args.out or RESULTS_DIR / f"scenario_{args.scenario.stem}.json"
    progress_path = Path(str(out) + ".progress")   # wall-clock heartbeat, next to --out
    if args.workers <= 0:
        ap.error("--workers must be positive")
    if args.progress_interval <= 0:
        ap.error("--progress-interval must be positive")
    if args.event_batch_size <= 0:
        ap.error("--event-batch-size must be positive")
    tasks = []
    for arm in arms:
        for seed in seeds:
            trace = (args.trace_dir / f"{args.scenario.stem}_{arm}_s{seed}.jsonl.gz"
                     if args.trace_dir else None)
            task_progress = (
                progress_path
                if len(arms) * len(seeds) == 1
                else Path(f"{out}.{arm}.s{seed}.progress")
            )
            tasks.append(
                (
                    sc,
                    arm,
                    seed,
                    trace,
                    policy,
                    task_progress,
                    args.backstop,
                    args.progress_interval,
                    args.event_batch_size,
                )
            )

    rows = []
    worker_count = min(args.workers, len(tasks))
    if worker_count == 1:
        results = map(_execute_run_task, tasks)
        pool = None
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count)
        results = pool.map(_execute_run_task, tasks)
    try:
        for row in results:
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    print("\n=== AGGREGATE ===")
    agg = {}
    for arm in arms:
        r = [x for x in rows if x["arm"] == arm]
        ends = [x["train_end_s"] for x in r]
        agg[arm] = {
            "completion_mean_s": round(statistics.mean(ends), 1),
            "completion_range_s": [min(ends), max(ends)],
            "durable_frac": round(statistics.mean(
                x["durable_frac"] or 0.0 for x in r), 4),
            "total_loss_mean": round(statistics.mean(
                x["total_loss"] for x in r), 1),
            "lost_iters_max": max(x["lost_iters_max"] or 0 for x in r)}
        print(f'{arm}: completion mean {agg[arm]["completion_mean_s"]}s '
              f'{agg[arm]["completion_range_s"]} | durable '
              f'{agg[arm]["durable_frac"]:.1%} | loss {agg[arm]["total_loss_mean"]} '
              f'| worst lost iters {agg[arm]["lost_iters_max"]}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scenario": str(args.scenario), "rows": rows,
                               "aggregate": agg}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
