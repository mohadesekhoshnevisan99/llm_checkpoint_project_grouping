"""Non-blocking checkpoint capture (protocol CL-017).

exp1 hardware refuted the capture-as-GPU-lock model: the 2 GB GPU->DRAM copy
takes 413 ms against a 485 ms iteration, so a blocking capture would spike
every capture iteration -- measured p99 stretch is 1.002. "The capture does
not stall training at all" (meetings_summary/2026-07-31_exp1_results.md
section 5, point 3; PAPER_NOTES.md 2026-08-01). The simulator charged a
+0.2937 pp capture floor for exactly this lock (CL-012.A(d),
checkpointing/tiered.py). `model.nonblocking_capture: true` removes the lock
and NOTHING else: the capture duration is unchanged and still gates the
checkpoint pipeline (the DRAM copy is placed and the stripe starts only when
the D2H timeout completes). Zero free parameters -- the flag deletes a lock.

PENCIL (scenarios/handcheck_fabric.yaml, constants derived in its header;
capture = 1.000 s, ar_base = 1.000 s, compute = 1.000 s, stripe = 100.000 s;
wave fires at it=3, t=6.000):

  BLOCKING (committed arms -- must not move):
    blind  ends 15.000   capture 8.000..9.000 (queued behind iteration 4's
    scoped ends 13.000   GPU re-request), stripe 9.000..109.000

  NON-BLOCKING (injected arms):
    * capture runs 6.000..7.000, overlapping iteration 4's compute -- it
      starts at the wave boundary because it no longer waits for the GPU;
    * the stripe starts at exactly 7.000 = capture end (still gated), and
      still takes 100.000 s (no rate change anywhere);
    * scoped+nonblocking ends 12.000 = 13.000 - 1.000: the capture stall is
      removed EXACTLY, nothing else moves (no coupling on a separate fabric);
    * blind+nonblocking ends 15.000 = 15.000 - 1.000 (stall removed) + 1.000
      (iteration 4's all-reduce, 7.000..8.000, now overlaps the earlier
      stripe and takes ar_base x (1 + tasks) = 2.000 s). The coupling charge
      is untouched by this flag -- the two effects cancel on these round
      constants, which pins both directions at once.

Unit level (micro-harness, real simpy backend): with the flag ON a competing
GPU request during the capture window is granted immediately; with it OFF it
waits until capture end; in BOTH cases the persist (checkpoint) / DRAM-copy
placement (snapshot) happens at exactly capture end.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import simpy
import yaml

from checkpointing.crossjob import CrossJobPeerStrategy
from checkpointing.tiered import TieredCheckpointStrategy
from run_scenario import run_arm
from simulation.backend import SimPyBackend

SCENARIO = (Path(__file__).resolve().parent.parent
            / "scenarios" / "handcheck_fabric.yaml")

BLIND_END = 15.0      # committed digit (tests/test_fabric_coupling.py)
SCOPED_END = 13.0     # committed digit
CAPTURE_S = 1.0       # checkpoint_gb_per_rank / gpu_cpu_bandwidth_gbps
NB_SCOPED_END = SCOPED_END - CAPTURE_S            # 12.000
NB_BLIND_END = BLIND_END - CAPTURE_S + 1.0        # 15.000, see header

NB = {"kpeers": True, "slots": False, "nonblocking_capture": True}
NB_SCOPED = dict(NB, fabric_aware_coupling=True)


def _load() -> dict:
    return yaml.safe_load(SCENARIO.read_text())


def _run(arm: str, inject: dict | None = None, trace: Path | None = None,
         sc: dict | None = None) -> dict:
    sc = sc or _load()
    if inject is not None:
        sc["arms"][arm] = inject
    return run_arm(sc, arm, 7, trace)


def _end(row: dict) -> float:
    return row["per_class_end_s"]["trainer"]


# ------------------- 1. the regression contract: flag OFF --------------------

def test_flag_absent_reproduces_both_committed_digits() -> None:
    assert _end(_run("blind")) == pytest.approx(BLIND_END, abs=1e-9)
    assert _end(_run("scoped")) == pytest.approx(SCOPED_END, abs=1e-9)


def test_flag_absent_adds_no_keys_to_the_results_row() -> None:
    """Committed results are compared byte-for-byte mod wall_s; a new key in a
    flag-off row would break every one of them."""
    assert "nonblocking_capture" not in _run("blind")
    assert "nonblocking_capture" not in _run("scoped")


def test_flag_does_not_leak_between_arms() -> None:
    """A class attribute set per run: a nonblocking arm then a plain one."""
    assert _end(_run("nb", NB_SCOPED)) == pytest.approx(NB_SCOPED_END, abs=1e-9)
    assert _end(_run("blind")) == pytest.approx(BLIND_END, abs=1e-9)
    assert CrossJobPeerStrategy.nonblocking_capture is False
    assert TieredCheckpointStrategy.nonblocking_capture is False


# ------------------- 2. flag ON: the capture stall is gone -------------------

def test_capture_no_longer_stretches_iterations_removed_exactly() -> None:
    """Separate fabric => zero coupling, so the ONLY difference to the scoped
    pencil is the capture stall: 13.000 - 1.000 = 12.000, to the digit."""
    row = _run("nb", NB_SCOPED)
    assert _end(row) == pytest.approx(NB_SCOPED_END, abs=1e-9)
    assert row["nonblocking_capture"] is True


def test_coupling_charge_is_untouched_by_this_flag() -> None:
    """blind+nonblocking: -1.000 s stall +1.000 s newly-overlapped iteration-4
    all-reduce = 15.000 again (derivation in the header). The flag removes the
    GPU lock only; the network coupling model is not this change's business."""
    assert _end(_run("nb", dict(NB))) == pytest.approx(NB_BLIND_END, abs=1e-9)


def test_scenario_model_block_enables_it_too() -> None:
    sc = _load()
    sc["model"] = {"fabric_aware_coupling": True, "nonblocking_capture": True}
    assert _end(_run("blind", sc=sc)) == pytest.approx(NB_SCOPED_END, abs=1e-9)


# --------- 3. the pipeline is still gated by the capture DURATION ------------

def _trace_windows(arm: str, inject: dict | None, tmp_path: Path):
    tp = tmp_path / f"{arm}.jsonl"
    _run(arm, inject, trace=tp)
    ev = [json.loads(line) for line in tp.read_text().splitlines()]
    caps = sorted({(e["start"], e["end"]) for e in ev
                   if e["operation"] == "checkpoint_stage_gpu_to_dram"})
    stripes = sorted({(e["start"], e["end"]) for e in ev
                      if "crossjob_peers" in e["operation"]})
    return caps, stripes


def test_stripe_still_starts_exactly_at_capture_end(tmp_path: Path) -> None:
    # blocking: capture queues behind iteration 4 (8..9), stripe 9..109
    caps, stripes = _trace_windows("scoped", None, tmp_path)
    assert caps == [(8.0, 9.0)]
    assert stripes == [(9.0, 109.0)]
    # nonblocking: capture starts at the wave boundary (no GPU wait), same
    # 1.000 s duration, and the stripe starts at exactly capture end -- two
    # seconds earlier, one second of which is the removed stall and one the
    # removed queueing behind iteration 4. Stripe duration unchanged: 100 s.
    caps, stripes = _trace_windows("nb", NB_SCOPED, tmp_path)
    assert caps == [(6.0, 7.0)]
    assert stripes == [(7.0, 107.0)]


# ------------------- 4. unit level: the lock itself --------------------------

def _capture_rig(nonblocking: bool):
    """Real simpy backend, mock strategy: one worker, capture = 1.000 s."""
    env = simpy.Environment()
    backend = SimPyBackend(env)
    worker = SimpleNamespace(
        job_id="j", rank=0, data_parallel_rank=0, pipeline_stage=0,
        physical_node="n0", name="j-rank-0",
        gpu=backend.resource(1), cpu=backend.priority_resource(1),
        failed=False, failure_generation=0, checkpoint_epoch=0,
        current_iteration=0, active_checkpoint_gb=0.0, represents=1)
    log: dict[str, list] = {"persist": [], "put": [], "gpu_acquired": []}

    def _checkpoint_timeout(workers, *, generations, duration):
        yield backend.timeout(duration)
        return True

    def _persist_checkpoint(w, job, **kw):
        log["persist"].append(env.now)
        return True
        yield  # pragma: no cover -- makes this a generator

    strat = SimpleNamespace(
        backend=backend,
        cluster=SimpleNamespace(gpu_cpu_bandwidth_gbps=1.0),
        checkpoint_locks={0: backend.resource(1)},
        nonblocking_capture=nonblocking,
        strategy_name="rig",
        stats={"snapshots": 0},
        _wait_until_healthy=lambda w: iter(()),
        _checkpoint_timeout=_checkpoint_timeout,
        _record=lambda *a, **k: None,
        _checkpoint_details=lambda *a, **k: {},
        _put_copy=lambda **kw: log["put"].append(env.now),
        _persist_checkpoint=_persist_checkpoint,
    )
    job = SimpleNamespace(job_id="j", checkpoint_gb=1.0, pipeline_stages=1)

    def probe():  # a training iteration re-requesting the GPU mid-capture
        yield backend.timeout(0.25)
        req = worker.gpu.request()
        yield req
        log["gpu_acquired"].append(env.now)
        worker.gpu.release(req)

    return env, backend, strat, worker, job, probe, log


@pytest.mark.parametrize("nonblocking,acquired_at", [(False, 1.0), (True, 0.25)])
def test_checkpoint_gpu_lock_only_under_the_old_model(nonblocking,
                                                      acquired_at) -> None:
    env, backend, strat, worker, job, probe, log = _capture_rig(nonblocking)
    backend.process(
        TieredCheckpointStrategy.checkpoint(strat, worker, job, iteration=1))
    backend.process(probe())
    env.run()
    assert log["gpu_acquired"] == [acquired_at]
    # both ways: the persist may not begin before the capture completes
    assert log["persist"] == [1.0]
    assert log["put"] == [1.0]


@pytest.mark.parametrize("nonblocking,acquired_at", [(False, 1.0), (True, 0.25)])
def test_snapshot_gpu_lock_only_under_the_old_model(nonblocking,
                                                    acquired_at) -> None:
    env, backend, strat, worker, job, probe, log = _capture_rig(nonblocking)
    backend.process(
        CrossJobPeerStrategy.snapshot(strat, worker, job, iteration=1))
    backend.process(probe())
    env.run()
    assert log["gpu_acquired"] == [acquired_at]
    # the DRAM copy exists only at capture end -- freshness is not free
    assert log["put"] == [1.0]
    assert strat.stats["snapshots"] == 1
