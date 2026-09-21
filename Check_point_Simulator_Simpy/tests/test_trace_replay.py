"""Deterministic failure-trace replay (EXP2_SPEC.md, protocol CL-018).

The scenario key `failures.trace_file` replays a hardware failure trace
verbatim: at each event's t_s the targeted node's JOB fails (blast radius =
the job, EXP2_SPEC §2), recovery is detect_s + reprovision_delay_s + restore_s
(measured frozen constants, §3/§7), the restore target is the latest complete
checkpoint committed strictly before t_kill (§3.3), equal-t_s events on one
job coalesce to ONE restart (§5), and a failure during recovery RESETS the
recovery clock (§3). Goodput accounting (§4) is snapshotted at t = W.

These tests pin:
  1. FLAG ABSENT => byte-identity: no trace key, no `trace_replay` row key,
     and a trace whose only event lies beyond the horizon changes NOTHING in
     the row (the committed-scenario byte-identity gate ran separately:
     handcheck.yaml ours_full s7 + handcheck_reboot.yaml ours_local s7,
     results JSON mod wall_s AND full trace bytes identical).
  2. A 2-event synthetic trace produces EXACTLY the hand-computed goodput
     (pencil arithmetic below).
  3. The 7:28-style simultaneous burst (two nodes of one job at one t_s) is
     ONE restart, not two.
  4. A failure landing mid-recovery RESETS recovery (one rule, §3).
  5. Frozen constants are required (no defaults) and may come from a YAML
     file (exp2_constants.yaml) instead of an inline mapping.

PENCIL SCENARIO (exp2 job shape, all tests share it):
  job A ("A0"): 2-way DDP (2 cohorts, m=1), iter compute 10 s,
    gradient_gb_per_rank 0 => ar_base = 0 => iteration wall EXACTLY 10 s
    (plus ~1 ms capture stall per wave: 1.0 GB / 1000 GB/s gpu->dram; the
    cumulative drift over the whole window is < 20 ms, far below every margin
    used here). checkpoint_every 3, kpeers 1 => at it = 3,6,9,... each cohort
    stripes its 1.0 GB shard to job B's node; flush = 1.0 GB at
    min(nic 1.0, disk 1.0) = 1.0 GB/s => the wave landed well under 4 s after
    its boundary — every kill below sits >= 13 s after a landing, so the
    restore target is never ambiguous.
  job B ("B0"): 1 node, never checkpoints (checkpoint_every 100000), donates
    its SSD — the exp2 donor. Never targeted.
  constants: detect_s 1.0, reprovision_delay_s 30.0, restore_s 9.0
    => one uninterrupted recovery = EXACTLY 40.0 s.
  window W = 596.0 s (chosen so no iteration boundary of either job ties
    with the horizon).
"""
from __future__ import annotations

import pytest
import yaml

from run_scenario import normalize_failure_trace, run_arm

CONSTANTS = {"detect_s": 1.0, "reprovision_delay_s": 30.0, "restore_s": 9.0}
WINDOW_S = 596.0


def _scenario(trace_path=None, constants=CONSTANTS) -> dict:
    sc = {
        "simulation": {"seeds": [7], "max_sim_s": 650.0, "engine": "event"},
        "cluster": {
            "node_count": 100,
            "gpu_cpu_bandwidth_gbps": 1000.0,
            "network_bandwidth_gbps": 1.0,
            "local_ssd_bandwidth_gbps": 1.0,
            "object_store_bandwidth_gbps": 8.0,
        },
        "classes": {
            "A": {                       # exp2 job A: 2-way DDP, stripes to B
                "count": 1, "ranks": 2, "cohorts": 2,
                "checkpoint_gb_per_rank": 1.0,
                "iteration_seconds": 10.0,
                "gradient_gb_per_rank": 0.0,
                "iterations": 1000,      # never completes inside the window
                "checkpoint_every": 3,
                "kpeers": 1, "rpo_s": 120.0,
            },
            "B": {                       # exp2 job B: donor, never checkpoints
                "count": 1, "ranks": 1, "cohorts": 1,
                "checkpoint_gb_per_rank": 1.0,
                "iteration_seconds": 10.0,
                "gradient_gb_per_rank": 0.0,
                "iterations": 1000,
                "checkpoint_every": 100000,
                "kpeers": 0, "rpo_s": 120.0,
            },
        },
        "failures": {
            "per_node_per_second": 0.0,  # the trace is the ONLY failure source
            "weights": {"process": 0.4, "node": 0.2, "spot": 0.4},
            "restart_seconds": {"process": 5.0, "node": 20.0, "spot": 30.0},
        },
        "store": {"in_gbps": 500.0, "out_gbps": 500.0, "disk_gbps": 500.0,
                  "stream_gbps": 1.0},
        "controller": {"slot_period_s": 30.0},
        "arms": {"ours": {"kpeers": True}},
    }
    if trace_path is not None:
        sc["failures"]["trace_file"] = str(trace_path)
        sc["failures"]["trace_node_map"] = {"node30": "A0", "node31": "A0"}
        sc["failures"]["trace_constants"] = (
            dict(constants) if isinstance(constants, dict) else constants)
    return sc


def _write_trace(tmp_path, name: str, events: list[dict]):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(
        {"provenance": {"window_s": WINDOW_S}, "events": events}))
    return path


def test_flag_absent_is_byte_identical_and_grows_no_key(tmp_path) -> None:
    """Absence of failures.trace_file => the Poisson-only code path, and a
    present trace whose only event never fires (t_s beyond max_sim_s) leaves
    every existing row field IDENTICAL — the trace machinery adds exactly the
    one `trace_replay` key and nothing else."""
    plain = run_arm(_scenario(), "ours", 7, None)
    assert "trace_replay" not in plain

    trace = _write_trace(tmp_path, "never.yaml", [
        {"t_s": 10000.0, "type": "node_loss", "target": "node30"},
    ])
    traced = run_arm(_scenario(trace), "ours", 7, None)
    assert "trace_replay" in traced
    replay = traced.pop("trace_replay")
    for row in (plain, traced):
        row.pop("wall_s", None)
        # the two runs legitimately differ in the echoed scenario config
        row.pop("policy_flags", None)
    plain["arrival_manifest"] = traced["arrival_manifest"]
    assert traced == plain
    assert replay["per_job"]["A0"]["trace_events"] == 0
    assert replay["per_job"]["A0"]["restarts"] == 0


def test_two_event_trace_matches_pencil_goodput(tmp_path) -> None:
    """PENCIL ARITHMETIC (every number derived before running):

    Event 1: t_s = 175.0, node30 -> job A0.
      A's boundaries sit at ~10*i, so the index standing at 175.0 is it = 17.
      Waves landed: @3,6,9,12,15 (wave @15 fires ~t=150, lands ~t=152 << 175)
      => latest complete checkpoint strictly before t_kill = @15.
      Chain: end = 175 + detect 1 + reprovision 30 + restore 9 = 215.0.
      lost = 17 - 15 = 2.       recovery = 215 - 175 = 40.0 s.
      (The sim's iteration 18 completes at ~180 before the boundary handler
      runs; it is rolled back and never counted — on hardware it never ran.)
      Resume at t = 215.0 with it = 15.

    Event 2: t_s = 351.0, node31 -> job A0 (other node, SAME job => same
      blast radius, §2).
      Post-resume boundaries: it = 15+n at ~215 + 10n => at 351.0, it = 28.
      Waves since resume: @18 (~245), @21, @24, @27 (~335, lands ~337 << 351)
      => restore target @27.  lost = 28 - 27 = 1.
      Chain: end = 351 + 40 = 391.0.   recovery = 391 - 351 = 40.0 s.
      Resume at t = 391.0 with it = 27.

    Horizon W = 596.0: A0 is TRAINING (no event after 351), so useful_iters
      is the live counter: it = 27 + floor((596 - 391)/10) = 27 + 20 = 47.
      goodput = 47 / 596.        lost_work_iters = 2 + 1 = 3.
      recovery_s = 40 + 40 = 80.0.     restarts = 2, resets = 0.

    Job B (never targeted, trains the whole window): useful = floor(596/10)
      = 59, goodput = 59/596, zero lost, zero recovery."""
    trace = _write_trace(tmp_path, "two_events.yaml", [
        {"t_s": 175.0, "type": "node_loss", "target": "node30"},
        {"t_s": 351.0, "type": "node_loss", "target": "node31"},
    ])
    row = run_arm(_scenario(trace), "ours", 7, None)

    replay = row["trace_replay"]
    assert replay["window_s"] == WINDOW_S
    assert replay["events_total"] == 2
    assert replay["constants"] == CONSTANTS

    a = replay["per_job"]["A0"]
    assert a["useful_iters"] == 47
    assert a["lost_work_iters"] == 3
    assert a["recovery_s"] == 80.0
    assert a["goodput_iters_per_s"] == round(47 / 596.0, 9)
    assert a["restarts"] == 2
    assert a["resets"] == 0
    assert a["trace_events"] == 2
    assert not a["recovering_at_window"]

    b = replay["per_job"]["B0"]
    assert b["useful_iters"] == 59
    assert b["lost_work_iters"] == 0
    assert b["recovery_s"] == 0.0
    assert b["goodput_iters_per_s"] == round(59 / 596.0, 9)
    assert b["restarts"] == 0

    # both restores came from the cross-job peer stripe (2 cohorts x 2 events)
    assert row["failures"] == {"trace_node_loss": 2}
    assert row["recovery_source_tiers"] == {"crossjob_peer": 4}
    assert row["total_loss"] == 0
    assert row["lost_iters_max"] == 2


def test_simultaneous_burst_is_one_restart_not_two(tmp_path) -> None:
    """EXP2_SPEC §5 (the 448.68 s / 7:28 burst): two events at ONE t_s
    targeting BOTH nodes of job A coalesce into ONE job failure — one
    detection, one reprovision, one restore. PENCIL: kill at 175.0 (it = 17,
    restore @15, lost 2), ONE chain ending 175 + 40 = 215.0, recovery_s =
    40.0 exactly (not 80). useful = 15 + floor((596 - 215)/10) = 53."""
    trace = _write_trace(tmp_path, "burst.yaml", [
        {"t_s": 175.0, "type": "node_loss", "target": "node30"},
        {"t_s": 175.0, "type": "node_loss", "target": "node31"},
    ])
    constants_file = tmp_path / "exp2_constants.yaml"   # file-based constants
    constants_file.write_text(yaml.safe_dump(CONSTANTS))
    row = run_arm(_scenario(trace, constants=str(constants_file)), "ours", 7, None)

    a = row["trace_replay"]["per_job"]["A0"]
    assert a["trace_events"] == 2          # two raw events delivered ...
    assert a["restarts"] == 1              # ... ONE restart, not two
    assert a["resets"] == 0
    assert a["recovery_s"] == 40.0         # one 40 s chain, not 80
    assert a["lost_work_iters"] == 2
    assert a["useful_iters"] == 53
    assert row["failures"] == {"trace_node_loss": 1}


def test_failure_during_recovery_resets_recovery(tmp_path) -> None:
    """EXP2_SPEC §3: an event at 185.0 lands 10 s into the 175.0 chain's
    recovery => RESET: the clock restarts at 185 + reprovision 30 + restore 9
    = 224.0 (one rule — no second detect, no queued second recovery).
    PENCIL: one chain, one restart, one reset; recovery_s = 224 - 175 = 49.0;
    lost counted ONCE = 17 - 15 = 2; useful = 15 + floor((596-224)/10) = 52."""
    trace = _write_trace(tmp_path, "reset.yaml", [
        {"t_s": 175.0, "type": "node_loss", "target": "node30"},
        {"t_s": 185.0, "type": "node_loss", "target": "node31"},
    ])
    row = run_arm(_scenario(trace), "ours", 7, None)

    a = row["trace_replay"]["per_job"]["A0"]
    assert a["restarts"] == 1
    assert a["resets"] == 1
    assert a["recovery_s"] == 49.0
    assert a["lost_work_iters"] == 2
    assert a["useful_iters"] == 52
    assert row["failures"] == {"trace_node_loss": 1}


def test_constants_are_required_frozen_inputs(tmp_path) -> None:
    """The recovery constants are measured pre-run and frozen (EXP2_SPEC §6):
    a missing key is a hard error, never a default."""
    trace = _write_trace(tmp_path, "t.yaml", [
        {"t_s": 10.0, "type": "node_loss", "target": "node30"},
    ])
    sc = _scenario(trace)
    del sc["failures"]["trace_constants"]["restore_s"]
    jobs = [("A0", sc["classes"]["A"]), ("B0", sc["classes"]["B"])]
    with pytest.raises(ValueError, match="restore_s"):
        normalize_failure_trace(sc, jobs)

    sc2 = _scenario(trace)
    sc2["failures"]["trace_node_map"] = {"node30": "nosuchjob"}
    with pytest.raises(ValueError, match="unknown job"):
        normalize_failure_trace(sc2, jobs)


def test_real_exp2_trace_parses_and_coalesces_the_burst() -> None:
    """The committed seed-0 exp2 trace (13 events, one 2-node burst at
    448.68 s) parses verbatim: 13 raw events -> 12 job-level failures for the
    mapped job, window from the file's own provenance (3600 s)."""
    from pathlib import Path
    trace_path = (Path(__file__).resolve().parents[2] / "checkpointing"
                  / "realgpu_coefficient" / "traces"
                  / "failure_trace_N2_T60min_seed0.yaml")
    if not trace_path.exists():
        pytest.skip("checkpointing submodule not checked out next to the sim")
    sc = _scenario(trace_path)
    jobs = [("A0", sc["classes"]["A"]), ("B0", sc["classes"]["B"])]
    trace = normalize_failure_trace(sc, jobs)
    assert trace.window_s == 3600.0
    assert trace.event_count == 13
    events = trace.events_by_job["A0"]
    assert len(events) == 12                       # burst coalesced
    burst = [e for e in events if len(e.targets) == 2]
    assert len(burst) == 1
    assert burst[0].t_s == 448.68
    assert set(burst[0].targets) == {"node30", "node31"}


def test_per_class_restore_selection(tmp_path):
    """CL-019 pencil test: two events of different classes pick different
    restore durations. detect=1, reprov=30; process_kill restores in 5 s,
    node_loss in 50 s. kill@175 -> recovery ends 175+1+30+5 = 211;
    kill@351 -> ends 351+1+30+50 = 432. With 10 s iteration walls, boundaries
    resume at 220 and 440 respectively (next wall after recovery end)."""
    trace = tmp_path / "mix.yaml"
    trace.write_text(yaml.safe_dump({
        "provenance": {"window_s": WINDOW_S},
        "events": [
            {"t_s": 175.0, "type": "process_kill", "target": "node30"},
            {"t_s": 351.0, "type": "node_loss", "target": "node31"},
        ]}))
    sc = _scenario(trace, constants={
        "detect_s": 1.0, "reprovision_delay_s": 30.0, "restore_s": 9.0,
        "restore_by_class": {"process_kill": 5.0, "node_loss": 50.0}})
    row = run_arm(sc, "ours", 7, None)
    tr = row["trace_replay"]["per_job"]["A0"]
    assert row["failures"] == {"trace_process_kill": 1, "trace_node_loss": 1}
    # recovery_s = (211-175) + (432-351) = 36 + 81 = 117
    assert abs(tr["recovery_s"] - 117.0) < 1e-6, tr["recovery_s"]


def test_burst_takes_severest_class(tmp_path):
    """A coalesced burst restarts ONCE and pays the deepest tier: process_kill
    + node_loss at the same t_s -> one restart, restore = max(5, 50) = 50."""
    trace = tmp_path / "burst.yaml"
    trace.write_text(yaml.safe_dump({
        "provenance": {"window_s": WINDOW_S},
        "events": [
            {"t_s": 175.0, "type": "process_kill", "target": "node30"},
            {"t_s": 175.0, "type": "node_loss", "target": "node31"},
        ]}))
    sc = _scenario(trace, constants={
        "detect_s": 1.0, "reprovision_delay_s": 30.0, "restore_s": 9.0,
        "restore_by_class": {"process_kill": 5.0, "node_loss": 50.0}})
    row = run_arm(sc, "ours", 7, None)
    tr = row["trace_replay"]["per_job"]["A0"]
    assert tr["restarts"] == 1 and tr["resets"] == 0
    # one recovery: 175+1+30+50 = 256 -> recovery_s = 81
    assert abs(tr["recovery_s"] - 81.0) < 1e-6, tr["recovery_s"]
