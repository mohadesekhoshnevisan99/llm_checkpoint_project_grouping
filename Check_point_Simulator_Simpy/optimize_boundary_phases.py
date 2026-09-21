"""Joint phase search on recorded iteration boundaries; periods stay fixed.

Frozen-trace screening, followed by mandatory simulator replay. The score is
pairwise overlap of job-level store-transfer envelopes, NOT goodput or a proof
of network capacity. Duration envelopes are conservatively taken from the run.
"""
import argparse
import copy
import itertools
import json
import math
from pathlib import Path


def overlap_metrics(waves):
    events = {}
    for w in waves:
        start, end = w['start_s'], w['end_s']
        if end <= start:
            raise ValueError('Transfer envelopes must have positive duration')
        events[start] = events.get(start, 0) + 1
        events[end] = events.get(end, 0) - 1
    active = peak = 0
    pair_seconds = 0.
    prev = None
    for t, delta in sorted(events.items()):
        if prev is not None:
            pair_seconds += active * (active - 1) / 2 * (t - prev)
        active += delta
        peak = max(peak, active)
        prev = t
    return dict(pair_overlap_s=pair_seconds, peak_active_job_transfers=peak)


def calibration(run):
    if 'boundaries' not in run or 'store_waves' not in run['audit']:
        raise ValueError('Rerun using the updated run_direct_contract.py to record every boundary')
    captures = {}
    for c in run['audit']['captures']:
        key = (c['job_id'], c['iteration'])
        captures[key] = min(captures.get(key, math.inf), c['start_s'])
    rows = {jid: [] for jid in run['boundaries']}
    for w in run['audit']['store_waves']:
        captured = captures[(w['job_id'], w['iteration'])]
        rows[w['job_id']].append((w['start_s'] - captured, w['end_s'] - w['start_s']))
    if any(not r for r in rows.values()):
        raise ValueError('Every job needs at least one observed store wave')
    return {j: dict(capture_to_store_s=max(x[0] for x in row),
                    transfer_s=max(x[1] for x in row)) for j, row in rows.items()}


def project(boundaries, period, phase, origin, durations):
    """Mirror coalescing and busy gating on frozen boundaries, without new training physics."""
    due, busy_until = origin + phase, -math.inf
    waves, decisions = [], []
    for b in boundaries:
        now = b['time_s']
        if not b['eligible'] or now + 1e-9 < due or now < busy_until - 1e-9:
            continue
        skipped = max(0, math.floor((now - due + 1e-9) / period))
        nominal = due + skipped * period
        start = now + durations['capture_to_store_s']
        busy_until = start + durations['transfer_s']
        waves.append(dict(start_s=start, end_s=busy_until))
        decisions.append(dict(boundary_s=now, nominal_s=nominal,
                              lag_s=now - nominal, skipped=skipped))
        due = nominal + period
    gaps = [b['boundary_s'] - a['boundary_s'] for a, b in zip(decisions, decisions[1:])]
    return dict(waves=waves, decisions=decisions, phase_s=phase,
                max_gap_s=max(gaps, default=0.),
                first_capture_s=decisions[0]['boundary_s'] if decisions else math.inf,
                last_capture_s=decisions[-1]['boundary_s'] if decisions else -math.inf)


def joint_score(options):
    metric = overlap_metrics([w for option in options for w in option['waves']])
    lag = sum(d['lag_s'] for o in options for d in o['decisions'])
    return (round(metric['pair_overlap_s'], 9), metric['peak_active_job_transfers'], round(lag, 9))


def optimize(run, phase_step=1., exhaustive_limit=200000):
    if not math.isfinite(phase_step) or phase_step <= 0:
        raise ValueError('phase_step must be positive')
    contract = run['contract']
    durations = calibration(run)
    ids = sorted(contract['jobs'])
    if set(ids) != set(run['boundaries']):
        raise ValueError('Contract and boundary job IDs differ')
    candidates, baseline = [], []
    for jid in ids:
        p = contract['jobs'][jid]
        if p['route'] != 'direct':
            raise ValueError('Only direct routes supported in this stage')
        period = p['interval_s']
        base = project(run['boundaries'][jid], period, p['phase_s'], contract['epoch_origin_s'], durations[jid])
        if len(base['decisions']) < 2:
            raise ValueError('Need at least two projected waves per job')
        baseline.append(base)
        phases = sorted({p['phase_s']} | {i * phase_step for i in range(math.ceil(period / phase_step))
                                       if i * phase_step < period})
        choices = []
        for phase in phases:
            option = project(run['boundaries'][jid], period, phase, contract['epoch_origin_s'], durations[jid])
            # Prevent a spurious win obtained by checkpointing less or dropping the tails.
            if (len(option['waves']) == len(base['waves']) and option['max_gap_s'] <= base['max_gap_s'] + 1e-8
                    and option['first_capture_s'] <= base['first_capture_s'] + 1e-8
                    and option['last_capture_s'] >= base['last_capture_s'] - 1e-8):
                choices.append(option)
        candidates.append(choices)
    combinations = math.prod(map(len, candidates))
    best, score = baseline, joint_score(baseline)
    evaluated = 0
    if combinations <= exhaustive_limit:
        method = 'exhaustive within filtered frozen-trace phase catalogue'
        for options in itertools.product(*candidates):
            value = joint_score(options)
            evaluated += 1
            if value < score:
                best, score = list(options), value
    else:
        method = 'coordinate descent; no global optimality guarantee'
        for _ in range(5):
            changed = False
            for i, choices in enumerate(candidates):
                for option in choices:
                    trial = best.copy()
                    trial[i] = option
                    value = joint_score(trial)
                    evaluated += 1
                    if value < score:
                        best, score, changed = trial, value, True
            if not changed:
                break
    output = copy.deepcopy(contract)
    for jid, option in zip(ids, best):
        output['jobs'][jid]['phase_s'] = option['phase_s']
    output['integration_ready'] = False
    output['phase_search'] = 'frozen-boundary prediction; must replay simulator'
    details = dict(method=method, candidate_combinations=combinations, evaluated=evaluated,
        baseline_score=list(joint_score(baseline)), candidate_score=list(score),
        score_order=['pair_overlap_s', 'peak_active_job_transfers', 'sum_nominal_boundary_lag_s'],
        calibration=durations, phases={j: dict(before=contract['jobs'][j]['phase_s'], after=o['phase_s'])
                                      for j, o in zip(ids, best)},
        constraints='same predicted wave count, no larger max gap, no later first or earlier last capture',
        periods_changed=False, simulator_replay_required=True, capacity_certified=False)
    return output, details


def compare(before, after):
    def summary(run):
        per_job = {}
        for jid in run['contract']['jobs']:
            # Latest rank start describes complete job capture initiation conservatively.
            starts = {}
            for c in run['audit']['captures']:
                if c['job_id'] == jid:
                    starts[c['iteration']] = max(starts.get(c['iteration'], -math.inf), c['start_s'])
            times = sorted(starts.values())
            per_job[jid] = dict(waves=len(times), max_gap_s=max((b-a for a,b in zip(times,times[1:])), default=None))
        return dict(train_end_s=run['row']['train_end_s'], store_gb_written=run['row']['store_gb_written'],
                    max_capture_lag_s=run['audit']['max_capture_lag_s'], jobs=per_job,
                    **overlap_metrics(run['audit']['store_waves']))
    b, a = summary(before), summary(after)
    if set(b['jobs']) != set(a['jobs']):
        raise ValueError('Cannot compare different job sets')
    checks = dict(same_waves_per_job=all(b['jobs'][j]['waves'] == a['jobs'][j]['waves'] for j in b['jobs']),
        max_gap_not_worse=all(a['jobs'][j]['max_gap_s'] is not None and b['jobs'][j]['max_gap_s'] is not None
                              and a['jobs'][j]['max_gap_s'] <= b['jobs'][j]['max_gap_s'] + 1e-7 for j in b['jobs']),
        same_store_bytes=abs(a['store_gb_written'] - b['store_gb_written']) <= 1e-7,
        overlap_reduced=a['pair_overlap_s'] < b['pair_overlap_s'] - 1e-7,
        completion_not_worse=a['train_end_s'] <= b['train_end_s'] + 1e-7)
    return dict(before=b, after=a, replay_checks=checks,
                passes_this_limited_replay_check=all(checks.values()),
                note='Measured envelope overlap; not exact per-link utilization or reliability validation')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--phase-step', type=float, default=1.)
    ap.add_argument('--out-contract', type=Path)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--compare-run', type=Path)
    args = ap.parse_args()
    run = json.loads(args.run.read_text(encoding='utf-8-sig'))
    if args.compare_run:
        report = compare(run, json.loads(args.compare_run.read_text(encoding='utf-8-sig')))
    else:
        if args.out_contract is None:
            ap.error('--out-contract is required for phase search')
        contract, report = optimize(run, args.phase_step)
        args.out_contract.parent.mkdir(parents=True, exist_ok=True)
        args.out_contract.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
