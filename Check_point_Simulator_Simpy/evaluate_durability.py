"""Audit complete direct-store checkpoints and shared transfer budgets from trace.

No re-simulation. Requires the direct-boundary runner result and its gzip trace.
Logical snapshot time is the recorded iteration boundary. A job version is
durable only after every rank's full bytes have arrived. No initial durable
checkpoint is assumed. Transfer-rate profiles are explicitly average-rate proxies.
"""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path


def age_metrics(versions, end_s, max_age_s, startup_grace_s):
    if end_s < 0 or max_age_s <= 0 or startup_grace_s < 0:
        raise ValueError('Invalid freshness parameters')
    updates = defaultdict(list)
    for v in versions:
        if v['ready_s'] < v['snapshot_s']:
            raise ValueError('Version ready before snapshot')
        if v['ready_s'] <= end_s:
            updates[v['ready_s']].append(v['snapshot_s'])
    ordered = sorted(updates)
    first = ordered[0] if ordered else end_s
    latest = -math.inf
    max_age = integral = violations = protected = 0.
    for i, t in enumerate(ordered):
        latest = max(latest, max(updates[t]))
        until = ordered[i+1] if i+1 < len(ordered) else end_s
        dt = until - t
        if dt <= 0:
            continue
        max_age = max(max_age, until - latest)
        integral += ((t - latest) + (until - latest)) * dt / 2
        protected += dt
        violations += max(0., until - max(t, latest + max_age_s))
    return dict(first_durable_s=ordered[0] if ordered else None,
                no_durable_version_s=first,
                startup_excess_s=max(0., first - startup_grace_s),
                protected_observation_s=protected,
                max_age_after_first_durable_s=max_age if protected else None,
                mean_age_after_first_durable_s=integral / protected if protected else None,
                age_violation_s=violations,
                passes_observed_age_and_startup=bool(protected > 0 and violations <= 1e-8
                                                     and first <= startup_grace_s + 1e-8))


def traffic_metrics(transfers, budget):
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Budget must be positive')
    events = defaultdict(float)
    same_windows = defaultdict(float)
    total = 0.
    for e in transfers:
        lo, hi, size = e['start'], e['end'], e['data_gb']
        if hi <= lo or size < 0:
            raise ValueError('Invalid transfer duration/bytes')
        rate = size / (hi - lo)
        events[lo] += rate
        events[hi] -= rate
        same_windows[(lo, hi)] += size
        total += size
    peak = current = excess_seconds = 0.
    previous = None
    for t, change in sorted(events.items()):
        if previous is not None and current > budget + 1e-8:
            excess_seconds += t - previous
        current += change
        peak = max(peak, current)
        previous = t
    # A necessary lower bound: transfers sharing an exact window must move all
    # their bytes within that window. Other overlapping windows may strengthen it.
    lower = max((size / (hi-lo) for (lo, hi), size in same_windows.items()), default=0.)
    return dict(budget_gbps=budget, total_gb=total,
                peak_uniform_rate_proxy_gbps=peak,
                uniform_rate_proxy_over_budget_s=excess_seconds,
                necessary_peak_lower_bound_gbps=lower,
                budget_verdict=('infeasible_for_recorded_intervals' if lower > budget + 1e-8
                                else 'not_disproved_not_certified'),
                instantaneous_capacity_verified=False)


def evaluate(run, trace, max_age_s, startup_grace_s, network_gbps, store_gbps):
    if run.get('schema') != 'direct-boundary-integration-v1' or 'boundaries' not in run:
        raise ValueError('Use output from the updated direct-boundary runner')
    if run['row'].get('failures') or run['row'].get('censored_jobs'):
        raise ValueError('This audit currently supports completed failure-free runs')
    if run['row'].get('peer_flushes') or run['row'].get('fallback_flushes'):
        raise ValueError('Only direct-store transfers are supported')
    decisions = {(e['job_id'], e['iteration']): e for e in run['decisions']}
    captures, sends = defaultdict(dict), defaultdict(lambda: defaultdict(list))
    transfers = []
    with gzip.open(trace, 'rt', encoding='utf-8') as f:
        for line in f:
            e = json.loads(line)
            key = (e.get('job_id'), e.get('iteration'))
            if e.get('operation') == 'checkpoint_stage_gpu_to_dram':
                if key not in decisions or e['rank'] in captures[key]:
                    raise ValueError('Unexpected or duplicate rank capture')
                captures[key][e['rank']] = e
            if (e.get('details') or {}).get('path') == 'store' and e.get('category') == 'Checkpoint':
                if key not in decisions:
                    raise ValueError('Store version without a contract decision')
                sends[key][e['rank']].append(e)
                transfers.append(e)
    nodes = {a['job_id']: a['nodes'] for a in run['row']['arrival_manifest']}
    complete, incomplete = defaultdict(list), []
    for key, decision in decisions.items():
        ranks = captures[key]
        arrived = sends[key]
        full = len(ranks) == nodes[key[0]] and set(ranks) == set(arrived)
        if full:
            full = all(math.isclose(sum(e['data_gb'] for e in arrived[r]), ranks[r]['data_gb'],
                                    rel_tol=1e-7, abs_tol=1e-7) for r in ranks)
        if not full:
            incomplete.append(dict(job_id=key[0], iteration=key[1]))
            continue
        ready = max(e['end'] for events in arrived.values() for e in events)
        complete[key[0]].append(dict(iteration=key[1], snapshot_s=decision['boundary_s'], ready_s=ready))
    jobs = {}
    for jid, boundaries in run['boundaries'].items():
        if not boundaries or boundaries[-1]['eligible']:
            raise ValueError('Missing final job boundary')
        end = boundaries[-1]['time_s']
        jobs[jid] = dict(training_end_s=end, complete_versions=len(complete[jid]),
                         versions=complete[jid],
                         **age_metrics(complete[jid], end, max_age_s, startup_grace_s))
    return dict(schema='durability-budget-audit-v1',
                max_age_s=max_age_s, startup_grace_s=startup_grace_s,
                jobs=jobs, incomplete_versions=incomplete,
                network=traffic_metrics(transfers, network_gbps),
                store=traffic_metrics(transfers, store_gbps),
                limitations=[
                    'Direct-store route only; no failure/recovery or peer partial reconstruction',
                    'No initial durable checkpoint assumed; startup reported separately',
                    'Freshness observed until each job finishes; completed transfers afterwards excluded from age',
                    'Age uses logical iteration-boundary timestamp, not newest live training state',
                    'Uniform transfer rates are proxies; lower-bound violation can refute a budget, passing cannot certify it',
                    'Network budget covers checkpoint payload only, excludes training and protocol traffic',
                ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--trace', type=Path)
    ap.add_argument('--max-age-s', required=True, type=float)
    ap.add_argument('--startup-grace-s', type=float, default=0.)
    ap.add_argument('--network-gbps', required=True, type=float)
    ap.add_argument('--store-gbps', required=True, type=float)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    for value in (args.max_age_s, args.startup_grace_s):
        if not math.isfinite(value):
            ap.error('Age parameters must be finite')
    report = evaluate(json.loads(args.run.read_text(encoding='utf-8-sig')),
                      args.trace or args.run.with_suffix('.trace.jsonl.gz'),
                      args.max_age_s, args.startup_grace_s, args.network_gbps, args.store_gbps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    brief = {**report, 'jobs': {j: {k: v for k, v in metrics.items() if k != 'versions'}
                                for j, metrics in report['jobs'].items()}}
    print(json.dumps(brief, indent=2))


if __name__ == '__main__':
    main()
