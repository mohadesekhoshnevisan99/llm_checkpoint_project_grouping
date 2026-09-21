"""Joint period/phase search with complete-version age and shared profile budgets.

Direct-store only, on frozen iteration boundaries. Uses measured per-rank
transfer templates. Objective: minimum bytes over the observed finite horizon,
then pairwise overlap. It is not failure-cost optimization or a runtime capacity
certificate. A simulator replay and durability audit are mandatory.
"""
import argparse
from collections import defaultdict
import copy
import gzip
import json
import math
from pathlib import Path
import time

from evaluate_durability import age_metrics, evaluate, traffic_metrics
from optimize_boundary_phases import project, overlap_metrics


def templates_from_trace(run, trace):
    decisions = {(d['job_id'], d['iteration']): d for d in run['decisions']}
    manifest = {a['job_id']: a['nodes'] for a in run['row']['arrival_manifest']}
    samples = defaultdict(list)
    with gzip.open(trace, 'rt', encoding='utf-8') as f:
        for line in f:
            e = json.loads(line)
            if e.get('category') != 'Checkpoint' or (e.get('details') or {}).get('path') != 'store':
                continue
            if e['details'].get('chunk_count', 1) != 1:
                raise ValueError('This template extractor requires one logged transfer per rank/wave')
            key = (e['job_id'], e['iteration'])
            if key not in decisions:
                raise ValueError('Transfer without decision')
            samples[(e['job_id'], e['rank'])].append(dict(
                offset_s=e['start'] - decisions[key]['boundary_s'],
                duration_s=e['end'] - e['start'], size_gb=e['data_gb']))
    templates = {}
    for jid, count in manifest.items():
        ranks = sorted(r for j, r in samples if j == jid)
        if len(ranks) != count:
            raise ValueError('Missing rank measurements')
        templates[jid] = []
        for rank in ranks:
            rows = samples[(jid, rank)]
            size = rows[0]['size_gb']
            if any(not math.isclose(r['size_gb'], size, rel_tol=1e-8) for r in rows):
                raise ValueError('Variable checkpoint sizes unsupported')
            offset = max(r['offset_s'] for r in rows)
            duration = max(r['duration_s'] for r in rows)
            if size <= 0 or duration <= 0 or offset < -1e-7:
                raise ValueError('Invalid rank template')
            templates[jid].append(dict(rank=rank, offset_s=max(0., offset),
                                       duration_s=duration, size_gb=size))
    return templates


def make_option(boundaries, templates, period, phase, origin, max_age, startup):
    offset = min(t['offset_s'] for t in templates)
    finish = max(t['offset_s'] + t['duration_s'] for t in templates)
    projection = project(boundaries, period, phase, origin,
                         dict(capture_to_store_s=offset, transfer_s=finish-offset))
    versions = [dict(snapshot_s=d['boundary_s'], ready_s=d['boundary_s']+finish)
                for d in projection['decisions']]
    freshness = age_metrics(versions, boundaries[-1]['time_s'], max_age, startup)
    if not freshness['passes_observed_age_and_startup']:
        return None
    transfers = [dict(start=d['boundary_s']+t['offset_s'],
                      end=d['boundary_s']+t['offset_s']+t['duration_s'], data_gb=t['size_gb'])
                 for d in projection['decisions'] for t in templates]
    return dict(period_s=period, phase_s=phase, projection=projection, freshness=freshness,
                transfers=transfers, bytes_gb=sum(t['data_gb'] for t in transfers))


def solve(run, templates, periods, phase_step, max_age, startup, network_budget, store_budget, seconds=10):
    for value in [*periods, phase_step, max_age, network_budget, store_budget, seconds]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Periods, steps, budgets and time limit must be positive and finite')
    if not periods or not math.isfinite(startup) or startup < 0:
        raise ValueError('Invalid periods/startup')
    contract = run['contract']
    if run['row'].get('failures') or run['row'].get('censored_jobs'):
        raise ValueError('Completed failure-free baseline required')
    ids = sorted(contract['jobs'])
    if set(ids) != set(templates) or set(ids) != set(run['boundaries']):
        raise ValueError('Job sets differ')
    if any(p['route'] != 'direct' for p in contract['jobs'].values()):
        raise ValueError('Direct-only search')
    if len({p['group_id'] for p in contract['jobs'].values()}) != len(ids):
        raise ValueError('Singleton groups only in this integration stage')
    if any(not b or b[-1]['eligible'] for b in run['boundaries'].values()):
        raise ValueError('Need final boundary for every job')
    budget = min(network_budget, store_budget)
    candidates, counts = [], {}
    for jid in ids:
        unique = {}
        p0 = contract['jobs'][jid]
        for period in sorted(set(periods)):
            phases = {k * phase_step for k in range(math.ceil(period / phase_step))}
            if period == p0['interval_s']:
                phases.add(p0['phase_s'])
            for phase in sorted(phases):
                option = make_option(run['boundaries'][jid], templates[jid], period, phase,
                                     contract['epoch_origin_s'], max_age, startup)
                if option is None:
                    continue
                if traffic_metrics(option['transfers'], budget)['peak_uniform_rate_proxy_gbps'] > budget + 1e-8:
                    continue
                signature = tuple(d['boundary_s'] for d in option['projection']['decisions'])
                unique.setdefault(signature, option)
        choices = sorted(unique.values(), key=lambda o: (o['bytes_gb'], o['freshness']['max_age_after_first_durable_s'], o['phase_s']))
        counts[jid] = len(choices)
        candidates.append(choices)
    report = dict(scope='finite-horizon frozen boundaries and measured rank-profile catalogue',
                  objective='minimum checkpoint GB, then pair-overlap seconds',
                  candidates_per_job=counts, max_age_s=max_age, startup_grace_s=startup,
                  network_budget_gbps=network_budget, store_budget_gbps=store_budget,
                  runtime_capacity_certified=False, simulator_replay_required=True)
    if any(not c for c in candidates):
        return None, dict(report, status='infeasible_catalogue', reason='a job has no admissible option')
    suffix = [0.] * (len(ids)+1)
    for i in range(len(ids)-1, -1, -1):
        suffix[i] = suffix[i+1] + candidates[i][0]['bytes_gb']
    best = None
    best_score = (math.inf, math.inf)
    started = time.monotonic()
    timeout = False
    visited = 0

    def dfs(i, chosen, transfers, cost):
        nonlocal best, best_score, visited, timeout
        if time.monotonic() - started >= seconds:
            timeout = True
            return
        visited += 1
        if cost + suffix[i] > best_score[0] + 1e-8:
            return
        if i == len(ids):
            overlap = overlap_metrics([w for o in chosen for w in o['projection']['waves']])['pair_overlap_s']
            score = (round(cost, 8), round(overlap, 8))
            if score < best_score:
                best_score, best = score, chosen.copy()
            return
        for option in candidates[i]:
            combined = transfers + option['transfers']
            if traffic_metrics(combined, budget)['peak_uniform_rate_proxy_gbps'] <= budget + 1e-8:
                dfs(i+1, chosen+[option], combined, cost+option['bytes_gb'])
            if timeout:
                return
    dfs(0, [], [], 0.)
    report.update(search_s=time.monotonic()-started, visited_nodes=visited,
                  independent_bytes_lower_bound=suffix[0], search_complete=not timeout)
    if best is None:
        return None, dict(report, status='unknown_time_limit' if timeout else 'infeasible_catalogue')
    result = copy.deepcopy(contract)
    result['integration_ready'] = False
    result.pop('phase_search', None)
    result['period_phase_search'] = 'frozen-boundary rank-profile model; replay required'
    result['evaluation_horizon_s'] = max(b[-1]['time_s'] for b in run['boundaries'].values())
    selected = {}
    for jid, o in zip(ids, best):
        result['jobs'][jid]['interval_s'] = o['period_s']
        result['jobs'][jid]['phase_s'] = o['phase_s']
        selected[jid] = dict(period_s=o['period_s'], phase_s=o['phase_s'],
                             predicted_waves=len(o['projection']['waves']), predicted_gb=o['bytes_gb'],
                             **o['freshness'])
    combined = [t for o in best for t in o['transfers']]
    report.update(status='feasible_time_limit' if timeout else 'optimal_in_frozen_catalogue',
                  selected=selected, total_gb=best_score[0], pair_overlap_s=best_score[1],
                  predicted_peak_gbps=traffic_metrics(combined,budget)['peak_uniform_rate_proxy_gbps'],
                  byte_gap_to_independent_bound=best_score[0]-suffix[0])
    return result, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--trace', type=Path)
    ap.add_argument('--periods', nargs='+', type=float, default=[30,45,60,90,120])
    ap.add_argument('--phase-step', type=float, default=3)
    ap.add_argument('--max-age-s', type=float, default=120)
    ap.add_argument('--startup-grace-s', type=float, default=45)
    ap.add_argument('--network-gbps', type=float, required=True)
    ap.add_argument('--store-gbps', type=float, required=True)
    ap.add_argument('--seconds', type=float, default=10)
    ap.add_argument('--out-contract', type=Path)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--validate-replay', action='store_true')
    args = ap.parse_args()
    run = json.loads(args.run.read_text(encoding='utf-8-sig'))
    trace = args.trace or args.run.with_suffix('.trace.jsonl.gz')
    if args.validate_replay:
        report = evaluate(run, trace, args.max_age_s, args.startup_grace_s, args.network_gbps, args.store_gbps)
        report['passes_observed_freshness'] = not report['incomplete_versions'] and all(
            j['passes_observed_age_and_startup'] for j in report['jobs'].values())
        report['budget_refuted_by_trace'] = any(report[k]['budget_verdict'] == 'infeasible_for_recorded_intervals'
                                              for k in ('network','store'))
        report['full_runtime_capacity_certified'] = False
        brief_report = {**report, 'jobs': {j: {k:v for k,v in m.items() if k != 'versions'}
                                         for j,m in report['jobs'].items()}}
    else:
        if args.out_contract is None:
            ap.error('--out-contract required for optimization')
        templates = templates_from_trace(run, trace)
        result, report = solve(run, templates, args.periods, args.phase_step, args.max_age_s,
                               args.startup_grace_s, args.network_gbps, args.store_gbps, args.seconds)
        report['rank_templates'] = templates
        if result is not None:
            args.out_contract.parent.mkdir(parents=True, exist_ok=True)
            args.out_contract.write_text(json.dumps(result, indent=2), encoding='utf-8')
        else:
            report['contract_written'] = False
            report['warning'] = 'Do not replay any old contract at the requested output path'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(brief_report if args.validate_replay else
                     {k:v for k,v in report.items() if k != 'rank_templates'}, indent=2))


if __name__ == '__main__':
    main()
