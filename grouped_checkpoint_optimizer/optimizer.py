"""Finite-catalogue global checkpoint scheduling research prototype.

Units: seconds, GB, GB/s, failures/second. No production daemon integration.
Every candidate reserves a whole group for capture -> peer -> store drain.
Groups have disjoint nodes. Global network and store budgets remain shared.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

LOCAL_DEPS = Path(__file__).parent / '.deps'
if sys.prefix == sys.base_prefix and LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from runtime_model import RuntimePenaltyModel


def validate(data):
    dt, horizon = data['slot_s'], data['horizon_s']
    if dt <= 0 or horizon <= 0 or not math.isclose(horizon / dt, round(horizon / dt)):
        raise ValueError('Positive horizon must be an integer number of slots')
    periods = data['periods_s']
    if not periods or any(t <= 0 or not math.isclose(t / dt, round(t / dt))
                          or not math.isclose(horizon / t, round(horizon / t)) for t in periods):
        raise ValueError('Every positive period must divide horizon and contain whole slots')
    if not 0 < data.get('headroom', .85) <= 1:
        raise ValueError('headroom must be in (0, 1]')
    for key in ('network_gbps', 'store_gbps', 'store_stream_gbps'):
        if not math.isfinite(data[key]) or data[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    jobs = data['jobs']
    if not jobs or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Need nonempty jobs with unique IDs')
    # Each entry represents a disjoint homogeneous node pool, not overlapping cohorts.
    for j in jobs:
        for key in ('size_gb', 'nic_gbps', 'donor_gbps', 'free_ssd_gb', 'capture_s',
                    'failure_rate_s', 'store_failure_rate_s', 'restart_s', 'weight', 'overlap'):
            if not math.isfinite(j[key]) or j[key] < 0:
                raise ValueError(f'{j["id"]}: invalid {key}')
        if min(j['size_gb'], j['nic_gbps'], j['weight']) <= 0:
            raise ValueError('size, NIC and weight must be positive')
        for key in ('max_age_s', 'max_store_age_s'):
            if key in j and (not math.isfinite(j[key]) or j[key] <= 0):
                raise ValueError(f'{key} must be positive and finite if specified')


def partitions(data, max_group_size=8):
    """Search family, NOT enumeration of all set partitions.

    Alternate high/low checkpoint demand; vary target size. Remainders make
    sizes heterogeneous. Also try donor-rich-first packing. No fixed five.
    """
    jobs = data['jobs']
    ranked = sorted(range(len(jobs)), key=lambda i: jobs[i]['size_gb'] /
                    max(jobs[i]['donor_gbps'], 1e-9), reverse=True)
    balanced = []
    for k in range((len(ranked) + 1) // 2):
        balanced.append(ranked[k])
        if k != len(ranked) - 1 - k:
            balanced.append(ranked[-1-k])
    donor_order = sorted(range(len(jobs)), key=lambda i: jobs[i]['donor_gbps'], reverse=True)
    seen = set()
    for order in (balanced, donor_order):
        for size in range(1, min(max_group_size, len(jobs)) + 1):
            groups = [order[k:k+size] for k in range(0, len(order), size)]
            signature = tuple(sorted(tuple(sorted(g)) for g in groups))
            if signature not in seen:
                seen.add(signature)
                yield [list(g) for g in signature]


def catalogue(data, groups, max_phases=12, runtime_model=None, min_peer_rate=0.0, require_peer=False):
    """Build sparse resource vectors for all (job, route, T, phase) choices."""
    validate(data)
    jobs, dt = data['jobs'], data['slot_s']
    n, slots = len(jobs), round(data['horizon_s'] / dt)
    if sorted(i for g in groups for i in g) != list(range(n)) or any(not g for g in groups):
        raise ValueError('Groups must partition jobs exactly once')
    if max_phases < 1:
        raise ValueError('max_phases must be positive')
    # Resource rows: network calendar, store calendar, group calendars, donor SSD.
    storage_start = (2 + len(groups)) * slots
    caps = np.r_[np.full(slots, data['network_gbps'] * data.get('headroom', .85)),
                 np.full(slots, data['store_gbps'] * data.get('headroom', .85)),
                 np.ones(len(groups) * slots), [j['free_ssd_gb'] for j in jobs]]
    modes, by_job = [], [[] for _ in jobs]
    for gid, group in enumerate(groups):
        for i in group:
            j = jobs[i]
            donors = [d for d in group if d != i and jobs[d]['donor_gbps'] > 0]
            supply = sum(jobs[d]['donor_gbps'] for d in donors)
            for route in ('direct', 'peer'):
                if route == 'peer' and supply == 0:
                    continue
                cap_slots = math.ceil(j['capture_s'] / dt)
                peer_slots = (math.ceil(j['size_gb'] / min(j['nic_gbps'], supply) / dt)
                              if route == 'peer' else 0)
                drain_rate = min(data['store_stream_gbps'], supply if route == 'peer' else j['nic_gbps'])
                drain_slots = math.ceil(j['size_gb'] / drain_rate / dt)
                length = cap_slots + peer_slots + drain_slots
                lag = length * dt
                peer_lag = (cap_slots + peer_slots) * dt if route == 'peer' else lag
                allocations = ({d: j['size_gb'] * jobs[d]['donor_gbps'] / supply for d in donors}
                               if route == 'peer' else {})
                if any(2 * amount > jobs[d]['free_ssd_gb'] + 1e-9 for d, amount in allocations.items()):
                    continue
                for period in sorted(set(data['periods_s'])):
                    steps = round(period / dt)
                    if (length > steps or period + peer_lag > j.get('max_age_s', math.inf)
                            or period + lag > j.get('max_store_age_s', math.inf)):
                        continue
                    cost = j['weight'] * (cap_slots * dt / period +
                            j['overlap'] * (peer_slots + drain_slots) * dt / period +
                            j['failure_rate_s'] * (period / 2 + peer_lag + j['restart_s']) +
                            j['store_failure_rate_s'] * (period / 2 + lag + j['restart_s']))
                    phases = sorted({int(k * steps / min(steps, max_phases))
                                     for k in range(min(steps, max_phases))})
                    for phase in phases:
                        res = {}
                        for start in range(phase, slots, steps):
                            for offset in range(length):
                                t = (start + offset) % slots
                                res[(2 + gid) * slots + t] = 1.0
                                if cap_slots <= offset < cap_slots + peer_slots:
                                    res[t] = j['size_gb'] / (peer_slots * dt)
                                elif offset >= cap_slots + peer_slots:
                                    rate = j['size_gb'] / (drain_slots * dt)
                                    res[t] = rate
                                    res[slots + t] = rate
                        for d, amount in allocations.items():
                            # Previous complete version + next in-flight version, permanently reserved.
                            res[storage_start + d] = 2 * amount
                        rows = np.array(list(res), dtype=int)
                        values = np.array(list(res.values()), dtype=float)
                        if np.any(values > caps[rows] + 1e-9):
                            continue
                        by_job[i].append(len(modes))
                        runtime = (
                            runtime_model.estimate(
                                grouped=len(group) > 1,
                                phased=phase != 0,
                                group_size=len(group),
                                kpeers=1 if route == 'peer' else 0,
                            )
                            if runtime_model is not None
                            else {
                                'penalty': 0.0,
                                'estimated_peer_rate': 0.0,
                                'estimated_fallback': 0.0,
                                'estimated_own_flush': 0.0,
                            }
                        )
                        if (
                            runtime_model is not None
                            and route == "peer"
                            and runtime["estimated_peer_rate"] < min_peer_rate
                        ):
                            continue

                        modes.append(dict(
                            job=i,
                            group=gid,
                            route=route,
                            period_s=period,
                            phase_s=phase * dt,
                            frequency_hz=1 / period,
                            durable_lag_s=lag,
                            peer_lag_s=peer_lag,
                            analytical_cost=cost,
                            runtime_penalty=runtime['penalty'],
                            estimated_peer_rate=runtime['estimated_peer_rate'],
                            estimated_fallback=runtime['estimated_fallback'],
                            estimated_own_flush=runtime['estimated_own_flush'],
                            cost=cost + runtime['penalty'],
                            rows=rows,
                            values=values,
                            donor_gb={jobs[d]['id']: a for d, a in allocations.items()},
                        ))
    return modes, by_job, caps


def greedy(modes, by_job, caps):
    """Find a feasible seed, then strictly improve cost with coordinate moves."""
    usage = np.zeros_like(caps)
    chosen = {}
    order = sorted(range(len(by_job)), key=lambda i: (len(by_job[i]),
                   -max((modes[m]['durable_lag_s'] for m in by_job[i]), default=0)))
    for i in order:
        # Low duty first helps feasibility; this is not a proof of feasibility/infeasibility.
        options = sorted(by_job[i], key=lambda m: (modes[m]['durable_lag_s'] / modes[m]['period_s'], modes[m]['cost']))
        fit = next((m for m in options if np.all(usage[modes[m]['rows']] +
                    modes[m]['values'] <= caps[modes[m]['rows']] + 1e-8)), None)
        if fit is None:
            return None
        chosen[i] = fit
        mode = modes[fit]
        usage[mode['rows']] += mode['values']
    for _ in range(3):
        changed = False
        for i in order:
            old = modes[chosen[i]]
            usage[old['rows']] -= old['values']
            for m in sorted(by_job[i], key=lambda m: modes[m]['cost']):
                mode = modes[m]
                if mode['cost'] >= old['cost'] - 1e-12:
                    break
                if np.all(usage[mode['rows']] + mode['values'] <= caps[mode['rows']] + 1e-8):
                    chosen[i] = m
                    changed = True
                    break
            mode = modes[chosen[i]]
            usage[mode['rows']] += mode['values']
        if not changed:
            break
    return [chosen[i] for i in range(len(by_job))]


def verify(modes, by_job, caps, selected):
    if selected is None or len(selected) != len(by_job):
        return False, None
    if sorted(modes[m]['job'] for m in selected) != list(range(len(by_job))):
        return False, None
    usage = np.zeros_like(caps)
    for m in selected:
        usage[modes[m]['rows']] += modes[m]['values']
    return bool(np.all(usage <= caps + 1e-7)), usage


def solve_partition(data, groups, max_phases=12, seconds=3, runtime_model=None, min_peer_rate=0.0, require_peer=False):
    started = time.monotonic()
    modes, by_job, caps = catalogue(data, groups, max_phases, runtime_model, min_peer_rate)
    result = dict(groups=[[data['jobs'][i]['id'] for i in g] for g in groups],
                  mode_count=len(modes), status='unknown', scope='fixed partition and discrete catalogue')
    if any(not options for options in by_job):
        return dict(result, status='infeasible_catalogue', reason='a job has no admissible mode')
    costs = np.array([m['cost'] for m in modes])

    # When peer is mandatory, make direct modes unattractive to the
    # selector while preserving the feasibility constraints.
    if require_peer:
        costs = costs + np.array([
            1000.0 if mode.get("route") == "direct" else 0.0
            for mode in modes
        ])
    lower = sum(min(costs[m] for m in options) for options in by_job)
    independent = [min(options, key=lambda m: costs[m]) for options in by_job]
    independent_valid, independent_usage = verify(modes, by_job, caps, independent)
    positive_caps = caps > 0
    result['independent_baseline'] = dict(objective=float(lower), feasible=independent_valid,
        max_capacity_ratio=float(max(independent_usage[positive_caps] / caps[positive_caps])))
    selected = greedy(modes, by_job, caps)
    seed_cost = sum(costs[m] for m in selected) if selected is not None else math.inf
    status = 'heuristic_feasible' if selected is not None else 'unknown'
    solver_status = None
    if seconds > 0:
        n = len(by_job)
        row, col, values = [], [], []
        for k, mode in enumerate(modes):
            row.append(mode['job']); col.append(k); values.append(1.)
            row.extend(mode['rows'] + n)
            col.extend([k] * len(mode['rows']))
            values.extend(mode['values'])
        matrix = coo_matrix((values, (row, col)), shape=(n + len(caps), len(modes))).tocsc()
        opt = milp(costs, integrality=np.ones(len(modes)), bounds=Bounds(0, 1),
                   constraints=LinearConstraint(matrix, np.r_[np.ones(n), np.full(len(caps), -np.inf)],
                                                np.r_[np.ones(n), caps]),
                   options={'time_limit': seconds, 'mip_rel_gap': .001})
        solver_status = int(opt.status)
        bound = getattr(opt, 'mip_dual_bound', None)
        if bound is not None and math.isfinite(bound):
            lower = max(lower, float(bound))
        if opt.x is not None:
            candidate = list(np.flatnonzero(opt.x > .5))
            valid, _ = verify(modes, by_job, caps, candidate)
            if valid and sum(costs[m] for m in candidate) <= seed_cost + 1e-9:
                selected = candidate
                status = 'optimal_within_solver_tolerance' if opt.status == 0 else 'feasible_time_limit'
        if opt.status == 2 and selected is None:
            status = 'infeasible_catalogue'
    if require_peer:
        has_peer = (
            selected is not None
            and any(modes[m].get("route") == "peer" for m in selected)
        )

        if not has_peer:
            peer_modes = [
                m for m, mode in enumerate(modes)
                if mode.get("route") == "peer"
            ]

            forced_candidate = None

            for forced in peer_modes:
                forced_job = modes[forced]["job"]
                trial = [forced]

                # بهترین mode برای هر job دیگر، با احتساب mode اجباری
                for job_index, options in enumerate(by_job):
                    if job_index == forced_job:
                        continue

                    choices = [
                        m for m in options
                        if m != forced
                    ]
                    if not choices:
                        trial = None
                        break

                    best_mode = min(
                        choices,
                        key=lambda m: costs[m],
                    )
                    trial.append(best_mode)

                if trial is None:
                    continue

                trial_valid, _ = verify(
                    modes,
                    by_job,
                    caps,
                    trial,
                )

                if trial_valid:
                    forced_candidate = trial
                    break

            if forced_candidate is not None:
                selected = forced_candidate
                status = "feasible_with_required_peer"

    valid, usage = verify(modes, by_job, caps, selected)
    result.update(status=status, solver_status=solver_status, lower_bound=lower,
                  elapsed_s=time.monotonic() - started, feasible=valid)
    if valid:
        objective = float(sum(costs[m] for m in selected))
        slots = round(data['horizon_s'] / data['slot_s'])
        result.update(objective=objective, relative_gap_bound=max(0., objective - lower) / max(abs(objective), 1e-12),
                      peak_network_gbps=float(max(usage[:slots])), peak_store_gbps=float(max(usage[slots:2*slots])),
                      policy=[{**{k: v for k, v in modes[m].items() if k not in ('rows', 'values', 'job')},
                               'job_id': data['jobs'][modes[m]['job']]['id']} for m in selected])
    if require_peer:
        selected_policy = result.get("policy", [])
        if not any(item.get("route") == "peer" for item in selected_policy):
            return dict(
                result,
                status="infeasible_runtime_constraint",
                reason="require_peer was requested but no peer mode was selected",
            )

    return result


def optimize(data, max_group_size=8, max_phases=12, seconds=3, runtime_model=None, min_peer_rate=0.0, require_grouping=False, require_peer=False):
    started = time.monotonic()
    if max_group_size < 1 or seconds < 0:
        raise ValueError('max_group_size must be positive; seconds must be nonnegative')
    validate(data)
    results = []
    for groups in partitions(data, max_group_size):
        if require_grouping and all(len(group) == 1 for group in groups):
            continue

        result = solve_partition(
            data,
            groups,
            max_phases,
            seconds,
            runtime_model=runtime_model,
            min_peer_rate=min_peer_rate,
            require_peer=require_peer,
        )
        results.append(result)
    feasible = [r for r in results if r.get('feasible')]
    best = min(feasible, key=lambda r: r['objective']) if feasible else None
    # Report all alternatives, including infeasible and time-limited searches.
    return dict(model='group-reserved sequential peer/store pipeline v1',
                optimality_scope='Heuristic partition search; inner finite-catalogue MILP or greedy',
                elapsed_s=time.monotonic() - started, input=data, best=best, candidates=results)


def demo(count=12):
    # Synthetic inputs deliberately include donor-poor jobs and donor-rich jobs.
    return dict(slot_s=5, horizon_s=240, periods_s=[30, 60, 120, 240],
                headroom=.9, network_gbps=max(4., count * .8), store_gbps=max(2., count * .25),
                store_stream_gbps=.25, jobs=[dict(id=f'job-{i:03}', size_gb=2. + (i % 4) * 2,
                nic_gbps=2., donor_gbps=.2 if i % 3 == 0 else 2., free_ssd_gb=80.,
                capture_s=1., failure_rate_s=1 / (1800 + (i % 5) * 900), restart_s=20.,
                store_failure_rate_s=1 / 86400, weight=1. + (i % 3), overlap=.08) for i in range(count)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--demo-jobs', type=int, default=12)
    parser.add_argument('--max-group-size', type=int, default=8)
    parser.add_argument('--max-phases', type=int, default=12)
    parser.add_argument('--seconds', type=float, default=3, help='MILP limit PER partition; 0 uses greedy only')
    parser.add_argument('--output', type=Path, default=Path('outputs/result.json'))
    parser.add_argument('--runtime-calibration', type=Path)
    parser.add_argument('--min-peer-rate', type=float, default=0.0)
    parser.add_argument('--require-grouping', action='store_true')
    parser.add_argument("--require-peer", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8')) if args.input else demo(args.demo_jobs)
    runtime_model = None
    if args.runtime_calibration:
        calibration = json.loads(
            args.runtime_calibration.read_text(encoding='utf-8-sig')
        )
        runtime_model = RuntimePenaltyModel(calibration)

    report = optimize(
        data,
        args.max_group_size,
        args.max_phases,
        args.seconds,
        runtime_model=runtime_model,
        min_peer_rate=args.min_peer_rate,
        require_grouping=args.require_grouping,
        require_peer=args.require_peer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    best = report['best']
    print(json.dumps({'output': str(args.output.resolve()), 'partitions': len(report['candidates']),
                      'best': {k: best[k] for k in ('objective', 'status', 'relative_gap_bound', 'groups',
                                                   'peak_network_gbps', 'peak_store_gbps')} if best else None}))


if __name__ == '__main__':
    main()
