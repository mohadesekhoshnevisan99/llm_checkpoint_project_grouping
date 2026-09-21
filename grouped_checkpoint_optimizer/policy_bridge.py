"""Export a NEW interface contract; NOT compatible with simulator --policy yet.

No optimizer dependency: this module can later be imported by the simulator.
The simulator must implement capture timing, group exclusion, route enforcement,
global capacity accounting and per-node shard placement before using it live.
"""
import argparse
import json
import math
from pathlib import Path


def export_contract(report, epoch_origin_s=0.):
    if not math.isfinite(epoch_origin_s):
        raise ValueError('epoch_origin_s must be finite')
    best = report.get('best')
    if not best or not best.get('feasible'):
        raise ValueError('No feasible optimizer policy to export')
    group_of = {}
    for gid, group in enumerate(best['groups']):
        for job_id in group:
            if job_id in group_of:
                raise ValueError('Duplicate group membership')
            group_of[job_id] = gid
    policies = {}
    for p in best['policy']:
        jid = p['job_id']
        if jid in policies or jid not in group_of:
            raise ValueError('Invalid policy membership')
        period, phase = p['period_s'], p['phase_s']
        if not math.isfinite(period) or period <= 0 or not 0 <= phase < period:
            raise ValueError('Invalid period or phase')
        if p['route'] not in ('direct', 'peer') or p['group'] != group_of[jid]:
            raise ValueError('Invalid route or group')
        donors = list(p['donor_gb'])
        if any(d == jid or group_of.get(d) != group_of[jid] for d in donors):
            raise ValueError('Donor outside group')
        if (p['route'] == 'direct' and donors) or (p['route'] == 'peer' and not donors):
            raise ValueError('Route/donor mismatch')
        policies[jid] = dict(group_id=group_of[jid], interval_s=period, phase_s=phase,
                             route=p['route'], allowed_donor_job_ids=donors,
                             donor_gb_per_version=p['donor_gb'])
    if set(policies) != set(group_of):
        raise ValueError('Missing job policy')
    return dict(schema='grouped-checkpoint-contract-v1', integration_ready=False,
                epoch_origin_s=epoch_origin_s, horizon_s=report['input']['horizon_s'],
                time_semantics='fresh capture starts at epoch_origin_s + phase_s + k * interval_s; k >= 0',
                warning='Research interface only. Do not pass to existing run_scenario.py --policy.',
                jobs=policies)


def next_capture_time(contract, job_id, now_s):
    """First nominal capture time strictly after now; missed slots are skipped.

    Caller must still check readiness, active transfer, failure and capacity.
    For an initial event exactly at epoch, schedule the first event explicitly.
    """
    if not math.isfinite(now_s):
        raise ValueError('now_s must be finite')
    p = contract['jobs'][job_id]
    first = contract['epoch_origin_s'] + p['phase_s']
    k = max(0, math.floor((now_s - first) / p['interval_s']) + 1)
    return first + k * p['interval_s']


def donor_allowed(contract, source_job_id, donor_job_id):
    p = contract['jobs'].get(source_job_id)
    return bool(p and source_job_id != donor_job_id and
                donor_job_id in p['allowed_donor_job_ids'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/policy_contract.json'))
    parser.add_argument('--epoch-origin-s', type=float, default=0.)
    args = parser.parse_args()
    contract = export_contract(json.loads(args.result.read_text(encoding='utf-8-sig')), args.epoch_origin_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Exported {len(contract["jobs"])} jobs to {args.output.resolve()}')
    print(contract['warning'])


if __name__ == '__main__':
    main()
