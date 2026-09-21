"""Execute direct-only singleton contracts using the existing simulator.

First integration stage: fresh capture at the first completed iteration after
the nominal deadline, never a previously captured state held for its slot.
This DOES NOT certify the optimizer's exact calendar or enforce its aggregate
budgets. Original simulator resource accounting remains active. No source file
is edited: a version-checked training-loop hook exists only during this process.
"""
import argparse
import copy
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import textwrap

local_deps = Path(__file__).parent / '.runner-deps'
if sys.prefix == sys.base_prefix and local_deps.exists():
    sys.path.insert(0, str(local_deps))
import yaml


def validate_contract(sc, contract, allow_failures=False):
    if contract.get('schema') != 'grouped-checkpoint-contract-v1':
        raise ValueError('Unsupported contract schema')
    if sc.get('job_arrivals') or sc.get('failure_trace'):
        raise ValueError('Dynamic arrivals and trace replay are unsupported')
    if not allow_failures and sc.get('preemption'):
        raise ValueError('Scenario has preemption; pass --allow-failures explicitly')
    if not allow_failures and (sc['failures']['per_node_per_second'] != 0 or sc['failures'].get('rack_weight', 0)):
        raise ValueError('Failure rates must be zero')
    # Fail closed for scheduled/replay modes, whose schemas may vary by version.
    if any('scheduled' in k or 'trace' in k or k.startswith('rack_') for k in sc['failures']):
        raise ValueError('Remove scheduled/trace/rack failure configuration for this smoke test')
    if any(k in sc for k in ('scheduled_failures', 'rack_failures', 'failure_replay', 'rack_scheduled', 'rack_size')):
        raise ValueError('Scheduled failures/replay not supported')
    ids = {f'{name}{i}' for name, spec in sc['classes'].items() for i in range(spec['count'])}
    if set(contract['jobs']) != ids:
        raise ValueError('Contract job IDs differ from scenario')
    if any('model' in spec or spec.get('cohorts', spec['ranks']) != spec['ranks']
           for spec in sc['classes'].values()):
        raise ValueError('First integration supports literal shards and one worker per rank')
    groups = []
    origin = contract['epoch_origin_s']
    if not math.isfinite(origin) or origin < 0:
        raise ValueError('Invalid epoch origin')
    for jid, p in contract['jobs'].items():
        if p['route'] != 'direct' or p['allowed_donor_job_ids'] or p['donor_gb_per_version']:
            raise ValueError('This runner accepts direct-only contracts, not peer routes')
        if not math.isfinite(p['interval_s']) or p['interval_s'] <= 0 or not 0 <= p['phase_s'] < p['interval_s']:
            raise ValueError(f'Invalid timing: {jid}')
        groups.append(p['group_id'])
    if len(set(groups)) != len(groups):
        raise ValueError('Only singleton groups supported; shared-group exclusion is not implemented')


class BoundaryScheduler:
    def __init__(self, contract):
        self.contract = contract
        self.next_due = {jid: contract['epoch_origin_s'] + p['phase_s']
                         for jid, p in contract['jobs'].items()}
        self.events = []
        self.boundaries = {}
        self.busy_boundaries = 0

    def due(self, runtime):
        jid, now = runtime.config.job_id, runtime.backend.now
        self.boundaries.setdefault(jid, []).append(dict(time_s=now, iteration=runtime.it,
                                                        eligible=runtime.it < runtime.iterations))
        if runtime.it >= runtime.iterations or now + 1e-9 < self.next_due[jid]:
            return False
        if any(not p.triggered for p in runtime.checkpoint_strategy.background_flushes):
            self.busy_boundaries += 1
            return False
        period = self.contract['jobs'][jid]['interval_s']
        original_due = self.next_due[jid]
        skipped = max(0, math.floor((now - original_due + 1e-9) / period))
        nominal = original_due + skipped * period
        self.next_due[jid] = nominal + period
        self.events.append(dict(job_id=jid, iteration=runtime.it, nominal_s=nominal,
                                boundary_s=now, boundary_lag_s=now - nominal,
                                skipped_deadlines=skipped))
        return True


def patched_run(module, scheduler):
    source = textwrap.dedent(inspect.getsource(module.CohortJobRuntime.run))
    start = '    strat = self.checkpoint_strategy\n'
    decision = ('        persist_wave = (self.it % self.config.checkpoint_every == 0\n'
                '                        and self.it < self.iterations)')
    launch = '''            for w in self.workers.values():
                self.backend.process(
                    strat.checkpoint(
                        w,
                        self.config,
                        iteration=self.it,
                        checkpoint_epoch=w.checkpoint_epoch,
                    )
                )'''
    for anchor in (start, decision, launch):
        if source.count(anchor) != 1:
            raise RuntimeError('Unsupported training-loop version: source anchor mismatch; no files modified')
    source = source.replace(start, start + '    self.capture_every = 0\n    self.store_every = 0\n')
    source = source.replace(decision, '        persist_wave = _contract_scheduler.due(self)')
    source = source.replace(launch, '''            captures = [self.backend.process(strat.checkpoint(
                w, self.config, iteration=self.it, checkpoint_epoch=w.checkpoint_epoch))
                for w in self.workers.values()]
            yield self.backend.all_of(captures)''')
    namespace = dict(module.__dict__, _contract_scheduler=scheduler)
    exec(compile(source, '<direct-contract-boundary-hook>', 'exec'), namespace)
    return namespace['run']


def audit_trace(path, events):
    requests = {(e['job_id'], e['iteration']): e for e in events}
    captures, store_events, other_paths = [], 0, {}
    store_waves = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            e = json.loads(line)
            if e['operation'] == 'checkpoint_stage_gpu_to_dram':
                request = requests.get((e['job_id'], e['iteration']))
                if request is None:
                    raise AssertionError('Capture has no matching contract decision')
                captures.append(dict(job_id=e['job_id'], iteration=e['iteration'], rank=e['rank'],
                                     start_s=e['start'], nominal_s=request['nominal_s'],
                                     lag_s=e['start'] - request['nominal_s']))
            route = (e.get('details') or {}).get('path')
            if route == 'store':
                store_events += 1
                key = (e['job_id'], e['iteration'])
                wave = store_waves.setdefault(key, dict(job_id=key[0], iteration=key[1],
                    start_s=e['start'], end_s=e['end'], size_gb=0.))
                wave['start_s'] = min(wave['start_s'], e['start'])
                wave['end_s'] = max(wave['end_s'], e['end'])
                wave['size_gb'] += e.get('data_gb') or 0.
            elif route:
                other_paths[route] = other_paths.get(route, 0) + 1
    if other_paths or not captures or not store_events:
        raise AssertionError(f'Unexpected route/capture trace: {other_paths}')
    return dict(rank_captures=len(captures), store_trace_events=store_events,
                max_capture_lag_s=max(e['lag_s'] for e in captures), captures=captures,
                store_waves=list(store_waves.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator-dir', type=Path, default=Path.cwd())
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--allow-failures', action='store_true',
                        help='Allow scenario preemption/organic failures; output is recovery audit')
    args = parser.parse_args()
    sys.path.insert(0, str(args.simulator_dir.resolve()))
    import run_scenario as sim
    sc = yaml.safe_load(args.scenario.read_text(encoding='utf-8-sig'))
    contract = json.loads(args.contract.read_text(encoding='utf-8-sig'))
    validate_contract(sc, contract, allow_failures=args.allow_failures)
    # Validate the simulator's authoritative event parsers as well.
    jobs = sim.expand_jobs(sc)
    if any(sim.normalize_scheduled_failures(sc, jobs).values()):
        raise ValueError('Scheduled failures not supported')
    arm = 'contract_direct_boundary'
    sc = copy.deepcopy(sc)
    sc['arms'] = {arm: {'store_mode': True, 'slots': False, 'kpeers': True, 'controller_epoch_s': 0}}
    scheduler = BoundaryScheduler(contract)
    original = sim.CohortJobRuntime.run
    replacement = patched_run(sim, scheduler)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = args.out.with_suffix('.trace.jsonl.gz')
    try:
        sim.CohortJobRuntime.run = replacement
        row = sim.run_arm(sc, arm, args.seed, trace)
    finally:
        sim.CohortJobRuntime.run = original
    if not args.allow_failures and (row['failures'] or row['restores'] or row['fallback_flushes'] or row['peer_flushes']):
        raise AssertionError('Unexpected failures or non-store routes')
    trace_audit = audit_trace(trace, scheduler.events)
    report = dict(schema='direct-boundary-integration-v1', row=row, decisions=scheduler.events,
                  boundaries=scheduler.boundaries, contract=contract,
                  audit=trace_audit, busy_boundaries=scheduler.busy_boundaries,
                  simulator_sha256=hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest(),
                  optimizer_calendar_verified=False,
                  recovery_audit=args.allow_failures,
                  limitations=['Direct singleton groups only; no failures or dynamic arrivals',
                               'Capture at iteration boundary, not exact nominal phase',
                               'Simulator native per-node/store capacities; optimizer aggregate caps not imported',
                               'Simulator per-stream cap retained, not optimizer per-job stream cap',
                               'No assertion of objective equivalence or goodput improvement'])
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(args.out), trace=str(trace), train_end_s=row['train_end_s'],
                         store_gb_written=row['store_gb_written'],
                         checkpoint_waves=len(scheduler.events), rank_captures=trace_audit['rank_captures'],
                         max_capture_lag_s=trace_audit['max_capture_lag_s'],
                         optimizer_calendar_verified=False), indent=2))


if __name__ == '__main__':
    main()
