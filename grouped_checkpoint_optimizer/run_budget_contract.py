"""Enforce shared checkpoint-payload budgets in the simulator event allocator.

Direct-only singleton/no-failure contracts, same scope as run_direct_contract.
Adds two virtual resources to every direct-store stream. Node NIC/store limits
remain active. Records exact piecewise-constant allocator rates, not byte/time
proxies. Hooks are process-local and restored; simulator files are not edited.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import run_direct_contract as runner

NET = '__contract_checkpoint_network__'
STORE = '__contract_checkpoint_store__'


class BudgetMonitor:
    def __init__(self, network, store):
        for value in (network, store):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Budgets must be positive and finite')
        self.network, self.store = network, store
        self.samples = []
        self.tagged_flows = 0
        self.registry = None

    def snapshot(self, registry, out_resource):
        if self.registry is not None and registry is not self.registry:
            raise RuntimeError('Multiple capacity registries are outside this runner scope')
        self.registry = registry
        if registry.backend is None:
            raise RuntimeError('Event backend required')
        selected = [f for f in registry.flows if (NET, out_resource) in f.uses]
        network = sum(f.allocation for f in selected)
        store = sum(f.allocation for f in registry.flows if (STORE, out_resource) in f.uses)
        if network > self.network + 1e-7 or store > self.store + 1e-7:
            raise AssertionError('Allocator exceeded shared budget')
        sample = dict(time_s=float(registry.backend.now), network_gbps=network,
                      store_gbps=store, active_streams=len(selected),
                      stream_rates=[dict(flow_id=f._birth, allocated_gbps=f.allocation)
                                    for f in sorted(selected, key=lambda x:x._birth)])
        # Only the final allocation at a timestamp occupies a nonzero interval.
        if self.samples and self.samples[-1]['time_s'] == sample['time_s']:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)

    def report(self, expected_gb):
        volume = sum(a['network_gbps']*(b['time_s']-a['time_s'])
                     for a,b in zip(self.samples,self.samples[1:]))
        if not self.tagged_flows or not self.samples or self.samples[-1]['active_streams']:
            raise AssertionError('No observed flows or unfinished transfer accounting')
        if not math.isclose(volume, expected_gb, rel_tol=1e-7, abs_tol=1e-5):
            raise AssertionError(f'Allocation integral {volume} != completed bytes {expected_gb}')
        return dict(scope='checkpoint payload, direct-store event-engine streams in this run',
                    shared_network_budget_gbps=self.network, shared_store_budget_gbps=self.store,
                    peak_allocated_network_gbps=max(s['network_gbps'] for s in self.samples),
                    peak_allocated_store_gbps=max(s['store_gbps'] for s in self.samples),
                    integrated_network_gb=volume, completed_store_gb=expected_gb,
                    tagged_streams=self.tagged_flows, allocation_integral_matches_bytes=True,
                    runtime_checkpoint_budget_verified=True, allocation_samples=self.samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--simulator-dir', type=Path, default=Path.cwd())
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--contract', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--network-gbps', type=float, required=True)
    ap.add_argument('--store-gbps', type=float, required=True)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    sc = runner.yaml.safe_load(args.scenario.read_text(encoding='utf-8-sig'))
    if sc.get('simulation',{}).get('engine','event') != 'event' or os.getenv('SIM_ENGINE','event') != 'event':
        raise ValueError('Only event engine supported; remove polling override')
    if sc.get('cluster',{}).get('background_nic_gbps',0):
        raise ValueError('Background traffic is outside this first budget experiment')
    sys.path.insert(0,str(args.simulator_dir.resolve()))
    from checkpointing.capacity import CapacityRegistry, NodeCaps, RES_OUT, RES_IN
    from checkpointing.crossjob import STORE_NODE
    monitor = BudgetMonitor(args.network_gbps,args.store_gbps)
    original_make = CapacityRegistry.make_flow
    original_touch = CapacityRegistry._ev_touch

    def make_flow(registry, uses, demand, group=None, weight=1.):
        uses = tuple(uses)
        if (STORE_NODE, RES_IN) in uses:
            for name, budget in ((NET,monitor.network),(STORE,monitor.store)):
                if name in registry.nodes and registry.nodes[name].nic_out != budget:
                    raise ValueError('Virtual budget resource collision')
                registry.set_node(name,NodeCaps(nic_in=budget,nic_out=budget,disk_w=budget))
            uses += ((NET,RES_OUT),(STORE,RES_OUT))
            monitor.tagged_flows += 1
        return original_make(registry,uses,demand,group,weight)

    def touch(registry, seed):
        result = original_touch(registry,seed)
        monitor.snapshot(registry,RES_OUT)
        return result

    argv = ['run_direct_contract.py','--simulator-dir',str(args.simulator_dir),
            '--scenario',str(args.scenario),'--contract',str(args.contract),'--out',str(args.out),
            '--seed',str(args.seed)]
    old_argv = sys.argv
    try:
        CapacityRegistry.make_flow = make_flow
        CapacityRegistry._ev_touch = touch
        sys.argv = argv
        runner.main()
    finally:
        sys.argv = old_argv
        CapacityRegistry.make_flow = original_make
        CapacityRegistry._ev_touch = original_touch
    result = json.loads(args.out.read_text(encoding='utf-8'))
    result['runtime_budget_audit'] = monitor.report(result['row']['store_gb_written'])
    result['limitations'] = [s for s in result['limitations'] if 'aggregate caps not imported' not in s]
    result['limitations'].append('Enforced caps cover direct checkpoint payload only, not training traffic or peer/recovery')
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result['runtime_budget_audit'].items() if k != 'allocation_samples'},indent=2))


if __name__ == '__main__':
    main()
