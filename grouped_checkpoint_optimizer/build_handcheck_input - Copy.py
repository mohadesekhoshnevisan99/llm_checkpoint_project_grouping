"""Convert a static no-failure input audit into a labelled optimizer smoke test.

This is an aggregate surrogate, not an equivalent simulator model. Budgets and
overlap are explicit experiment choices. Simulator donor slots are transient
write reservations; they cannot be interpreted as persistent free SSD bytes.
"""
import argparse
import json
import math
from pathlib import Path


def convert(audit, *, network_gbps, free_ssd_gb_per_node, store_stream_gbps,
            overlap, period_s=30, slot_s=1):
    if audit.get('schema') != 'simulator-input-audit-v1':
        raise ValueError('Expected simulator-input-audit-v1')
    if audit.get('preemption') is not None or audit['failures']['per_node_per_second'] != 0:
        raise ValueError('This smoke-test adapter only supports zero failure rates and no preemption')
    for key, value in dict(network_gbps=network_gbps, free_ssd_gb_per_node=free_ssd_gb_per_node,
                           store_stream_gbps=store_stream_gbps, period_s=period_s, slot_s=slot_s).items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(overlap) or overlap < 0:
        raise ValueError('overlap must be finite and nonnegative')
    if not math.isclose(period_s / slot_s, round(period_s / slot_s)):
        raise ValueError('period_s must contain a whole number of slots')
    store = audit['store_config']
    jobs = []
    for j in audit['jobs']:
        n = j['nodes']
        if n <= 0 or int(n) != n:
            raise ValueError('Invalid node count')
        # Source pool outbound capacity; donor service bounded by both NIC directions and disk.
        donor_rate = n * min(j['nic_in_gbps_per_node'], j['nic_out_gbps_per_node'], j['ssd_gbps_per_node'])
        job = dict(id=j['id'], size_gb=j['checkpoint_gb_job'],
                   nic_gbps=n * j['nic_out_gbps_per_node'], donor_gbps=donor_rate,
                   free_ssd_gb=n * free_ssd_gb_per_node,
                   capture_s=j['nominal_capture_s_per_rank'], failure_rate_s=0.,
                   store_failure_rate_s=0., restart_s=0., weight=n, overlap=overlap)
        if j['configured_rpo_s'] is not None:
            job['max_age_s'] = j['configured_rpo_s']
        jobs.append(job)
    if not jobs or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Need unique nonempty job list')
    if sum(j['nodes'] for j in audit['jobs']) != audit['active_nodes']:
        raise ValueError('Active node count mismatch')
    return dict(slot_s=slot_s, horizon_s=4 * period_s, periods_s=[period_s], headroom=1.,
                network_gbps=network_gbps, store_gbps=min(store['in_gbps'], store['disk_gbps']),
                store_stream_gbps=store_stream_gbps, jobs=jobs,
                _adapter=dict(purpose='fixed-period aggregate smoke test; NOT calibrated integration',
                    runtime_integration_ready=False,
                    explicit_choices=dict(network_gbps=network_gbps,
                        free_ssd_gb_per_node=free_ssd_gb_per_node,
                        store_stream_gbps_per_job=store_stream_gbps, overlap=overlap),
                    assumptions=[
                        'Disjoint homogeneous node pools; pooled bandwidth is not a per-node placement proof',
                        'Configured RPO is interpreted as maximum recoverable checkpoint age',
                        'Capture is a nominal parallel-rank duration, rounded up by the optimizer',
                        'Weight equals node count; no extra headroom',
                        'Donor retention budget is chosen explicitly, not inferred from hosted_cap',
                        'Per-job stream cap is an explicit surrogate, not the simulator per-stream cap',
                        'One fixed period: this run cannot demonstrate frequency optimization',
                        'Zero hazards: this run cannot demonstrate peer recovery benefits',
                    ]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--network-gbps', required=True, type=float)
    parser.add_argument('--free-ssd-gb-per-node', required=True, type=float)
    parser.add_argument('--store-stream-gbps', required=True, type=float)
    parser.add_argument('--overlap', required=True, type=float)
    parser.add_argument('--period-s', default=30., type=float)
    parser.add_argument('--slot-s', default=1., type=float)
    args = parser.parse_args()
    data = convert(json.loads(args.audit.read_text(encoding='utf-8-sig')),
                   network_gbps=args.network_gbps, free_ssd_gb_per_node=args.free_ssd_gb_per_node,
                   store_stream_gbps=args.store_stream_gbps, overlap=args.overlap,
                   period_s=args.period_s, slot_s=args.slot_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f'Wrote {len(data["jobs"])} jobs to {args.output}')
    print(json.dumps(data['_adapter'], indent=2))


if __name__ == '__main__':
    main()
