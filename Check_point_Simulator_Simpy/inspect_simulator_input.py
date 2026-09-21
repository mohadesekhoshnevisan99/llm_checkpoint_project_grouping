"""Extract handcheck dimensions without inventing calibration or runtime policy.

Run in the simulator's environment (requires PyYAML). This is an input audit,
NOT an optimizer input and NOT a policy accepted by run_scenario.py.
"""
import argparse
import json
from pathlib import Path
import yaml


def inspect(scenario, result):
    rows = result['rows']
    if len(rows) != 1:
        raise ValueError('Supply a result with one arm and one seed')
    if scenario.get('job_arrivals'):
        raise ValueError('This first adapter stage supports static jobs only')
    cluster = scenario['cluster']
    jobs = []
    ids = set()
    for arrival in rows[0]['arrival_manifest']:
        jid = arrival['job_id']
        if jid in ids or arrival['arrival_s'] != 0:
            raise ValueError('Expected unique, static initial job IDs')
        ids.add(jid)
        job = scenario['classes'][arrival['class']]
        if 'model' in job:
            raise ValueError('Model-based shard sizes require the existing parallelism calculator')
        ranks = int(job['ranks'])
        if ranks != int(arrival['nodes']):
            raise ValueError('Scenario/result node-count mismatch')
        size = float(job['checkpoint_gb_per_rank'])
        rates = job.get('rates', {})
        nic = float(rates.get('nic', cluster['network_bandwidth_gbps']))
        capture_bandwidth = float(cluster['gpu_cpu_bandwidth_gbps'])
        jobs.append(dict(
            id=jid, class_name=arrival['class'], nodes=ranks,
            checkpoint_gb_per_rank=size, checkpoint_gb_job=ranks * size,
            nominal_capture_s_per_rank=size / capture_bandwidth,
            nominal_interval_s=job['checkpoint_every'] * job['iteration_seconds'],
            nic_in_gbps_per_node=float(rates.get('nic_in', nic)),
            nic_out_gbps_per_node=float(rates.get('nic_out', nic)),
            ssd_gbps_per_node=float(rates.get('disk', cluster['local_ssd_bandwidth_gbps'])),
            configured_rpo_s=job.get('rpo_s'),
        ))
    return dict(
        schema='simulator-input-audit-v1', optimizer_ready=False,
        jobs=jobs, active_nodes=sum(j['nodes'] for j in jobs),
        configured_cluster_nodes=cluster['node_count'],
        store_config=scenario.get('store', {}), failures=scenario['failures'],
        preemption=scenario.get('preemption'),
        missing_model_choices=[
            'A shared fabric budget: per-node NIC rate is not the global network budget',
            'Donor SSD reservation capacity and mapping from shard slots to GB',
            'Effective store concurrency and stream cap for an entire multi-node job',
            'Measured foreground capture and asynchronous training slowdown',
            'Weights, period catalogue, time resolution and failure-tier mapping',
        ],
        limitations=[
            'Nominal interval excludes capture, flush, waiting and recovery',
            'Nominal capture assumes parallel ranks; it is not a measured duration',
            'Aggregate bytes do not establish fractional per-node donor feasibility',
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sc = yaml.safe_load(args.scenario.read_text(encoding='utf-8-sig'))
    result = json.loads(args.result.read_text(encoding='utf-8-sig'))
    audit = inspect(sc, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
