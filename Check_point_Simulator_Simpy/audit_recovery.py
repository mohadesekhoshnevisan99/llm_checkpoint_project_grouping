"""Summarize recovery outcomes from a direct-contract simulator run."""
import argparse
import json
from pathlib import Path


def audit(report):
    row = report['row']
    if not report.get('recovery_audit'):
        raise ValueError('Run with --allow-failures')
    failures = row.get('failures', {})
    restores = row.get('recovery_source_tiers', {})
    return dict(schema='recovery-audit-v1', train_end_s=row['train_end_s'],
                makespan_s=row['makespan_s'], failures=failures, restores=row.get('restores', 0),
                recovery_source_tiers=restores, total_loss=row.get('total_loss'),
                lost_iters_p50=row.get('lost_iters_p50'), lost_iters_max=row.get('lost_iters_max'),
                durable_frac=row.get('durable_frac'), peer_flushes=row.get('peer_flushes'),
                fallback_flushes=row.get('fallback_flushes'), store_gb_written=row.get('store_gb_written'),
                passes_basic_recovery_accounting=(row.get('restores', 0) == sum(restores.values())),
                limitations=['Recovery metrics are from one seed and one scenario',
                             'This runner still supports direct route only; peer contract is rejected',
                             'No confidence interval or expected-loss estimate'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    result = audit(json.loads(args.run.read_text(encoding='utf-8-sig')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
