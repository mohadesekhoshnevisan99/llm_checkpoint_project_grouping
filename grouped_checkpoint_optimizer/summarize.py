"""Generate a compact reproducible record from saved experiment JSON files."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('RESULTS.md'))
    args = parser.parse_args()
    lines = ['# Synthetic experiment results', '',
             'These are scheduling-model checks, not simulator or hardware goodput results.', '']
    for path in args.files:
        data = json.loads(path.read_text(encoding='utf-8'))
        best = data['best']
        lines += [f'## {path.name}', '', f'- Jobs: {len(data["input"]["jobs"])}',
                  f'- Partitions evaluated: {len(data["candidates"])}',
                  f'- Elapsed seconds: {data.get("elapsed_s", "not recorded")}',
                  f'- Feasible candidates: {sum(bool(c.get("feasible")) for c in data["candidates"])}']
        if best:
            lines += [f'- Selected group sizes (size: count): {dict(sorted(Counter(map(len, best["groups"])).items()))}',
                      f'- Routes: {dict(Counter(p["route"] for p in best["policy"]))}',
                      f'- Period seconds (period: count): {dict(sorted(Counter(p["period_s"] for p in best["policy"]).items()))}',
                      f'- Status: {best["status"]}', f'- Weighted surrogate objective: {best["objective"]:.9f}',
                      f'- Fixed-partition catalogue gap bound: {best["relative_gap_bound"]:.6f}',
                      f'- Peak network GB/s: {best["peak_network_gbps"]:.6f}',
                      f'- Peak store GB/s: {best["peak_store_gbps"]:.6f}',
                      f'- Independent baseline: {best.get("independent_baseline", "not recorded")}', '']
        else:
            lines += ['- No feasible policy found.', '']
    lines += ['Partition search is heuristic. A zero inner gap does not certify the unrestricted grouping problem.', '']
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
