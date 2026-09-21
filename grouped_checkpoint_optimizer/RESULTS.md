# Synthetic experiment results

These are scheduling-model checks, not simulator or hardware goodput results.

## demo12.json

- Jobs: 12
- Partitions evaluated: 15
- Elapsed seconds: 4.76600000000326
- Feasible candidates: 13
- Selected group sizes (size: count): {3: 4}
- Routes: {'direct': 3, 'peer': 9}
- Period seconds (period: count): {120: 3, 240: 9}
- Status: optimal_within_solver_tolerance
- Weighted surrogate objective: 1.993032407
- Fixed-partition catalogue gap bound: 0.000000
- Peak network GB/s: 2.640000
- Peak store GB/s: 0.668571
- Independent baseline: {'objective': 1.9930324074074073, 'feasible': False, 'max_capacity_ratio': 3.0}

## demo300.json

- Jobs: 300
- Partitions evaluated: 15
- Elapsed seconds: 8.671999999991385
- Feasible candidates: 11
- Selected group sizes (size: count): {3: 100}
- Routes: {'direct': 74, 'peer': 226}
- Period seconds (period: count): {120: 45, 240: 255}
- Status: heuristic_feasible
- Weighted surrogate objective: 47.733217593
- Fixed-partition catalogue gap bound: 0.000000
- Peak network GB/s: 95.200000
- Peak store GB/s: 20.925714
- Independent baseline: {'objective': 47.733217592592624, 'feasible': False, 'max_capacity_ratio': 3.0}

Partition search is heuristic. A zero inner gap does not certify the unrestricted grouping problem.
