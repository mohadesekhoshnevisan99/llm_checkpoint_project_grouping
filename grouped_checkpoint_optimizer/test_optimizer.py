import itertools
import unittest
import numpy as np
from optimizer import catalogue, demo, optimize, solve_partition, validate, verify


class SchedulerTests(unittest.TestCase):
    def small(self):
        data = demo(2)
        data.update(horizon_s=60, periods_s=[30, 60], slot_s=5)
        return data

    def test_milp_matches_exhaustive_catalogue(self):
        data = self.small()
        modes, by_job, caps = catalogue(data, [[0, 1]], 3)
        feasible = [sum(modes[m]['cost'] for m in selection)
                    for selection in itertools.product(*by_job)
                    if verify(modes, by_job, caps, selection)[0]]
        answer = solve_partition(data, [[0, 1]], 3, 5)
        self.assertTrue(answer['feasible'])
        self.assertAlmostEqual(answer['objective'], min(feasible), places=7)

    def test_global_store_couples_distinct_groups(self):
        data = self.small()
        for j in data['jobs']:
            j.update(size_gb=5, capture_s=0, failure_rate_s=.05)
        data.update(store_gbps=.25, store_stream_gbps=.25, headroom=1)
        answer = solve_partition(data, [[0], [1]], 12, 5)
        self.assertTrue(answer['feasible'])
        self.assertTrue(any(p['period_s'] == 60 for p in answer['policy']))
        self.assertLessEqual(answer['peak_store_gbps'], .25 + 1e-8)

    def test_wrap_around_consumes_early_slots(self):
        data = self.small()
        modes, _, _ = catalogue(data, [[0], [1]], 12)
        mode = next(m for m in modes if m['job'] == 0 and m['period_s'] == 60 and m['phase_s'] == 55)
        self.assertIn(0, mode['rows'])

    def test_peer_membership_and_reserved_storage(self):
        data = demo(4)
        groups = [[0, 1], [2, 3]]
        modes, _, _ = catalogue(data, groups, 2)
        peers = [m for m in modes if m['route'] == 'peer']
        self.assertTrue(peers)
        for m in peers:
            allowed = {data['jobs'][i]['id'] for i in groups[m['group']] if i != m['job']}
            self.assertEqual(set(m['donor_gb']), allowed)
        for j in data['jobs']:
            j['free_ssd_gb'] = 0
        modes, _, _ = catalogue(data, groups, 2)
        self.assertFalse(any(m['route'] == 'peer' for m in modes))

    def test_no_mode_is_infeasible_not_fake_policy(self):
        data = self.small()
        data['jobs'][0]['max_age_s'] = 1
        answer = solve_partition(data, [[0], [1]], 2, 0)
        self.assertEqual(answer['status'], 'infeasible_catalogue')
        self.assertNotIn('policy', answer)

    def test_invalid_partition_and_period_rejected(self):
        data = self.small()
        with self.assertRaises(ValueError):
            catalogue(data, [[0], [0, 1]])
        data['periods_s'] = [35]
        with self.assertRaises(ValueError):
            validate(data)

    def test_greedy_result_is_feasible(self):
        report = optimize(demo(6), max_group_size=3, max_phases=12, seconds=0)
        self.assertIsNotNone(report['best'])
        self.assertTrue(report['best']['feasible'])
        self.assertEqual(len(report['best']['policy']), 6)


if __name__ == '__main__':
    unittest.main()
