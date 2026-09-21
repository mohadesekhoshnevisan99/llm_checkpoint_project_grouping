import copy
import unittest
from policy_bridge import donor_allowed, export_contract, next_capture_time


class BridgeTests(unittest.TestCase):
    def report(self):
        return {'input': {'horizon_s': 120}, 'best': {'feasible': True,
            'groups': [['a', 'b'], ['c']], 'policy': [
                {'job_id': 'a', 'group': 0, 'route': 'peer', 'period_s': 60, 'phase_s': 10, 'donor_gb': {'b': 4}},
                {'job_id': 'b', 'group': 0, 'route': 'direct', 'period_s': 60, 'phase_s': 30, 'donor_gb': {}},
                {'job_id': 'c', 'group': 1, 'route': 'direct', 'period_s': 120, 'phase_s': 0, 'donor_gb': {}}]}}

    def test_membership_and_direct_route(self):
        c = export_contract(self.report())
        self.assertTrue(donor_allowed(c, 'a', 'b'))
        for pair in [('a', 'a'), ('a', 'c'), ('b', 'a'), ('unknown', 'b')]:
            self.assertFalse(donor_allowed(c, *pair))
        self.assertFalse(c['integration_ready'])

    def test_time_boundary_epoch_and_missed_slots(self):
        c = export_contract(self.report(), epoch_origin_s=100)
        self.assertEqual(next_capture_time(c, 'a', 0), 110)
        self.assertEqual(next_capture_time(c, 'a', 110), 170)
        self.assertEqual(next_capture_time(c, 'a', 180), 230)

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            export_contract({'best': None})
        r = copy.deepcopy(self.report())
        r['best']['policy'][0]['donor_gb'] = {'c': 4}
        with self.assertRaises(ValueError):
            export_contract(r)


if __name__ == '__main__':
    unittest.main()
