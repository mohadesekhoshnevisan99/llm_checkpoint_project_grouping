import unittest
from build_handcheck_input import convert


class InputMappingTests(unittest.TestCase):
    def audit(self):
        return dict(schema='simulator-input-audit-v1', preemption=None,
                    failures={'per_node_per_second': 0}, active_nodes=12,
                    store_config={'in_gbps': 500, 'disk_gbps': 500},
                    jobs=[dict(id=j, nodes=4, checkpoint_gb_job=4,
                               nominal_capture_s_per_rank=.08, nic_in_gbps_per_node=1.85,
                               nic_out_gbps_per_node=1.85, ssd_gbps_per_node=.29,
                               configured_rpo_s=120) for j in ['alpha0', 'bravo0', 'be0']])

    def convert(self, a):
        return convert(a, network_gbps=22.2, free_ssd_gb_per_node=8,
                       store_stream_gbps=1, overlap=0)

    def test_rank_pool_units_and_explicit_assumptions(self):
        data = self.convert(self.audit())
        self.assertEqual(data['periods_s'], [30])
        self.assertEqual(data['jobs'][0]['size_gb'], 4)
        self.assertAlmostEqual(data['jobs'][0]['nic_gbps'], 7.4)
        self.assertAlmostEqual(data['jobs'][0]['donor_gbps'], 1.16)
        self.assertEqual(data['jobs'][0]['capture_s'], .08)
        self.assertEqual(data['jobs'][0]['free_ssd_gb'], 32)
        self.assertFalse(data['_adapter']['runtime_integration_ready'])

    def test_failure_and_count_mismatch_rejected(self):
        data = self.audit()
        data['preemption'] = {'every_s': 120}
        with self.assertRaises(ValueError):
            self.convert(data)

    def test_optimizer_accepts_mapped_input(self):
        from optimizer import optimize
        report = optimize(self.convert(self.audit()), max_group_size=3, max_phases=6, seconds=1)
        self.assertIsNotNone(report['best'])
        self.assertTrue(report['best']['feasible'])
        self.assertEqual({p['job_id'] for p in report['best']['policy']}, {'alpha0', 'bravo0', 'be0'})
        self.assertTrue(all(p['period_s'] == 30 for p in report['best']['policy']))
        data = self.audit()
        data['active_nodes'] = 100
        with self.assertRaises(ValueError):
            self.convert(data)


if __name__ == '__main__':
    unittest.main()
