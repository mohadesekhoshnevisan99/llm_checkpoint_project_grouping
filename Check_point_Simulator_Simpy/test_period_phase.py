import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from optimize_period_phase import solve, templates_from_trace
from evaluate_durability import evaluate


class JointSearchTests(unittest.TestCase):
    def fixture(self):
        run = dict(row={}, contract=dict(epoch_origin_s=0, jobs={
            j: dict(route='direct', interval_s=30, phase_s=0, group_id=i)
            for i,j in enumerate(['a','b'])}),
            boundaries={j: [dict(time_s=t, iteration=t//10, eligible=t<120)
                            for t in range(10,121,10)] for j in ['a','b']})
        templates = {j: [dict(rank=r, offset_s=1, duration_s=1, size_gb=1) for r in [0,1]]
                     for j in ['a','b']}
        return run, templates

    def test_joint_budget_and_period_selection(self):
        run, templates = self.fixture()
        contract, report = solve(run, templates, [30,60], 10, 65, 30, 2, 500, 5)
        self.assertIsNotNone(contract)
        self.assertTrue(report['search_complete'])
        self.assertLessEqual(report['predicted_peak_gbps'], 2+1e-8)
        self.assertTrue(all(p['interval_s']==60 for p in contract['jobs'].values()))
        self.assertNotEqual(contract['jobs']['a']['phase_s'], contract['jobs']['b']['phase_s'])

    def test_unattainable_age_rejected(self):
        run, templates = self.fixture()
        contract, report = solve(run, templates, [30,60], 10, 1, 30, 4, 500, 5)
        self.assertIsNone(contract)
        self.assertEqual(report['status'], 'infeasible_catalogue')

    def test_budget_below_one_job_rejects_fixed_rank_profiles(self):
        run, templates = self.fixture()
        contract, report = solve(run, templates, [30,60], 10, 65, 30, 1, 500, 5)
        self.assertIsNone(contract)
        self.assertEqual(report['status'], 'infeasible_catalogue')

    def test_real_simulator_replay(self):
        import run_direct_contract as runner
        simulator = (Path.cwd() if (Path.cwd()/'run_scenario.py').exists() else
                     Path(__file__).resolve().parents[3]/'Check_point_Simulator_Simpy')
        if not (simulator/'run_scenario.py').exists():
            self.skipTest('Sibling simulator unavailable')
        sc = runner.yaml.safe_load((simulator/'scenarios/handcheck.yaml').read_text())
        sc.pop('preemption', None)
        initial = dict(schema='grouped-checkpoint-contract-v1', epoch_origin_s=0, horizon_s=120,
                       jobs={j: dict(group_id=i, route='direct', allowed_donor_job_ids=[],
                            donor_gb_per_version={}, interval_s=30, phase_s=27)
                       for i,j in enumerate(['alpha0','bravo0','be0'])})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'scenario.yaml').write_text(runner.yaml.safe_dump(sc))
            def simulate(name, contract):
                (root/f'{name}_contract.json').write_text(json.dumps(contract))
                with patch.object(sys, 'argv', ['run_direct_contract.py', '--simulator-dir',str(simulator),
                    '--scenario',str(root/'scenario.yaml'), '--contract',str(root/f'{name}_contract.json'),
                    '--out',str(root/f'{name}.json')]):
                    runner.main()
                return json.loads((root/f'{name}.json').read_text())
            before = simulate('before', initial)
            templates = templates_from_trace(before, root/'before.trace.jsonl.gz')
            candidate, report = solve(before, templates, [30,45,60,90,120],3,120,45,8,500,10)
            self.assertIsNotNone(candidate)
            self.assertEqual(len(templates['alpha0']),4)
            after = simulate('after', candidate)
            audit = evaluate(after, root/'after.trace.jsonl.gz',120,45,8,500)
            self.assertTrue(all(j['passes_observed_age_and_startup'] for j in audit['jobs'].values()))
            self.assertLess(after['row']['store_gb_written'], before['row']['store_gb_written'])
            self.assertLessEqual(audit['network']['peak_uniform_rate_proxy_gbps'],8+1e-7)
            self.assertFalse(audit['network']['instantaneous_capacity_verified'])


if __name__ == '__main__':
    unittest.main()
