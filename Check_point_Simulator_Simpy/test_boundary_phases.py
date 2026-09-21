import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from optimize_boundary_phases import overlap_metrics, project, optimize, compare


class ProjectionTests(unittest.TestCase):
    def test_half_open_and_pairwise_overlap(self):
        self.assertEqual(overlap_metrics([{'start_s': 0, 'end_s': 2},
                                         {'start_s': 2, 'end_s': 4}])['pair_overlap_s'], 0)
        self.assertEqual(overlap_metrics([{'start_s': 0, 'end_s': 2}] * 3)['pair_overlap_s'], 6)

    def test_busy_gating_and_deadline_coalescing(self):
        b = [dict(time_s=t, iteration=i, eligible=True) for i, t in enumerate([10,20,30,40,50])]
        result = project(b, 10, 0, 0, {'capture_to_store_s': 1, 'transfer_s': 24})
        self.assertEqual([d['boundary_s'] for d in result['decisions']], [10,40])
        self.assertEqual([d['skipped'] for d in result['decisions']], [1,2])

    def test_missing_boundaries_rejected(self):
        with self.assertRaises(ValueError):
            optimize({'contract': {}, 'audit': {}})

    def test_replay_against_real_simulator(self):
        import run_direct_contract as runner
        from test_direct_contract import contract
        simulator = (Path.cwd() if (Path.cwd() / 'run_scenario.py').exists() else
                     Path(__file__).resolve().parents[3] / 'Check_point_Simulator_Simpy')
        if not (simulator / 'run_scenario.py').exists():
            self.skipTest('Sibling simulator unavailable')
        root = Path(__file__).parent / 'outputs' / 'boundary_validation'
        root.mkdir(parents=True, exist_ok=True)
        sc = runner.yaml.safe_load((simulator / 'scenarios/handcheck.yaml').read_text())
        sc.pop('preemption', None)
        (root / 'scenario.yaml').write_text(runner.yaml.safe_dump(sc))
        (root / 'before_contract.json').write_text(json.dumps(contract()))
        def run(name):
            with patch.object(sys, 'argv', ['run_direct_contract.py', '--simulator-dir', str(simulator),
                     '--scenario', str(root / 'scenario.yaml'), '--contract', str(root / f'{name}_contract.json'),
                     '--out', str(root / f'{name}.json')]):
                runner.main()
            return json.loads((root / f'{name}.json').read_text())
        before = run('before')
        candidate, report = optimize(before)
        self.assertLessEqual(tuple(report['candidate_score']), tuple(report['baseline_score']))
        (root / 'after_contract.json').write_text(json.dumps(candidate, indent=2))
        (root / 'search.json').write_text(json.dumps(report, indent=2))
        after = run('after')
        comparison = compare(before, after)
        (root / 'comparison.json').write_text(json.dumps(comparison, indent=2))
        self.assertEqual(set(after['contract']['jobs']), set(before['contract']['jobs']))
        self.assertFalse(after['optimizer_calendar_verified'])
        self.assertEqual(after['row']['fallback_flushes'], 0)
        self.assertTrue(comparison['passes_this_limited_replay_check'])
        print(json.dumps({'search': report, 'replay': comparison}, indent=2))


if __name__ == '__main__':
    unittest.main()
