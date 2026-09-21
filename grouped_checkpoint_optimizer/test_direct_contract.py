import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_direct_contract as runner


def contract():
    return dict(schema='grouped-checkpoint-contract-v1', epoch_origin_s=0,
                jobs={j: dict(group_id=i, route='direct', allowed_donor_job_ids=[],
                              donor_gb_per_version={}, interval_s=30, phase_s=27)
                      for i, j in enumerate(['alpha0', 'bravo0', 'be0'])})


class BoundaryTests(unittest.TestCase):
    def test_deadlines_and_busy_pipeline(self):
        scheduler = runner.BoundaryScheduler(contract())
        rt = SimpleNamespace(config=SimpleNamespace(job_id='alpha0'), it=2, iterations=24,
                             backend=SimpleNamespace(now=26),
                             checkpoint_strategy=SimpleNamespace(background_flushes=[]))
        self.assertFalse(scheduler.due(rt))
        rt.backend.now = 39
        self.assertTrue(scheduler.due(rt))
        self.assertEqual(scheduler.events[-1]['boundary_lag_s'], 12)
        rt.backend.now = 60
        rt.checkpoint_strategy.background_flushes = [SimpleNamespace(triggered=False)]
        self.assertFalse(scheduler.due(rt))
        rt.backend.now = 89
        rt.checkpoint_strategy.background_flushes = []
        self.assertTrue(scheduler.due(rt))
        self.assertEqual(scheduler.events[-1]['skipped_deadlines'], 1)

    def test_real_simulator_direct_route_and_capture_trace(self):
        simulator = (Path.cwd() if (Path.cwd() / 'run_scenario.py').exists() else
                     Path(__file__).resolve().parents[3] / 'Check_point_Simulator_Simpy')
        if not (simulator / 'run_scenario.py').exists():
            self.skipTest('Local sibling simulator unavailable; run CLI against your simulator')
        sc = runner.yaml.safe_load((simulator / 'scenarios/handcheck.yaml').read_text())
        sc.pop('preemption', None)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'scenario.yaml').write_text(runner.yaml.safe_dump(sc))
            (root / 'contract.json').write_text(json.dumps(contract()))
            argv = ['run_direct_contract.py', '--simulator-dir', str(simulator),
                    '--scenario', str(root / 'scenario.yaml'), '--contract', str(root / 'contract.json'),
                    '--out', str(root / 'result.json')]
            with patch.object(sys, 'argv', argv):
                runner.main()
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['row']['fallback_flushes'], 0)
            self.assertEqual(result['row']['peer_flushes'], 0)
            self.assertGreater(result['row']['store_gb_written'], 0)
            self.assertEqual(result['audit']['rank_captures'], len(result['decisions']) * 4)
            self.assertGreater(result['audit']['max_capture_lag_s'], 0)


if __name__ == '__main__':
    unittest.main()
