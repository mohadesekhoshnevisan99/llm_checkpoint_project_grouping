import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import run_budget_contract as budget


class BudgetTests(unittest.TestCase):
    def test_invalid_budget(self):
        for v in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                budget.BudgetMonitor(v,500)

    def test_integral_catches_missing_traffic(self):
        m=budget.BudgetMonitor(8,500)
        m.tagged_flows=1
        m.samples=[dict(time_s=0,network_gbps=8,store_gbps=8,active_streams=1),
                   dict(time_s=1,network_gbps=0,store_gbps=0,active_streams=0)]
        with self.assertRaises(AssertionError):
            m.report(12)
        self.assertTrue(m.report(8)['runtime_checkpoint_budget_verified'])

    def test_real_overload_is_throttled_for_network_and_store(self):
        simulator=(Path.cwd() if (Path.cwd()/'run_scenario.py').exists() else
                   Path(__file__).resolve().parents[3]/'Check_point_Simulator_Simpy')
        if not (simulator/'run_scenario.py').exists():
            self.skipTest('Simulator unavailable')
        sc=budget.runner.yaml.safe_load((simulator/'scenarios/handcheck.yaml').read_text())
        sc.pop('preemption',None)
        c=dict(schema='grouped-checkpoint-contract-v1',epoch_origin_s=0,
               jobs={j:dict(group_id=i,route='direct',allowed_donor_job_ids=[],
                    donor_gb_per_version={},interval_s=30,phase_s=27)
               for i,j in enumerate(['alpha0','bravo0','be0'])})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'sc.yaml').write_text(budget.runner.yaml.safe_dump(sc))
            (root/'contract.json').write_text(json.dumps(c))
            for net,store in [(8,500),(22.2,4)]:
                with self.subTest(network=net,store=store):
                    argv=['run_budget_contract.py','--simulator-dir',str(simulator),
                          '--scenario',str(root/'sc.yaml'),'--contract',str(root/'contract.json'),
                          '--out',str(root/'out.json'),'--network-gbps',str(net),'--store-gbps',str(store)]
                    with patch.object(sys,'argv',argv):
                        budget.main()
                    result=json.loads((root/'out.json').read_text())
                    a=result['runtime_budget_audit']
                    self.assertTrue(a['runtime_checkpoint_budget_verified'])
                    self.assertAlmostEqual(a['peak_allocated_network_gbps'],min(net,store))
                    self.assertAlmostEqual(a['integrated_network_gb'],120)
                    first=min(result['audit']['store_waves'],key=lambda w:w['start_s'])
                    self.assertAlmostEqual(first['end_s']-first['start_s'],12/min(net,store),places=6)


if __name__=='__main__':
    unittest.main()
