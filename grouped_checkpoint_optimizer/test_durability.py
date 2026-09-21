import unittest
import gzip
import json
from pathlib import Path
import tempfile
from evaluate_durability import age_metrics, traffic_metrics, evaluate


class DurabilityTests(unittest.TestCase):
    def test_missing_rank_is_not_a_complete_job_checkpoint(self):
        run = dict(schema='direct-boundary-integration-v1', row={'arrival_manifest': [{'job_id': 'j', 'nodes': 2}]},
                   decisions=[dict(job_id='j', iteration=1, boundary_s=10)],
                   boundaries={'j': [dict(time_s=20, eligible=False)]})
        captures = [dict(job_id='j', iteration=1, rank=r, operation='checkpoint_stage_gpu_to_dram',
                         data_gb=1) for r in [0, 1]]
        send = dict(job_id='j', iteration=1, rank=0, operation='checkpoint_stage_dram_to_object_store',
                    category='Checkpoint', details={'path': 'store'}, start=11, end=12, data_gb=1)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'trace.gz'
            with gzip.open(p, 'wt') as f:
                for e in captures + [send]:
                    f.write(json.dumps(e) + '\n')
            result = evaluate(run, p, 120, 45, 22.2, 500)
        self.assertEqual(result['jobs']['j']['complete_versions'], 0)
        self.assertEqual(len(result['incomplete_versions']), 1)
        self.assertFalse(result['jobs']['j']['passes_observed_age_and_startup'])

    def test_age_uses_ready_time_and_previous_snapshot(self):
        v = [{'snapshot_s': 0, 'ready_s': 5}, {'snapshot_s': 10, 'ready_s': 15}]
        r = age_metrics(v, 20, 12, 5)
        self.assertEqual(r['no_durable_version_s'], 5)
        self.assertEqual(r['max_age_after_first_durable_s'], 15)
        self.assertEqual(r['age_violation_s'], 3)
        self.assertFalse(r['passes_observed_age_and_startup'])

    def test_older_late_completion_does_not_replace_newer(self):
        v = [{'snapshot_s': 10, 'ready_s': 12}, {'snapshot_s': 0, 'ready_s': 15}]
        r = age_metrics(v, 20, 10, 12)
        self.assertEqual(r['max_age_after_first_durable_s'], 10)
        self.assertTrue(r['passes_observed_age_and_startup'])

    def test_no_complete_version_never_passes(self):
        self.assertFalse(age_metrics([], 20, 120, 30)['passes_observed_age_and_startup'])

    def test_shared_budget_lower_bound_and_touching_intervals(self):
        transfers = [dict(start=0, end=1, data_gb=1)] * 12
        result = traffic_metrics(transfers, 3)
        self.assertEqual(result['necessary_peak_lower_bound_gbps'], 12)
        self.assertEqual(result['budget_verdict'], 'infeasible_for_recorded_intervals')
        result = traffic_metrics([dict(start=0, end=1, data_gb=1), dict(start=1, end=2, data_gb=1)], 1)
        self.assertEqual(result['peak_uniform_rate_proxy_gbps'], 1)
        self.assertFalse(result['instantaneous_capacity_verified'])


if __name__ == '__main__':
    unittest.main()
