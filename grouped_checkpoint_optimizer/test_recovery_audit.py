import unittest
from audit_recovery import audit


class RecoveryAuditTests(unittest.TestCase):
    def test_recovery_accounting(self):
        r = audit({'recovery_audit': True, 'row': dict(train_end_s=10, makespan_s=11,
            failures={'preemption': 1}, restores=2, recovery_source_tiers={'crossjob_peer': 2},
            total_loss=0, lost_iters_p50=1, lost_iters_max=2, durable_frac=1,
            peer_flushes=4, fallback_flushes=0, store_gb_written=8)})
        self.assertTrue(r['passes_basic_recovery_accounting'])

    def test_non_recovery_run_rejected(self):
        with self.assertRaises(ValueError):
            audit({'recovery_audit': False, 'row': {}})


if __name__ == '__main__':
    unittest.main()
