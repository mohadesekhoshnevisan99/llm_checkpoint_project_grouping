import unittest

from runtime_objective import (
    is_eligible,
    score_observation,
    select_policy,
)


class RuntimeObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "name": "no_group",
                "peer_rate": 0.0,
                "fallback": 8,
                "own_flush": 0,
                "donor_failed": 0,
                "flush_duration_s": 15.0,
                "simulation_s": 221.63,
                "total_loss": 0,
                "durable_frac": 1.0,
            },
            {
                "name": "group_phased",
                "peer_rate": 0.333,
                "fallback": 4,
                "own_flush": 10,
                "donor_failed": 4,
                "flush_duration_s": 33.76,
                "simulation_s": 231.86,
                "total_loss": 0,
                "durable_frac": 1.0,
            },
        ]

    def test_no_group_fails_peer_constraint(self):
        self.assertFalse(
            is_eligible(
                self.rows[0],
                min_peer_rate=0.25,
                require_loss_free=True,
                require_durable=True,
            )
        )

    def test_grouped_policy_is_eligible(self):
        self.assertTrue(
            is_eligible(
                self.rows[1],
                min_peer_rate=0.25,
                require_loss_free=True,
                require_durable=True,
            )
        )

    def test_selector_returns_grouped_policy(self):
        selected = select_policy(
            self.rows,
            min_peer_rate=0.25,
            require_loss_free=True,
            require_durable=True,
        )
        self.assertEqual(selected["name"], "group_phased")


if __name__ == "__main__":
    unittest.main()
