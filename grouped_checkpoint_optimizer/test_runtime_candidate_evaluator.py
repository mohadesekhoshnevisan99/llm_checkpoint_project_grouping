import unittest

from runtime_candidate_evaluator import RuntimeCandidateEvaluator


class RuntimeCandidateEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RuntimeCandidateEvaluator(
            min_peer_rate=0.25,
            require_loss_free=True,
            require_durable=True,
        )

        self.good_observation = {
            "peer_rate": 0.333,
            "fallback": 4,
            "own_flush": 10,
            "donor_failed": 4,
            "flush_duration_s": 33.76,
            "simulation_s": 231.86,
            "total_loss": 0,
            "durable_frac": 1.0,
        }

        self.bad_observation = {
            **self.good_observation,
            "peer_rate": 0.0,
        }

    def test_bad_candidate_is_rejected(self):
        result = self.evaluator.evaluate(
            {"name": "no_group"},
            self.bad_observation,
        )

        self.assertFalse(result["runtime_feasible"])
        self.assertEqual(result["runtime_score"], float("inf"))

    def test_good_candidate_is_scored(self):
        result = self.evaluator.evaluate(
            {"name": "group_phased"},
            self.good_observation,
        )

        self.assertTrue(result["runtime_feasible"])
        self.assertGreater(result["runtime_score"], 0)

    def test_select_ignores_infeasible_candidate(self):
        candidates = [
            self.evaluator.evaluate(
                {"name": "no_group"},
                self.bad_observation,
            ),
            self.evaluator.evaluate(
                {"name": "group_phased"},
                self.good_observation,
            ),
        ]

        selected = self.evaluator.select(candidates)
        self.assertEqual(selected["name"], "group_phased")


if __name__ == "__main__":
    unittest.main()
