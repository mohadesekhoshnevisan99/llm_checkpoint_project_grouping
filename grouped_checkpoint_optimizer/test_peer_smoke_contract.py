import unittest

from make_peer_smoke_contract import make_peer_contract


class PeerSmokeContractTests(unittest.TestCase):
    def test_donors_stay_inside_selected_group(self):
        source = {
            "schema": "grouped-checkpoint-contract-v1",
            "integration_ready": False,
            "epoch_origin_s": 0,
            "horizon_s": 300,
            "jobs": {
                "a": {"group_id": 0, "route": "direct", "allowed_donor_job_ids": [], "donor_gb_per_version": {}},
                "b": {"group_id": 1, "route": "direct", "allowed_donor_job_ids": [], "donor_gb_per_version": {}},
                "c": {"group_id": 2, "route": "direct", "allowed_donor_job_ids": [], "donor_gb_per_version": {}},
            },
        }
        result = make_peer_contract(source, ["a", "b"])
        self.assertEqual(result["jobs"]["a"]["route"], "direct")
        self.assertEqual(result["jobs"]["b"]["route"], "peer")
        self.assertEqual(result["jobs"]["b"]["allowed_donor_job_ids"], ["a"])
        self.assertEqual(result["jobs"]["a"]["group_id"], result["jobs"]["b"]["group_id"])
        self.assertNotEqual(result["jobs"]["c"]["group_id"], result["jobs"]["a"]["group_id"])


if __name__ == "__main__":
    unittest.main()
