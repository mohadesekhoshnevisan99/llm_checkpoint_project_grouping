"""Transferable runtime penalty model calibrated from simulator observations."""

from __future__ import annotations


class RuntimePenaltyModel:
    def __init__(self, calibration):
        self.calibration = calibration
        self.weights = calibration.get("weights", {})

        observations = calibration.get("observations", [])
        if not observations:
            raise ValueError("Calibration has no observations")

        self.observations = observations

    def estimate(self, *, grouped, phased, group_size, kpeers):
        """Estimate runtime penalty for one optimizer candidate.

        This is a calibrated surrogate, not a simulator replacement.
        """
        if not grouped or kpeers <= 0:
            return {
                "penalty": 0.0,
                "estimated_peer_rate": 0.0,
                "estimated_fallback": 0.0,
                "estimated_own_flush": 0.0,
            }

        candidates = [
            row for row in self.observations
            if row.get("grouping") is True
        ]

        if phased:
            candidates = [
                row for row in candidates
                if row.get("phasing") is True
            ]

        if not candidates:
            candidates = self.observations

        peer_rate = sum(r["peer_rate"] for r in candidates) / len(candidates)
        fallback = sum(r["fallback"] for r in candidates) / len(candidates)
        own_flush = sum(r["own_flush"] for r in candidates) / len(candidates)
        flush_duration = sum(
            r["flush_duration_s"] for r in candidates
        ) / len(candidates)

        # Scale peer contention with group size and requested stripe width.
        scale = max(1.0, group_size / 3.0) * max(1.0, kpeers)

        estimated_peer_rate = max(0.0, min(1.0, peer_rate / scale))
        estimated_fallback = fallback * scale
        estimated_own_flush = own_flush * scale

        penalty = (
            self.weights.get("fallback", 8.0) * estimated_fallback
            + self.weights.get("own_flush", 1.0) * estimated_own_flush
            + self.weights.get("flush_duration", 0.5) * flush_duration
        )

        return {
            "penalty": penalty,
            "estimated_peer_rate": estimated_peer_rate,
            "estimated_fallback": estimated_fallback,
            "estimated_own_flush": estimated_own_flush,
        }
