"""Evaluate global grouped checkpoint candidates with runtime constraints."""

from __future__ import annotations

from runtime_objective import is_eligible, score_observation


class RuntimeCandidateEvaluator:
    def __init__(
        self,
        *,
        min_peer_rate=0.25,
        require_loss_free=True,
        require_durable=True,
        weights=None,
    ):
        self.min_peer_rate = min_peer_rate
        self.require_loss_free = require_loss_free
        self.require_durable = require_durable
        self.weights = weights or {}

    def evaluate(self, candidate, observation):
        """Attach runtime metrics and return a scored candidate.

        `candidate` contains optimizer decisions such as groups, periods,
        phases and donors. `observation` contains simulator measurements.
        """
        feasible = is_eligible(
            observation,
            min_peer_rate=self.min_peer_rate,
            require_loss_free=self.require_loss_free,
            require_durable=self.require_durable,
        )

        result = {
            **candidate,
            "runtime": dict(observation),
            "runtime_feasible": feasible,
        }

        if not feasible:
            result["runtime_score"] = float("inf")
            result["rejection_reason"] = "runtime hard constraint failed"
            return result

        result["runtime_score"] = score_observation(
            observation,
            self.weights,
        )
        return result

    def select(self, candidates):
        feasible = [
            candidate
            for candidate in candidates
            if candidate.get("runtime_feasible", False)
        ]

        if not feasible:
            raise ValueError("No runtime-feasible candidate")

        return min(
            feasible,
            key=lambda candidate: candidate["runtime_score"],
        )
