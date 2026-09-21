"""Constraint-first objective for grouped checkpoint policies."""

from __future__ import annotations


DEFAULT_WEIGHTS = {
    "simulation_time": 1.0,
    "fallback": 8.0,
    "own_flush": 1.0,
    "donor_failed": 2.0,
    "flush_duration": 0.5,
}


def is_eligible(
    row,
    *,
    min_peer_rate=0.25,
    require_loss_free=True,
    require_durable=True,
):
    """Return whether a runtime observation satisfies hard constraints."""
    if row.get("peer_rate", 0.0) < min_peer_rate:
        return False

    if require_loss_free and row.get("total_loss", 0) != 0:
        return False

    if require_durable and row.get("durable_frac", 0.0) < 1.0:
        return False

    return True


def score_observation(row, weights=None):
    weights = dict(DEFAULT_WEIGHTS) | dict(weights or {})

    return (
        weights["simulation_time"] * row["simulation_s"]
        + weights["fallback"] * row["fallback"]
        + weights["own_flush"] * row["own_flush"]
        + weights["donor_failed"] * row["donor_failed"]
        + weights["flush_duration"] * row["flush_duration_s"]
    )


def rank_observations(observations, weights=None):
    ranked = []

    for row in observations:
        ranked.append({
            **row,
            "score": score_observation(row, weights),
        })

    return sorted(ranked, key=lambda row: row["score"])


def select_policy(
    observations,
    weights=None,
    *,
    min_peer_rate=0.25,
    require_loss_free=True,
    require_durable=True,
):
    eligible = [
        row for row in observations
        if is_eligible(
            row,
            min_peer_rate=min_peer_rate,
            require_loss_free=require_loss_free,
            require_durable=require_durable,
        )
    ]

    if not eligible:
        raise ValueError("No policy satisfies the hard constraints")

    return rank_observations(eligible, weights)[0]
