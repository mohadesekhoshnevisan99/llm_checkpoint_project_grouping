from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .backend import SimulationBackend


@dataclass(frozen=True, slots=True)
class Slowdown:
    factor: float = 1.0
    reasons: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.factor < 1.0:
            raise ValueError("slowdown factor must be at least 1.0")


@dataclass(slots=True)
class ProgressResult:
    completed: bool
    base_duration: float
    wall_elapsed: float
    completed_fraction: float
    contention_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def effective_slowdown(self) -> float:
        if self.base_duration <= 0:
            return 1.0
        return self.wall_elapsed / self.base_duration

    def contention(self, reason: str) -> float:
        return self.contention_seconds.get(reason, 0.0)


def advance_work(
    backend: SimulationBackend,
    *,
    base_duration: float,
    slowdown: Callable[[], Slowdown] = Slowdown,
    aborted: Callable[[], bool] = lambda: False,
    quantum: float = 0.05,
):
    """Advance abstract work while contention and failure state may change."""

    if base_duration < 0:
        raise ValueError("base_duration must be nonnegative")
    if quantum <= 0:
        raise ValueError("quantum must be positive")

    remaining = base_duration
    wall_elapsed = 0.0
    contention: dict[str, float] = {}
    epsilon = 1e-12
    while remaining > epsilon:
        if aborted():
            return ProgressResult(
                completed=False,
                base_duration=base_duration,
                wall_elapsed=wall_elapsed,
                completed_fraction=(
                    1.0 - remaining / base_duration
                    if base_duration > 0
                    else 0.0
                ),
                contention_seconds=contention,
            )

        current = slowdown()
        rate = 1.0 / current.factor
        wall_step = min(quantum, remaining / rate)
        for reason in current.reasons:
            contention[reason] = contention.get(reason, 0.0) + wall_step
        if current.reasons:
            contention["any"] = contention.get("any", 0.0) + wall_step
        yield backend.timeout(wall_step)
        wall_elapsed += wall_step
        remaining = max(0.0, remaining - wall_step * rate)

    completed = not aborted()
    return ProgressResult(
        completed=completed,
        base_duration=base_duration,
        wall_elapsed=wall_elapsed,
        completed_fraction=1.0 if completed else 0.0,
        contention_seconds=contention,
    )


def combine_slowdowns(
    slowdowns: Iterable[tuple[float, str]],
) -> Slowdown:
    active = [
        (factor, reason) for factor, reason in slowdowns if factor > 1.0
    ]
    if not active:
        return Slowdown()
    return Slowdown(
        factor=max(factor for factor, _ in active),
        reasons=frozenset(reason for _, reason in active),
    )
