from __future__ import annotations

import pytest
import simpy

from simulation import SimPyBackend, Slowdown, advance_work


def test_simpy_backend_serializes_capacity_one_resource() -> None:
    env = simpy.Environment()
    backend = SimPyBackend(env)
    resource = backend.resource(capacity=1)
    intervals: list[tuple[str, float, float]] = []

    def use_resource(name: str):
        with resource.request() as request:
            yield request
            start = backend.now
            yield backend.timeout(2.0)
            intervals.append((name, start, backend.now))

    completion = backend.all_of(
        [
            backend.process(use_resource("first")),
            backend.process(use_resource("second")),
        ]
    )
    env.run(until=completion)

    assert intervals == [
        ("first", 0.0, 2.0),
        ("second", 2.0, 4.0),
    ]


def test_advance_work_tracks_named_dynamic_contention() -> None:
    env = simpy.Environment()
    backend = SimPyBackend(env)
    process = backend.process(
        advance_work(
            backend,
            base_duration=2.0,
            slowdown=lambda: Slowdown(
                factor=2.0,
                reasons=frozenset({"checkpoint_cpu"}),
            ),
            quantum=0.25,
        )
    )
    env.run(until=process)
    result = process.value

    assert result.completed
    assert result.wall_elapsed == pytest.approx(4.0)
    assert result.completed_fraction == 1.0
    assert result.effective_slowdown == pytest.approx(2.0)
    assert result.contention("checkpoint_cpu") == pytest.approx(4.0)
    assert result.contention("any") == pytest.approx(4.0)


def test_advance_work_reports_partial_progress_when_aborted() -> None:
    env = simpy.Environment()
    backend = SimPyBackend(env)
    state = {"aborted": False}

    def abort_later():
        yield backend.timeout(0.75)
        state["aborted"] = True

    backend.process(abort_later())
    process = backend.process(
        advance_work(
            backend,
            base_duration=2.0,
            aborted=lambda: state["aborted"],
            quantum=0.25,
        )
    )
    env.run(until=process)
    result = process.value

    assert not result.completed
    assert result.wall_elapsed == pytest.approx(0.75)
    assert result.completed_fraction == pytest.approx(0.375)


def test_backend_interrupts_waiting_process() -> None:
    env = simpy.Environment()
    backend = SimPyBackend(env)
    observed: list[tuple[str, float, str]] = []

    def worker():
        try:
            yield backend.timeout(10.0)
        except simpy.Interrupt as exc:
            observed.append(("interrupted", backend.now, exc.cause))
        return "recovered"

    process = backend.process(worker())

    def interrupt_later():
        yield backend.timeout(2.0)
        process.interrupt("failure")

    backend.process(interrupt_later())
    env.run(until=process)

    assert observed == [("interrupted", 2.0, "failure")]
    assert process.value == "recovered"
