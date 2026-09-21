from __future__ import annotations

from typing import Any, Protocol

import simpy


class SimulationEvent(Protocol):
    @property
    def triggered(self) -> bool: ...

    def succeed(self, value: Any = None) -> Any: ...


class SimulationResource(Protocol):
    def request(self, *args: Any, **kwargs: Any) -> Any: ...

    def release(self, request: Any) -> None: ...


class SimulationBackend(Protocol):
    """Minimal scheduler/resource contract used by simulation domain code."""

    @property
    def now(self) -> float: ...

    def timeout(self, duration: float) -> Any: ...

    def process(self, generator: Any) -> Any: ...

    def all_of(self, events: list[Any]) -> Any: ...

    def event(self) -> SimulationEvent: ...

    def resource(self, capacity: int = 1) -> SimulationResource: ...

    def priority_resource(
        self,
        capacity: int = 1,
    ) -> SimulationResource: ...


class SimPyBackend:
    """SimPy implementation of the engine-neutral scheduling contract."""

    def __init__(self, environment: simpy.Environment) -> None:
        self.environment = environment

    @property
    def now(self) -> float:
        return float(self.environment.now)

    def timeout(self, duration: float):
        return self.environment.timeout(duration)

    def process(self, generator: Any):
        return self.environment.process(generator)

    def all_of(self, events: list[Any]):
        return self.environment.all_of(events)

    def event(self):
        return self.environment.event()

    def resource(self, capacity: int = 1):
        return simpy.Resource(self.environment, capacity=capacity)

    def priority_resource(self, capacity: int = 1):
        return simpy.PriorityResource(self.environment, capacity=capacity)
