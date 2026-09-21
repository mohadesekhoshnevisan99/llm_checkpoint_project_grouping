from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count
from types import ModuleType
from typing import Any, Callable, Iterable
import sys

from SerializableSimpy.core import Environment as _SerializableEnvironment


URGENT = 0
NORMAL = 1


class Interrupt(Exception):
    def __init__(self, cause: Any = None) -> None:
        super().__init__(cause)
        self.cause = cause


class _ScheduledCallback:
    __slots__ = ("time", "priority", "sequence", "func_ptr", "func_args")

    def __init__(
        self,
        time: float,
        priority: int,
        sequence: int,
        func_ptr: Callable[..., Any],
        func_args: tuple[Any, ...],
    ) -> None:
        self.time = time
        self.priority = priority
        self.sequence = sequence
        self.func_ptr = func_ptr
        self.func_args = func_args

    def __lt__(self, other: "_ScheduledCallback") -> bool:
        return (self.time, self.priority, self.sequence) < (
            other.time,
            other.priority,
            other.sequence,
        )


class Environment:
    def __init__(self, initial_time: float = 0) -> None:
        self._engine = _SerializableEnvironment(initial_time=initial_time)
        self._sequence = count()

    @property
    def now(self) -> float:
        return self._engine.get_now()

    @property
    def queue(self) -> list[Any]:
        return self._engine.queue

    def _schedule(
        self,
        delay: float,
        callback: Callable[..., Any],
        args: tuple[Any, ...] = (),
        *,
        priority: int = NORMAL,
    ) -> None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        heappush(
            self._engine.queue,
            _ScheduledCallback(
                self.now + float(delay),
                priority,
                next(self._sequence),
                callback,
                args,
            ),
        )

    def event(self) -> "Event":
        return Event(self)

    def timeout(self, delay: float, value: Any = None) -> "Timeout":
        return Timeout(self, delay, value)

    def process(self, generator: Any) -> "Process":
        return Process(self, generator)

    def all_of(self, events: Iterable["Event"]) -> "AllOf":
        return AllOf(self, list(events))

    def run(self, until: float | Event | None = None) -> Any:
        if until is None:
            while self._engine.queue:
                self._step()
            return None

        if isinstance(until, Event):
            while not until.triggered:
                if not self._engine.queue:
                    raise RuntimeError("No scheduled events left")
                self._step()
            if not until.ok:
                value = until.value
                if isinstance(value, BaseException):
                    raise value
                raise RuntimeError(value)
            return until.value

        horizon = float(until)
        if horizon < self.now:
            raise ValueError("until must be greater than or equal to now")
        while self._engine.queue and self._engine.queue[0].time < horizon:
            self._step()
        self._engine.set_now(horizon)
        return None

    def _step(self) -> None:
        callback = heappop(self._engine.queue)
        self._engine.set_now(callback.time)
        callback.func_ptr(*callback.func_args)


class Event:
    def __init__(self, env: Environment) -> None:
        self.env = env
        self.callbacks: list[Callable[["Event"], Any]] = []
        self._ok: bool | None = None
        self._value: Any = None

    @property
    def triggered(self) -> bool:
        return self._ok is not None

    @property
    def processed(self) -> bool:
        return self.triggered

    @property
    def ok(self) -> bool:
        if self._ok is None:
            raise AttributeError("event has not been triggered")
        return self._ok

    @property
    def value(self) -> Any:
        if self._ok is None:
            raise AttributeError("event has not been triggered")
        return self._value

    def add_callback(self, callback: Callable[["Event"], Any]) -> None:
        if self.triggered:
            self.env._schedule(0, callback, (self,))
        else:
            self.callbacks.append(callback)

    def succeed(self, value: Any = None) -> "Event":
        self._trigger(True, value)
        return self

    def fail(self, exception: BaseException | None = None) -> "Event":
        self._trigger(
            False,
            exception if exception is not None else RuntimeError("event failed"),
        )
        return self

    def _trigger(self, ok: bool, value: Any) -> None:
        if self.triggered:
            raise RuntimeError("event has already been triggered")
        self._ok = ok
        self._value = value
        callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            self.env._schedule(0, callback, (self,))

    def __and__(self, other: "Event") -> "AllOf":
        return AllOf(self.env, [self, other])

    def __or__(self, other: "Event") -> "AnyOf":
        return AnyOf(self.env, [self, other])


class Timeout(Event):
    def __init__(self, env: Environment, delay: float, value: Any = None) -> None:
        super().__init__(env)
        env._schedule(float(delay), self.succeed, (value,))


class Process(Event):
    def __init__(self, env: Environment, generator: Any) -> None:
        super().__init__(env)
        if not hasattr(generator, "send") or not hasattr(generator, "throw"):
            raise ValueError(f"{generator!r} is not a generator")
        self._generator = generator
        self._wait_token = 0
        self._pending_interrupt: Interrupt | None = None
        env._schedule(0, self._resume, (), priority=URGENT)

    @property
    def is_alive(self) -> bool:
        return not self.triggered

    def interrupt(self, cause: Any = None) -> None:
        if not self.is_alive:
            raise RuntimeError("Cannot interrupt a terminated process")
        self._wait_token += 1
        self._pending_interrupt = Interrupt(cause)
        self.env._schedule(0, self._resume, (), priority=URGENT)

    def _resume(self, event: Event | None = None, token: int | None = None) -> None:
        if self.triggered:
            return
        if token is not None and token != self._wait_token:
            return

        try:
            if self._pending_interrupt is not None:
                interrupt = self._pending_interrupt
                self._pending_interrupt = None
                yielded = self._generator.throw(interrupt)
            elif event is None:
                yielded = next(self._generator)
            elif event.ok:
                yielded = self._generator.send(event.value)
            else:
                value = event.value
                if isinstance(value, BaseException):
                    yielded = self._generator.throw(value)
                else:
                    yielded = self._generator.throw(RuntimeError(value))
        except StopIteration as stop:
            self.succeed(stop.value)
            return
        except BaseException as exc:
            self.fail(exc)
            return

        self._wait_for(yielded)

    def _wait_for(self, event: Any) -> None:
        if event is None:
            event = self.env.timeout(0)
        if not isinstance(event, Event):
            raise RuntimeError(f"Invalid yield value {event!r}")

        self._wait_token += 1
        token = self._wait_token

        def resume(triggered: Event) -> None:
            self._resume(triggered, token)

        event.add_callback(resume)


class Condition(Event):
    def __init__(self, env: Environment, events: list[Event]) -> None:
        super().__init__(env)
        self.events = events
        self._values: dict[Event, Any] = {}
        if not events:
            self.succeed({})
            return
        for event in events:
            event.add_callback(self._check)


class AllOf(Condition):
    def _check(self, event: Event) -> None:
        if self.triggered:
            return
        if not event.ok:
            self.fail(event.value)
            return
        self._values[event] = event.value
        if len(self._values) == len(self.events):
            self.succeed(dict(self._values))


class AnyOf(Condition):
    def _check(self, event: Event) -> None:
        if self.triggered:
            return
        if event.ok:
            self.succeed({event: event.value})
        else:
            self.fail(event.value)


class Request(Event):
    def __init__(self, resource: "Resource", priority: int = 0) -> None:
        super().__init__(resource.env)
        self.resource = resource
        self.priority = priority

    def __enter__(self) -> "Request":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.resource.release(self)


class Resource:
    def __init__(self, env: Environment, capacity: int = 1) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self.env = env
        self.capacity = int(capacity)
        self.users: list[Request] = []
        self.queue: deque[Request] = deque()

    def request(self, *args: Any, **kwargs: Any) -> Request:
        request = Request(self)
        self._enqueue(request)
        return request

    def release(self, request: Request) -> Event:
        if request in self.users:
            self.users.remove(request)
            self._trigger_queued()
        elif request in self.queue:
            self.queue.remove(request)
        return Event(self.env).succeed()

    def _enqueue(self, request: Request) -> None:
        if len(self.users) < self.capacity:
            self.users.append(request)
            request.succeed(request)
        else:
            self.queue.append(request)

    def _trigger_queued(self) -> None:
        while self.queue and len(self.users) < self.capacity:
            request = self.queue.popleft()
            self.users.append(request)
            request.succeed(request)


class PriorityResource(Resource):
    def __init__(self, env: Environment, capacity: int = 1) -> None:
        super().__init__(env, capacity=capacity)
        self.queue: list[tuple[int, int, Request]] = []
        self._request_sequence = count()

    def request(self, *args: Any, **kwargs: Any) -> Request:
        priority = int(kwargs.get("priority", 0))
        request = Request(self, priority=priority)
        self._enqueue(request)
        return request

    def release(self, request: Request) -> Event:
        if request in self.users:
            self.users.remove(request)
            self._trigger_queued()
        else:
            self.queue = [
                queued
                for queued in self.queue
                if queued[2] is not request
            ]
        return Event(self.env).succeed()

    def _enqueue(self, request: Request) -> None:
        if len(self.users) < self.capacity:
            self.users.append(request)
            request.succeed(request)
        else:
            heappush(
                self.queue,
                (request.priority, next(self._request_sequence), request),
            )

    def _trigger_queued(self) -> None:
        while self.queue and len(self.users) < self.capacity:
            _, _, request = heappop(self.queue)
            self.users.append(request)
            request.succeed(request)


class Container:
    def __init__(
        self,
        env: Environment,
        capacity: float = float("inf"),
        init: float = 0.0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        if init < 0 or init > capacity:
            raise ValueError("init must be between 0 and capacity")
        self.env = env
        self.capacity = float(capacity)
        self.level = float(init)
        self.put_queue: deque[tuple[float, Event]] = deque()
        self.get_queue: deque[tuple[float, Event]] = deque()

    def put(self, amount: float) -> Event:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        event = Event(self.env)
        self.put_queue.append((float(amount), event))
        self._trigger_queues()
        return event

    def get(self, amount: float) -> Event:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        event = Event(self.env)
        self.get_queue.append((float(amount), event))
        self._trigger_queues()
        return event

    def _trigger_queues(self) -> None:
        progressed = True
        while progressed:
            progressed = False
            for amount, event in list(self.get_queue):
                if self.level >= amount:
                    self.get_queue.remove((amount, event))
                    self.level -= amount
                    event.succeed(amount)
                    progressed = True
                    break
            for amount, event in list(self.put_queue):
                if self.level + amount <= self.capacity:
                    self.put_queue.remove((amount, event))
                    self.level += amount
                    event.succeed(amount)
                    progressed = True
                    break


events = ModuleType("simpy.events")
events.Event = Event
events.Timeout = Timeout
events.Process = Process
events.Condition = Condition
events.AllOf = AllOf
events.AnyOf = AnyOf
sys.modules[__name__ + ".events"] = events

resources = ModuleType("simpy.resources")
resource = ModuleType("simpy.resources.resource")
container = ModuleType("simpy.resources.container")
resource.Request = Request
resource.Resource = Resource
resource.PriorityResource = PriorityResource
container.Container = Container
resources.resource = resource
resources.container = container
sys.modules[__name__ + ".resources"] = resources
sys.modules[__name__ + ".resources.resource"] = resource
sys.modules[__name__ + ".resources.container"] = container


__all__ = [
    "AllOf",
    "AnyOf",
    "Condition",
    "Container",
    "Environment",
    "Event",
    "Interrupt",
    "NORMAL",
    "PriorityResource",
    "Process",
    "Request",
    "Resource",
    "Timeout",
    "URGENT",
]
