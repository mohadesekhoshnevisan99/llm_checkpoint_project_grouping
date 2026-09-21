from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import simpy


@dataclass(slots=True)
class SimulationEvent:
    """One completed interval in the discrete-event simulation."""

    event_id: int
    start: float
    end: float
    duration: float
    rank: int | None
    node: str
    category: str
    operation: str
    run_id: int | None = None
    job_id: str | None = None
    pipeline_stage: int | None = None
    data_parallel_rank: int | None = None
    physical_node: str | None = None
    resources: list[str] = field(default_factory=list)
    iteration: int | None = None
    source: str | None = None
    destination: str | None = None
    data_gb: float | None = None
    failure_type: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class EventLogger:
    """Collects structured events and writes them as JSON Lines."""

    def __init__(self) -> None:
        self.events: list[SimulationEvent] = []
        self._next_event_id = 1

    def record(
        self,
        *,
        start: float,
        end: float,
        rank: int | None,
        node: str,
        category: str,
        operation: str,
        run_id: int | None = None,
        job_id: str | None = None,
        pipeline_stage: int | None = None,
        data_parallel_rank: int | None = None,
        physical_node: str | None = None,
        resources: Iterable[str] = (),
        iteration: int | None = None,
        source: str | None = None,
        destination: str | None = None,
        data_gb: float | None = None,
        failure_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = SimulationEvent(
            event_id=self._next_event_id,
            start=round(float(start), 9),
            end=round(float(end), 9),
            duration=round(float(end - start), 9),
            rank=rank,
            node=node,
            category=category,
            operation=operation,
            run_id=run_id,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            physical_node=physical_node,
            resources=list(resources),
            iteration=iteration,
            source=source,
            destination=destination,
            data_gb=(None if data_gb is None else round(float(data_gb), 9)),
            failure_type=failure_type,
            details=details or {},
        )
        self.events.append(event)
        self._next_event_id += 1
        self._event_recorded(event)

    def _event_recorded(self, event: SimulationEvent) -> None:
        """Extension hook for streaming/batched logger implementations."""

        del event

    def write_jsonl(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        ordered = sorted(
            self.events,
            key=lambda event: (event.start, event.end, event.event_id),
        )

        with output.open("w", encoding="utf-8") as handle:
            for event in ordered:
                handle.write(json.dumps(asdict(event), sort_keys=True))
                handle.write("\n")

        return output


class ObjectStore:
    """Shared remote checkpoint store with finite I/O concurrency."""

    def __init__(
        self,
        env: simpy.Environment,
        *,
        bandwidth_gbps: float,
        io_concurrency: int = 4,
    ) -> None:
        if bandwidth_gbps <= 0:
            raise ValueError("Object-store bandwidth must be positive")
        if io_concurrency <= 0:
            raise ValueError("Object-store I/O concurrency must be positive")

        self.env = env
        self.bandwidth_gbps = float(bandwidth_gbps)
        self.io = simpy.Resource(env, capacity=io_concurrency)
        self.checkpoints: dict[int, dict[str, Any]] = {}

    def put_checkpoint(self, owner_rank: int, metadata: dict[str, Any]) -> None:
        current = self.checkpoints.get(owner_rank)
        if current is not None and metadata["iteration"] < current["iteration"]:
            return
        self.checkpoints[owner_rank] = dict(metadata)

    def latest_checkpoint(self, owner_rank: int = 0) -> dict[str, Any] | None:
        checkpoint = self.checkpoints.get(owner_rank)
        return None if checkpoint is None else dict(checkpoint)


class Node:
    """Base simulated machine with CPU, GPU, host DRAM, and local SSD."""

    def __init__(
        self,
        env: simpy.Environment,
        *,
        rank: int,
        cpu_cores: int,
        gpu_count: int,
        gpu_memory_gb: float,
        dram_gb: float,
        ssd_gb: float,
        logger: EventLogger,
    ) -> None:
        if cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        if gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        for name, value in {
            "gpu_memory_gb": gpu_memory_gb,
            "dram_gb": dram_gb,
            "ssd_gb": ssd_gb,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.env = env
        self.rank = rank
        self.name = f"rank-{rank}"
        self.logger = logger

        self.cpu = simpy.Resource(env, capacity=cpu_cores)
        self.gpu = simpy.Resource(env, capacity=gpu_count)

        self.gpu_memory_gb = float(gpu_memory_gb)
        self.dram = simpy.Container(
            env,
            capacity=float(dram_gb),
            init=float(dram_gb),
        )
        self.ssd = simpy.Container(
            env,
            capacity=float(ssd_gb),
            init=float(ssd_gb),
        )

    def flush_dram(self):
        used = self.dram.capacity - self.dram.level
        if used > 0:
            yield self.dram.put(used)
        return used

    def flush_ssd(self):
        used = self.ssd.capacity - self.ssd.level
        if used > 0:
            yield self.ssd.put(used)
        return used


class Link:
    """A shared, full-bandwidth network link between exactly two nodes."""

    def __init__(
        self,
        env: simpy.Environment,
        *,
        node_a: Node,
        node_b: Node,
        bandwidth_gbps: float,
        name: str | None = None,
    ) -> None:
        if node_a is node_b:
            raise ValueError("A link must connect two different nodes")
        if bandwidth_gbps <= 0:
            raise ValueError("Link bandwidth must be positive")

        self.env = env
        self.node_a = node_a
        self.node_b = node_b
        self.bandwidth_gbps = float(bandwidth_gbps)
        self.name = name or f"{node_a.name}<->{node_b.name}"
        self.channel = simpy.Resource(env, capacity=1)

    def connects(self, node_a: Node, node_b: Node) -> bool:
        return {id(node_a), id(node_b)} == {
            id(self.node_a),
            id(self.node_b),
        }

    def other_end(self, node: Node) -> Node:
        if node is self.node_a:
            return self.node_b
        if node is self.node_b:
            return self.node_a
        raise ValueError(f"{node.name} is not connected to {self.name}")

    def transfer_time(self, data_gb: float) -> float:
        if data_gb < 0:
            raise ValueError("data_gb cannot be negative")
        return data_gb / self.bandwidth_gbps
