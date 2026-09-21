from __future__ import annotations

from typing import Any, Iterable

import simpy

from .node import EventLogger, Link, Node, ObjectStore


class TrainingNode(Node):
    """
    Resource-level operations for one distributed-training rank.

    main.py owns orchestration: operation order, checkpoint frequency, retry
    policy, and failure injection. This class only exposes reusable simulated
    operations and resource behavior.
    """

    def __init__(
        self,
        env: simpy.Environment,
        *,
        rank: int,
        world_size: int,
        cpu_cores: int,
        gpu_count: int,
        gpu_memory_gb: float,
        dram_gb: float,
        ssd_gb: float,
        model_weights_gb: float,
        gradient_size_gb: float,
        checkpoint_size_gb: float,
        gpu_cpu_bandwidth_gbps: float,
        dram_read_gbps: float,
        dram_write_gbps: float,
        ssd_read_gbps: float,
        ssd_write_gbps: float,
        logger: EventLogger,
        object_store: ObjectStore,
        cpu_slowdown_when_gpu_active: float = 1.25,
        gpu_slowdown_when_cpu_active: float = 1.15,
        contention_quantum_seconds: float = 0.05,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            env,
            rank=rank,
            cpu_cores=cpu_cores,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            dram_gb=dram_gb,
            ssd_gb=ssd_gb,
            logger=logger,
        )

        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if model_weights_gb > gpu_memory_gb:
            raise ValueError(
                f"Model weights ({model_weights_gb:.2f} GB) exceed "
                f"GPU memory ({gpu_memory_gb:.2f} GB)"
            )

        positive_values = {
            "gradient_size_gb": gradient_size_gb,
            "checkpoint_size_gb": checkpoint_size_gb,
            "gpu_cpu_bandwidth_gbps": gpu_cpu_bandwidth_gbps,
            "dram_read_gbps": dram_read_gbps,
            "dram_write_gbps": dram_write_gbps,
            "ssd_read_gbps": ssd_read_gbps,
            "ssd_write_gbps": ssd_write_gbps,
            "contention_quantum_seconds": contention_quantum_seconds,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if cpu_slowdown_when_gpu_active < 1.0:
            raise ValueError("CPU slowdown multiplier must be at least 1.0")
        if gpu_slowdown_when_cpu_active < 1.0:
            raise ValueError("GPU slowdown multiplier must be at least 1.0")

        self.world_size = world_size
        self.model_weights_gb = float(model_weights_gb)
        self.gradient_size_gb = float(gradient_size_gb)
        self.checkpoint_size_gb = float(checkpoint_size_gb)

        self.gpu_cpu_bandwidth_gbps = float(gpu_cpu_bandwidth_gbps)
        self.dram_read_gbps = float(dram_read_gbps)
        self.dram_write_gbps = float(dram_write_gbps)
        self.ssd_read_gbps = float(ssd_read_gbps)
        self.ssd_write_gbps = float(ssd_write_gbps)

        self.cpu_slowdown_when_gpu_active = float(
            cpu_slowdown_when_gpu_active
        )
        self.gpu_slowdown_when_cpu_active = float(
            gpu_slowdown_when_cpu_active
        )
        self.contention_quantum_seconds = float(contention_quantum_seconds)

        self.object_store = object_store
        self.verbose = verbose
        self.peers: dict[int, dict[str, Any]] = {}
        self.represented_rank_count = 1

        # Only one rank-0 checkpoint may occupy the host-memory staging area.
        self.checkpoint_slot = simpy.Resource(env, capacity=1)
        self.loaded_iteration = 0

        # Counts include operations that have acquired the resource and are
        # currently doing work. They allow symmetric slowdown of already-running
        # CPU and GPU operations whenever a new overlapping operation begins.
        self._active_cpu_operations = 0
        self._active_gpu_operations = 0

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"{self.env.now:10.3f} | {self.name:<7} | {message}")

    @staticmethod
    def _bottleneck_time(data_gb: float, *bandwidths_gbps: float) -> float:
        if data_gb < 0:
            raise ValueError("data_gb cannot be negative")
        if not bandwidths_gbps or any(value <= 0 for value in bandwidths_gbps):
            raise ValueError("All bandwidths must be positive")
        return data_gb / min(bandwidths_gbps)

    def connect(self, other: "TrainingNode", link: Link) -> None:
        if other is self:
            raise ValueError("A node cannot connect to itself")
        if not link.connects(self, other):
            raise ValueError("The supplied link does not connect these nodes")

        self.peers[other.rank] = {"node": other, "link": link}
        other.peers[self.rank] = {"node": self, "link": link}

    def _acquire_resources(self, names: Iterable[str]):
        requests: list[tuple[str, simpy.Resource, simpy.events.Event]] = []
        for name in names:
            if name == "CPU":
                resource = self.cpu
            elif name == "GPU":
                resource = self.gpu
            else:
                raise ValueError(f"Unsupported resource: {name}")
            request = resource.request()
            requests.append((name, resource, request))

        if requests:
            yield self.env.all_of([request for _, _, request in requests])
        return requests

    @staticmethod
    def _release_resources(
        requests: list[tuple[str, simpy.Resource, simpy.events.Event]],
    ) -> None:
        for _, resource, request in reversed(requests):
            resource.release(request)

    def _run_contended_work(
        self,
        *,
        base_duration: float,
        resources: Iterable[str],
    ):
        """
        Execute base-duration work with dynamic CPU/GPU overlap penalties.

        The work is advanced in small simulated-time quanta. If an independent
        operation is active on the other resource, progress is reduced by the
        configured multiplier. This lets both sides of an overlap slow down,
        including an operation that was already running when contention began.
        """

        if base_duration < 0:
            raise ValueError("base_duration cannot be negative")

        resource_names = set(resources)
        uses_cpu = "CPU" in resource_names
        uses_gpu = "GPU" in resource_names

        if uses_cpu:
            self._active_cpu_operations += 1
        if uses_gpu:
            self._active_gpu_operations += 1

        remaining = float(base_duration)
        contention_seconds = 0.0
        cpu_contention_seconds = 0.0
        gpu_contention_seconds = 0.0

        try:
            # Give all processes starting at the same timestamp a chance to
            # register before the first contention decision.
            yield self.env.timeout(0)

            epsilon = 1e-12
            while remaining > epsilon:
                independent_gpu_active = (
                    self._active_gpu_operations - (1 if uses_gpu else 0)
                ) > 0
                independent_cpu_active = (
                    self._active_cpu_operations - (1 if uses_cpu else 0)
                ) > 0

                cpu_contended = uses_cpu and independent_gpu_active
                gpu_contended = uses_gpu and independent_cpu_active

                rates: list[float] = []
                if uses_cpu:
                    rates.append(
                        1.0 / self.cpu_slowdown_when_gpu_active
                        if cpu_contended
                        else 1.0
                    )
                if uses_gpu:
                    rates.append(
                        1.0 / self.gpu_slowdown_when_cpu_active
                        if gpu_contended
                        else 1.0
                    )

                # NETWORK-only work does not use this helper today, but unit
                # progress keeps the helper well-defined for that case.
                effective_rate = min(rates) if rates else 1.0
                wall_step = min(
                    self.contention_quantum_seconds,
                    remaining / effective_rate,
                )

                if cpu_contended or gpu_contended:
                    contention_seconds += wall_step
                if cpu_contended:
                    cpu_contention_seconds += wall_step
                if gpu_contended:
                    gpu_contention_seconds += wall_step

                yield self.env.timeout(wall_step)
                remaining = max(0.0, remaining - wall_step * effective_rate)
        finally:
            if uses_cpu:
                self._active_cpu_operations -= 1
            if uses_gpu:
                self._active_gpu_operations -= 1

        return {
            "base_duration": float(base_duration),
            "contention_seconds": contention_seconds,
            "cpu_contention_seconds": cpu_contention_seconds,
            "gpu_contention_seconds": gpu_contention_seconds,
            "cpu_slowdown_factor": self.cpu_slowdown_when_gpu_active,
            "gpu_slowdown_factor": self.gpu_slowdown_when_cpu_active,
        }

    def _occupy(
        self,
        *,
        duration: float,
        resources: Iterable[str],
        category: str,
        operation: str,
        iteration: int | None = None,
        source: str | None = None,
        destination: str | None = None,
        data_gb: float | None = None,
        failure_type: str | None = None,
        details: dict[str, Any] | None = None,
        started_event: simpy.Event | None = None,
    ):
        resource_names = list(resources)
        requests = yield from self._acquire_resources(resource_names)
        start = self.env.now
        if started_event is not None and not started_event.triggered:
            started_event.succeed(start)
        status = "completed"
        contention: dict[str, Any] = {}

        try:
            contention = yield from self._run_contended_work(
                base_duration=duration,
                resources=resource_names,
            )
        except simpy.Interrupt:
            status = "interrupted"
            raise
        finally:
            end = self.env.now
            actual_duration = float(end - start)
            merged_details = dict(details or {})
            merged_details.update(contention)
            merged_details.update(
                {
                    "status": status,
                    "actual_duration": actual_duration,
                    "effective_slowdown": (
                        actual_duration / duration if duration > 0 else 1.0
                    ),
                }
            )
            self.logger.record(
                start=start,
                end=end,
                rank=self.rank,
                node=self.name,
                category=category,
                operation=operation,
                resources=resource_names,
                iteration=iteration,
                source=source,
                destination=destination,
                data_gb=data_gb,
                failure_type=failure_type,
                details=merged_details,
            )
            self._release_resources(requests)

    # ------------------------------------------------------------------
    # Compute operations
    # ------------------------------------------------------------------

    def forward_pass(
        self,
        *,
        iteration: int,
        duration: float,
        started_event: simpy.Event | None = None,
    ):
        self.log(f"iteration {iteration}: forward")
        yield from self._occupy(
            duration=duration,
            resources=["GPU"],
            category="Compute",
            operation="forward_pass",
            iteration=iteration,
            started_event=started_event,
        )

    def backward_pass(
        self,
        *,
        iteration: int,
        duration: float,
        started_event: simpy.Event | None = None,
    ):
        self.log(f"iteration {iteration}: backward")
        yield from self._occupy(
            duration=duration,
            resources=["GPU"],
            category="Compute",
            operation="backward_pass",
            iteration=iteration,
            started_event=started_event,
        )

    def optimizer_step(self, *, iteration: int, duration: float):
        self.log(f"iteration {iteration}: optimizer")
        yield from self._occupy(
            duration=duration,
            resources=["GPU"],
            category="Compute",
            operation="optimizer_step",
            iteration=iteration,
        )
        self.loaded_iteration = iteration

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def transfer_gradient(
        self,
        *,
        destination_rank: int,
        iteration: int,
        phase: str,
        step: int,
        data_gb: float,
    ):
        if destination_rank not in self.peers:
            raise RuntimeError(
                f"{self.name} has no link to rank {destination_rank}"
            )

        link: Link = self.peers[destination_rank]["link"]
        gpu_request = self.gpu.request()
        link_request = link.channel.request()
        yield self.env.all_of([gpu_request, link_request])

        start = self.env.now
        base_duration = link.transfer_time(data_gb)
        contention: dict[str, Any] = {}
        try:
            contention = yield from self._run_contended_work(
                base_duration=base_duration,
                resources=["GPU"],
            )
        finally:
            end = self.env.now
            actual_duration = float(end - start)
            details = {
                "step": step,
                "link": link.name,
                **contention,
                "actual_duration": actual_duration,
                "effective_slowdown": (
                    actual_duration / base_duration if base_duration > 0 else 1.0
                ),
            }
            self.logger.record(
                start=start,
                end=end,
                rank=self.rank,
                node=self.name,
                category="Communication",
                operation=f"ring_all_reduce_{phase}",
                resources=["GPU", "NETWORK"],
                iteration=iteration,
                source=self.name,
                destination=f"rank-{destination_rank}",
                data_gb=data_gb,
                details=details,
            )
            link.channel.release(link_request)
            self.gpu.release(gpu_request)

    def aggregate_gradient_exchange(
        self,
        *,
        iteration: int,
        phase: str,
        duration: float,
        shard_gb: float,
        logical_world_size: int,
    ):
        """Model all synchronized ring steps as one interval per rank cohort."""

        gpu_request = self.gpu.request()
        yield gpu_request

        start = self.env.now
        contention: dict[str, Any] = {}
        try:
            contention = yield from self._run_contended_work(
                base_duration=duration,
                resources=["GPU"],
            )
        finally:
            end = self.env.now
            actual_duration = float(end - start)
            represented_ranks = self.represented_rank_count
            if self.name.startswith("ranks-"):
                link_name = "ring-remaining-workers-aggregated"
            else:
                link_name = f"ring-{self.name}-aggregated"
            self.logger.record(
                start=start,
                end=end,
                rank=self.rank,
                node=self.name,
                category="Communication",
                operation=f"ring_all_reduce_{phase}",
                resources=["GPU", "NETWORK"],
                iteration=iteration,
                source=self.name,
                destination="ring-next-rank",
                data_gb=shard_gb * (logical_world_size - 1),
                details={
                    "link": link_name,
                    "aggregate": True,
                    "represented_ranks": represented_ranks,
                    "steps_per_rank": logical_world_size - 1,
                    "transfers": (
                        represented_ranks * (logical_world_size - 1)
                    ),
                    **contention,
                    "actual_duration": actual_duration,
                    "effective_slowdown": (
                        actual_duration / duration if duration > 0 else 1.0
                    ),
                },
            )
            self.gpu.release(gpu_request)

    # ------------------------------------------------------------------
    # Asynchronous rank-0 checkpoint pipeline
    # ------------------------------------------------------------------

    def checkpoint_pipeline(
        self,
        *,
        iteration: int,
        staged_event: simpy.Event,
    ):
        """
        GPU -> host DRAM is synchronous with training.
        DRAM -> object store continues asynchronously.

        The checkpoint slot and its DRAM allocation remain held until the
        object-store write completes, so a later checkpoint cannot stage over
        an in-flight one.
        """

        slot_request = self.checkpoint_slot.request()
        memory_reserved = False
        yield slot_request

        try:
            if self.checkpoint_size_gb > self.dram.capacity:
                raise RuntimeError("Checkpoint is larger than node DRAM")

            yield self.dram.get(self.checkpoint_size_gb)
            memory_reserved = True

            stage_duration = self._bottleneck_time(
                self.checkpoint_size_gb,
                self.gpu_cpu_bandwidth_gbps,
                self.dram_write_gbps,
            )
            self.log(f"iteration {iteration}: checkpoint GPU -> DRAM")
            yield from self._occupy(
                duration=stage_duration,
                resources=["CPU", "GPU"],
                category="Checkpoint",
                operation="checkpoint_gpu_to_dram",
                iteration=iteration,
                data_gb=self.checkpoint_size_gb,
            )

            metadata = {
                "owner_rank": self.rank,
                "iteration": iteration,
                "size_gb": self.checkpoint_size_gb,
                "completed_at": None,
            }
            if not staged_event.triggered:
                staged_event.succeed(dict(metadata))

            io_request = self.object_store.io.request()
            cpu_request = self.cpu.request()
            yield self.env.all_of([io_request, cpu_request])

            upload_start = self.env.now
            base_upload_duration = self._bottleneck_time(
                self.checkpoint_size_gb,
                self.dram_read_gbps,
                self.object_store.bandwidth_gbps,
            )
            self.log(f"iteration {iteration}: checkpoint DRAM -> object store")
            contention: dict[str, Any] = {}

            try:
                contention = yield from self._run_contended_work(
                    base_duration=base_upload_duration,
                    resources=["CPU"],
                )
                metadata["completed_at"] = self.env.now
                self.object_store.put_checkpoint(self.rank, metadata)
            finally:
                upload_end = self.env.now
                actual_duration = float(upload_end - upload_start)
                self.logger.record(
                    start=upload_start,
                    end=upload_end,
                    rank=self.rank,
                    node=self.name,
                    category="Checkpoint",
                    operation="checkpoint_dram_to_object_store",
                    resources=["CPU", "NETWORK"],
                    iteration=iteration,
                    source=self.name,
                    destination="object-store",
                    data_gb=self.checkpoint_size_gb,
                    details={
                        "link": f"{self.name}-object-store",
                        **contention,
                        "actual_duration": actual_duration,
                        "effective_slowdown": (
                            actual_duration / base_upload_duration
                            if base_upload_duration > 0
                            else 1.0
                        ),
                    },
                )
                self.cpu.release(cpu_request)
                self.object_store.io.release(io_request)

        except BaseException as exc:
            if not staged_event.triggered:
                staged_event.fail(exc)
            raise
        finally:
            if memory_reserved:
                yield self.dram.put(self.checkpoint_size_gb)
            self.checkpoint_slot.release(slot_request)

    # ------------------------------------------------------------------
    # Failures and recovery
    # ------------------------------------------------------------------

    def simulate_failure(
        self,
        *,
        failure_type: str,
        restart_duration: float,
        iteration: int,
        event_rank: int | None = None,
        failure_stage: str | None = None,
        failure_progress: float | None = None,
    ):
        if failure_type not in {"process", "node", "spot"}:
            raise ValueError("failure_type must be process, node, or spot")

        requests = yield from self._acquire_resources(["CPU", "GPU"])
        start = self.env.now
        dram_flushed = 0.0
        ssd_flushed = 0.0
        contention: dict[str, Any] = {}

        try:
            if failure_type in {"node", "spot"}:
                dram_flushed = yield from self.flush_dram()
            if failure_type == "spot":
                ssd_flushed = yield from self.flush_ssd()

            self.log(
                f"iteration {iteration}: {failure_type} failure; restarting"
            )
            contention = yield from self._run_contended_work(
                base_duration=restart_duration,
                resources=["CPU", "GPU"],
            )
        finally:
            end = self.env.now
            actual_duration = float(end - start)
            self.logger.record(
                start=start,
                end=end,
                rank=self.rank if event_rank is None else event_rank,
                node=(
                    self.name
                    if event_rank is None
                    else f"rank-{event_rank}"
                ),
                category="Failure",
                operation=f"{failure_type}_failure_restart",
                resources=["CPU", "GPU"],
                iteration=iteration,
                failure_type=failure_type,
                details={
                    "dram_flushed_gb": dram_flushed,
                    "ssd_flushed_gb": ssd_flushed,
                    "failure_stage": failure_stage,
                    "failure_progress": failure_progress,
                    **contention,
                    "actual_duration": actual_duration,
                    "effective_slowdown": (
                        actual_duration / restart_duration
                        if restart_duration > 0
                        else 1.0
                    ),
                },
            )
            self._release_resources(requests)

    def recover_from_object_store(self, *, owner_rank: int = 0):
        checkpoint = self.object_store.latest_checkpoint(owner_rank)
        if checkpoint is None:
            raise RuntimeError("No completed object-store checkpoint exists")

        checkpoint_size = float(checkpoint["size_gb"])
        iteration = int(checkpoint["iteration"])
        if checkpoint_size > self.dram.capacity:
            raise RuntimeError("Recovery checkpoint is larger than node DRAM")

        yield self.dram.get(checkpoint_size)
        try:
            io_request = self.object_store.io.request()
            cpu_request = self.cpu.request()
            yield self.env.all_of([io_request, cpu_request])

            download_start = self.env.now
            base_download_duration = self._bottleneck_time(
                checkpoint_size,
                self.object_store.bandwidth_gbps,
                self.dram_write_gbps,
            )
            contention: dict[str, Any] = {}
            try:
                contention = yield from self._run_contended_work(
                    base_duration=base_download_duration,
                    resources=["CPU"],
                )
            finally:
                download_end = self.env.now
                actual_duration = float(download_end - download_start)
                self.logger.record(
                    start=download_start,
                    end=download_end,
                    rank=self.rank,
                    node=self.name,
                    category="Recovery",
                    operation="object_store_to_dram",
                    resources=["CPU", "NETWORK"],
                    iteration=iteration,
                    source="object-store",
                    destination=self.name,
                    data_gb=checkpoint_size,
                    details={
                        "link": f"{self.name}-object-store",
                        **contention,
                        "actual_duration": actual_duration,
                        "effective_slowdown": (
                            actual_duration / base_download_duration
                            if base_download_duration > 0
                            else 1.0
                        ),
                    },
                )
                self.cpu.release(cpu_request)
                self.object_store.io.release(io_request)

            restore_duration = self._bottleneck_time(
                checkpoint_size,
                self.dram_read_gbps,
                self.gpu_cpu_bandwidth_gbps,
            )
            yield from self._occupy(
                duration=restore_duration,
                resources=["CPU", "GPU"],
                category="Recovery",
                operation="dram_to_gpu_restore",
                iteration=iteration,
                data_gb=checkpoint_size,
            )
            self.loaded_iteration = iteration
        finally:
            yield self.dram.put(checkpoint_size)

        return iteration
