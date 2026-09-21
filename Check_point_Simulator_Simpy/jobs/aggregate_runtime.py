from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any

from nodes.node import EventLogger
from simulation.backend import SimulationBackend
from simulation.progress import Slowdown, advance_work

from .config import JobConfig, SimulatorConfig


@dataclass(frozen=True, slots=True)
class AggregatePartition:
    index: int
    rank_start: int
    rank_end: int

    @property
    def represented_rank_count(self) -> int:
        return self.rank_end - self.rank_start + 1


class AggregatedDataParallelRuntime:
    """Analytical cohort runtime for very large, mostly identical DP jobs."""

    PARTITION_SIZE = 10_000
    SUPPORTED_CHECKPOINT_STRATEGIES = {"object_store", "ssd_tiered"}

    def __init__(
        self,
        *,
        run_id: int,
        config: JobConfig,
        simulator_config: SimulatorConfig,
        logger: EventLogger,
        node_offset: int,
        object_store: Any | None = None,
        verbose: bool,
        backend: SimulationBackend,
    ) -> None:
        if config.strategy != "data_parallel":
            raise ValueError("Cohort execution currently supports data parallelism")
        if config.checkpoint_strategy not in self.SUPPORTED_CHECKPOINT_STRATEGIES:
            supported = ", ".join(sorted(self.SUPPORTED_CHECKPOINT_STRATEGIES))
            raise ValueError(
                "Cohort execution currently supports checkpoint strategies: "
                f"{supported}"
            )
        self.backend = backend
        self.run_id = run_id
        self.config = config
        self.cluster = simulator_config.cluster
        self.failure_settings = simulator_config.failures
        self.iterations = simulator_config.simulation.iterations
        self.logger = logger
        self.node_offset = node_offset
        self.verbose = verbose
        self.rank_count = config.rank_count
        self.worker_cores = self._available_worker_cores()
        self.partitions = self._build_partitions()
        self.partition_count = len(self.partitions)
        self.cohort_name = self._partition_name(self.partitions[0])
        self.physical_nodes = self._partition_physical_nodes(
            self.partitions[0]
        )
        self.object_store = object_store or backend.resource(
            self.cluster.object_store_concurrency
        )
        self.cpu = backend.resource()
        self.gpu = backend.resource()
        self.checkpoint_processes: list[Any] = []
        self.checkpoint_cpu_tasks = 0
        self.checkpoint_network_tasks = 0
        self.training_network_tasks = 0
        self.current_iteration = 0
        self.active = True
        self.deviant_ranks: set[int] = set()
        seed = simulator_config.simulation.seed + run_id + sum(
            ord(character) for character in config.job_id
        )
        self.rng = random.Random(seed)

    @classmethod
    def _available_worker_cores(cls) -> int:
        configured = os.environ.get("SIMULATOR_WORKER_CORES")
        if configured is not None:
            try:
                return max(1, int(configured))
            except ValueError:
                return 1
        process_cpu_count = getattr(os, "process_cpu_count", None)
        if process_cpu_count is not None:
            return max(1, int(process_cpu_count() or 1))
        return max(1, int(os.cpu_count() or 1))

    def _build_partitions(self) -> list[AggregatePartition]:
        return [
            AggregatePartition(
                index=index,
                rank_start=start,
                rank_end=min(
                    start + self.PARTITION_SIZE - 1,
                    self.rank_count - 1,
                ),
            )
            for index, start in enumerate(
                range(0, self.rank_count, self.PARTITION_SIZE)
            )
        ]

    def _partition_name(self, partition: AggregatePartition) -> str:
        return f"ranks-{partition.rank_start}-{partition.rank_end}"

    def _partition_physical_nodes(self, partition: AggregatePartition) -> str:
        return (
            f"nodes-{self.node_offset + partition.rank_start}-"
            f"{self.node_offset + partition.rank_end}"
        )

    def placement(self) -> list[dict[str, Any]]:
        checkpoint_partner_rule = (
            "rank XOR 1"
            if self.config.checkpoint_strategy == "ssd_tiered"
            else None
        )
        return [
            {
                "job_id": self.config.job_id,
                "rank_start": partition.rank_start,
                "rank_end": partition.rank_end,
                "represented_rank_count": partition.represented_rank_count,
                "data_parallel_rank": (
                    f"{partition.rank_start}-{partition.rank_end}"
                ),
                "pipeline_stage": 0,
                "physical_node": self._partition_physical_nodes(partition),
                "checkpoint_partner_rule": checkpoint_partner_rule,
                "aggregate": True,
                "aggregate_partition_index": partition.index,
                "aggregate_partitions": self.partition_count,
                "scheduler_worker_cores": self.worker_cores,
            }
            for partition in self.partitions
        ]

    def _record(
        self,
        *,
        partition: AggregatePartition,
        start: float,
        operation: str,
        category: str,
        resources: list[str],
        iteration: int,
        data_gb: float | None = None,
        source: str | None = None,
        destination: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        node_name = self._partition_name(partition)
        aggregate_details = {
            "aggregate": True,
            "represented_ranks": partition.represented_rank_count,
            "aggregate_rank_start": partition.rank_start,
            "aggregate_rank_end": partition.rank_end,
            "aggregate_partition_index": partition.index,
            "aggregate_partitions": self.partition_count,
            "scheduler_worker_cores": self.worker_cores,
            "scheduler_batch": partition.index // self.worker_cores,
            **(details or {}),
        }
        self.logger.record(
            start=start,
            end=self.backend.now,
            run_id=self.run_id,
            job_id=self.config.job_id,
            rank=partition.rank_start,
            node=node_name,
            physical_node=self._partition_physical_nodes(partition),
            pipeline_stage=0,
            data_parallel_rank=partition.rank_start,
            category=category,
            operation=operation,
            resources=resources,
            iteration=iteration,
            source=node_name if source == self.cohort_name else source,
            destination=destination,
            data_gb=data_gb,
            details=aggregate_details,
        )

    def _gpu_work(
        self,
        *,
        duration: float,
        operation: str,
        iteration: int,
    ):
        with self.gpu.request() as request:
            yield request
            start = self.backend.now
            result = yield from advance_work(
                self.backend,
                base_duration=duration,
                slowdown=lambda: Slowdown(
                    factor=(
                        self.config.checkpoint_upload_gpu_slowdown
                        if self.checkpoint_cpu_tasks > 0
                        else 1.0
                    ),
                    reasons=(
                        frozenset({"checkpoint_cpu"})
                        if self.checkpoint_cpu_tasks > 0
                        else frozenset()
                    ),
                ),
            )
            contention_seconds = result.contention("checkpoint_cpu")
        for partition in self.partitions:
            self._record(
                partition=partition,
                start=start,
                operation=operation,
                category="Compute",
                resources=["GPU"],
                iteration=iteration,
                details={
                    "status": "completed",
                    "attempt": 1,
                    "base_duration": duration,
                    "completed_fraction": 1.0,
                    "contention_seconds": contention_seconds,
                    "checkpoint_upload_contention_seconds": contention_seconds,
                    "effective_slowdown": result.effective_slowdown,
                },
            )

    def _all_reduce(self, *, iteration: int):
        replicas = self.config.data_parallel_replicas
        duration = (
            2
            * (replicas - 1)
            / replicas
            * self.config.gradient_gb
            / self.cluster.network_bandwidth_gbps
        )
        with self.gpu.request() as request:
            yield request
            start = self.backend.now
            self.training_network_tasks += 1
            try:
                result = yield from advance_work(
                    self.backend,
                    base_duration=duration,
                    slowdown=lambda: Slowdown(
                        factor=1.0 + self.checkpoint_network_tasks,
                        reasons=(
                            frozenset({"checkpoint_network"})
                            if self.checkpoint_network_tasks > 0
                            else frozenset()
                        ),
                    ),
                )
            finally:
                self.training_network_tasks -= 1
            contention_seconds = result.contention("checkpoint_network")
        collective_group = f"ranks-0-{self.rank_count - 1}"
        for partition in self.partitions:
            self._record(
                partition=partition,
                start=start,
                operation="data_parallel_all_reduce",
                category="Communication",
                resources=["GPU", "NETWORK"],
                iteration=iteration,
                data_gb=self.config.gradient_gb,
                source=self._partition_name(partition),
                destination=f"{self.config.job_id}-all-ranks",
                details={
                    "link": (
                        f"{self.config.job_id}/aggregate-all-reduce/"
                        f"{self._partition_name(partition)}"
                    ),
                    "collective_group": collective_group,
                    "base_duration": duration,
                    "contention_seconds": contention_seconds,
                    "bandwidth_contention_seconds": contention_seconds,
                    "checkpoint_upload_contention_seconds": (
                        contention_seconds
                    ),
                    "effective_slowdown": result.effective_slowdown,
                },
            )

    def _checkpoint(self, *, iteration: int):
        if self.config.checkpoint_strategy == "object_store":
            yield from self._object_store_checkpoint(iteration=iteration)
            return
        yield from self._ssd_tiered_checkpoint(iteration=iteration)

    def _object_store_checkpoint(self, *, iteration: int):
        checkpoint_gb = self.config.checkpoint_gb
        checkpoint_group = f"{self.config.job_id}-iteration-{iteration}"
        writer_partition = self.partitions[0]
        gpu_request = self.gpu.request()
        cpu_request = self.cpu.request()
        yield self.backend.all_of([gpu_request, cpu_request])
        start = self.backend.now
        stage_duration = checkpoint_gb / self.cluster.gpu_cpu_bandwidth_gbps
        yield self.backend.timeout(stage_duration)
        self._record(
            partition=writer_partition,
            start=start,
            operation="checkpoint_stage_gpu_to_dram",
            category="Checkpoint",
            resources=["CPU", "GPU"],
            iteration=iteration,
            data_gb=checkpoint_gb,
            details={
                "checkpoint_strategy": "object_store",
                "checkpoint_group": checkpoint_group,
                "shard": 0,
                "shard_count": 1,
                "base_duration": stage_duration,
                "effective_slowdown": 1.0,
            },
        )
        self.gpu.release(gpu_request)
        self.cpu.release(cpu_request)

        with self.object_store.request() as store_request:
            yield store_request
            cpu_request = self.cpu.request()
            yield cpu_request
            self.checkpoint_cpu_tasks += 1
            try:
                start = self.backend.now
                upload_duration = (
                    checkpoint_gb / self.cluster.object_store_bandwidth_gbps
                )
                yield self.backend.timeout(upload_duration)
                self._record(
                    partition=writer_partition,
                    start=start,
                    operation="checkpoint_stage_dram_to_object_store",
                    category="Checkpoint",
                    resources=["CPU", "NETWORK"],
                    iteration=iteration,
                    data_gb=checkpoint_gb,
                    source=f"{self._partition_name(writer_partition)}/dram",
                    destination="object-store",
                    details={
                        "link": f"{self.config.job_id}/aggregate-object-store",
                        "checkpoint_strategy": "object_store",
                        "checkpoint_group": checkpoint_group,
                        "shard": 0,
                        "shard_count": 1,
                        "base_duration": upload_duration,
                        "effective_slowdown": 1.0,
                        "overlap_gpu_slowdown": (
                            self.config.checkpoint_upload_gpu_slowdown
                        ),
                    },
                )
            finally:
                self.checkpoint_cpu_tasks -= 1
                self.cpu.release(cpu_request)

    def _ssd_tiered_checkpoint(self, *, iteration: int):
        checkpoint_gb = self.config.checkpoint_gb
        gpu_request = self.gpu.request()
        cpu_request = self.cpu.request()
        yield self.backend.all_of([gpu_request, cpu_request])
        start = self.backend.now
        stage_duration = checkpoint_gb / self.cluster.gpu_cpu_bandwidth_gbps
        yield self.backend.timeout(stage_duration)
        for partition in self.partitions:
            self._record(
                partition=partition,
                start=start,
                operation="checkpoint_stage_gpu_to_dram",
                category="Checkpoint",
                resources=["CPU", "GPU"],
                iteration=iteration,
                data_gb=checkpoint_gb,
                details={
                    "checkpoint_strategy": "ssd_tiered",
                    "checkpoint_group": (
                        f"{self.config.job_id}-iteration-{iteration}"
                    ),
                    "base_duration": stage_duration,
                    "effective_slowdown": 1.0,
                },
            )
        self.gpu.release(gpu_request)
        self.cpu.release(cpu_request)

        with self.cpu.request() as request:
            yield request
            self.checkpoint_cpu_tasks += 1
            try:
                start = self.backend.now
                local_duration = (
                    checkpoint_gb / self.cluster.local_ssd_bandwidth_gbps
                )
                yield self.backend.timeout(local_duration)
                for partition in self.partitions:
                    self._record(
                        partition=partition,
                        start=start,
                        operation="checkpoint_dram_to_local_ssd_chunk",
                        category="Checkpoint",
                        resources=["CPU"],
                        iteration=iteration,
                        data_gb=checkpoint_gb,
                        details={
                            "checkpoint_strategy": "ssd_tiered",
                            "chunk_index": 1,
                            "chunk_count": 1,
                            "base_duration": local_duration,
                            "effective_slowdown": 1.0,
                        },
                    )

                start = self.backend.now
                peer_data_gb = checkpoint_gb * 2
                base_duration = (
                    peer_data_gb / self.cluster.network_bandwidth_gbps
                )
                self.checkpoint_network_tasks += 1
                try:
                    result = yield from advance_work(
                        self.backend,
                        base_duration=base_duration,
                        slowdown=lambda: Slowdown(
                            factor=1.0 + self.training_network_tasks,
                            reasons=(
                                frozenset({"training_network"})
                                if self.training_network_tasks > 0
                                else frozenset()
                            ),
                        ),
                    )
                finally:
                    self.checkpoint_network_tasks -= 1
                contention_seconds = result.contention("training_network")
                for partition in self.partitions:
                    self._record(
                        partition=partition,
                        start=start,
                        operation="checkpoint_dram_to_peer_dram_chunk",
                        category="Checkpoint",
                        resources=["CPU", "NETWORK"],
                        iteration=iteration,
                        data_gb=peer_data_gb,
                        source=f"{self._partition_name(partition)}/dram",
                        destination="paired-rank-cohort/dram",
                        details={
                            "checkpoint_strategy": "ssd_tiered",
                            "peer_pair_checkpoint_gb": peer_data_gb,
                            "represented_pairs": (
                                partition.represented_rank_count // 2
                            ),
                            "chunk_index": 1,
                            "chunk_count": 1,
                            "base_duration": base_duration,
                            "bandwidth_contention_seconds": (
                                contention_seconds
                            ),
                            "effective_slowdown": result.effective_slowdown,
                        },
                    )

                start = self.backend.now
                peer_ssd_duration = (
                    checkpoint_gb / self.cluster.local_ssd_bandwidth_gbps
                )
                yield self.backend.timeout(peer_ssd_duration)
                for partition in self.partitions:
                    self._record(
                        partition=partition,
                        start=start,
                        operation="checkpoint_peer_dram_to_peer_ssd_chunk",
                        category="Checkpoint",
                        resources=["CPU"],
                        iteration=iteration,
                        data_gb=checkpoint_gb,
                        source="paired-rank-cohort/dram",
                        destination="paired-rank-cohort/ssd",
                        details={
                            "checkpoint_strategy": "ssd_tiered",
                            "represented_pairs": (
                                partition.represented_rank_count // 2
                            ),
                            "chunk_index": 1,
                            "chunk_count": 1,
                            "base_duration": peer_ssd_duration,
                            "effective_slowdown": 1.0,
                        },
                    )
            finally:
                self.checkpoint_cpu_tasks -= 1

    def _poisson(self, expected: float) -> int:
        if expected <= 0:
            return 0
        if expected > 30:
            return max(0, round(self.rng.gauss(expected, math.sqrt(expected))))
        limit = math.exp(-expected)
        product = 1.0
        count = 0
        while product > limit:
            count += 1
            product *= self.rng.random()
        return count - 1

    def _failure_monitor(self):
        expected_per_tick = self.rank_count * self.config.failure_rate
        while self.active:
            yield self.backend.timeout(1.0)
            if not self.active:
                return
            failures = self._poisson(expected_per_tick)
            for _ in range(failures):
                if len(self.deviant_ranks) >= 128:
                    return
                rank = self.rng.randrange(self.rank_count)
                if rank in self.deviant_ranks:
                    continue
                self.deviant_ranks.add(rank)
                self._record_failure(rank)

    def _record_failure(self, rank: int) -> None:
        failure_type = self.rng.choices(
            ("process", "node", "spot"),
            weights=(
                self.failure_settings.process_weight,
                self.failure_settings.node_weight,
                self.failure_settings.spot_weight,
            ),
            k=1,
        )[0]
        restart_seconds = getattr(
            self.failure_settings,
            f"{failure_type}_restart_seconds",
        )
        start = self.backend.now
        restart_end = start + restart_seconds
        self.logger.record(
            start=start,
            end=restart_end,
            run_id=self.run_id,
            job_id=self.config.job_id,
            rank=rank,
            node=f"rank-{rank}",
            physical_node=f"node-{self.node_offset + rank}",
            pipeline_stage=0,
            data_parallel_rank=rank,
            category="Failure",
            operation=f"{failure_type}_failure_restart",
            resources=["CPU", "GPU"],
            iteration=self.current_iteration,
            failure_type=failure_type,
            details={
                "aggregate_sampled_failure": True,
                "highest_severity": failure_type,
                "observed_failure_types": [failure_type],
                "coalesced_failure_signals": 1,
                "simultaneous_failed_nodes": 1,
                "dram_flushed_gb": (
                    self.config.checkpoint_gb
                    if failure_type in {"node", "spot"}
                    else 0.0
                ),
                "ssd_flushed_gb": (
                    self.config.checkpoint_gb
                    if (
                        self.config.checkpoint_strategy == "ssd_tiered"
                        and failure_type == "spot"
                    )
                    else 0.0
                ),
                "base_duration": restart_seconds,
                "effective_slowdown": 1.0,
            },
        )
        checkpoint_iteration = max(
            0,
            self.current_iteration
            - self.current_iteration % self.config.checkpoint_every,
        )
        cursor = restart_end
        if self.config.checkpoint_strategy == "object_store":
            duration = (
                self.config.checkpoint_gb
                / self.cluster.object_store_bandwidth_gbps
            )
            self.logger.record(
                start=cursor,
                end=cursor + duration,
                run_id=self.run_id,
                job_id=self.config.job_id,
                rank=rank,
                node=f"rank-{rank}",
                physical_node=f"node-{self.node_offset + rank}",
                pipeline_stage=0,
                data_parallel_rank=rank,
                category="Recovery",
                operation="object_store_to_dram",
                resources=["CPU", "NETWORK"],
                iteration=checkpoint_iteration,
                source="object-store",
                destination=f"rank-{rank}/dram",
                data_gb=self.config.checkpoint_gb,
                details={
                    "checkpoint_strategy": "object_store",
                    "checkpoint_iteration": checkpoint_iteration,
                    "checkpoint_source_tier": "object_store",
                    "checkpoint_source_rank": None,
                    "base_duration": duration,
                    "effective_slowdown": 1.0,
                },
            )
            cursor += duration
            checkpoint_source_tier = "object_store"
            checkpoint_source_rank = None
        elif failure_type in {"node", "spot"}:
            bandwidth = (
                min(
                    self.cluster.local_ssd_bandwidth_gbps,
                    self.cluster.network_bandwidth_gbps,
                )
                if failure_type == "spot"
                else self.cluster.local_ssd_bandwidth_gbps
            )
            duration = self.config.checkpoint_gb / bandwidth
            operation = (
                "checkpoint_peer_ssd_to_dram_recovery_chunk"
                if failure_type == "spot"
                else "checkpoint_ssd_to_dram_recovery_chunk"
            )
            resources = (
                ["CPU", "NETWORK"]
                if failure_type == "spot"
                else ["CPU"]
            )
            self.logger.record(
                start=cursor,
                end=cursor + duration,
                run_id=self.run_id,
                job_id=self.config.job_id,
                rank=rank,
                node=f"rank-{rank}",
                physical_node=f"node-{self.node_offset + rank}",
                pipeline_stage=0,
                data_parallel_rank=rank,
                category="Recovery",
                operation=operation,
                resources=resources,
                iteration=checkpoint_iteration,
                source=(
                    f"rank-{rank ^ 1}/ssd"
                    if failure_type == "spot"
                    else f"rank-{rank}/ssd"
                ),
                destination=f"rank-{rank}/dram",
                data_gb=self.config.checkpoint_gb,
                details={
                    "checkpoint_strategy": "ssd_tiered",
                    "checkpoint_iteration": checkpoint_iteration,
                    "checkpoint_source_tier": "ssd",
                    "checkpoint_source_rank": (
                        rank ^ 1 if failure_type == "spot" else rank
                    ),
                    "base_duration": duration,
                    "effective_slowdown": 1.0,
                },
            )
            cursor += duration
            checkpoint_source_tier = "ssd"
            checkpoint_source_rank = (
                rank ^ 1 if failure_type == "spot" else rank
            )
        else:
            checkpoint_source_tier = "dram"
            checkpoint_source_rank = rank
        restore_duration = (
            self.config.checkpoint_gb
            / self.cluster.gpu_cpu_bandwidth_gbps
        )
        self.logger.record(
            start=cursor,
            end=cursor + restore_duration,
            run_id=self.run_id,
            job_id=self.config.job_id,
            rank=rank,
            node=f"rank-{rank}",
            physical_node=f"node-{self.node_offset + rank}",
            pipeline_stage=0,
            data_parallel_rank=rank,
            category="Recovery",
            operation="dram_to_gpu_restore",
            resources=["CPU", "GPU"],
            iteration=checkpoint_iteration,
            source=f"rank-{rank}/dram",
            destination=f"rank-{rank}",
            data_gb=self.config.checkpoint_gb,
            details={
                "checkpoint_strategy": self.config.checkpoint_strategy,
                "checkpoint_iteration": checkpoint_iteration,
                "checkpoint_source_tier": checkpoint_source_tier,
                "checkpoint_source_rank": checkpoint_source_rank,
                "base_duration": restore_duration,
                "effective_slowdown": 1.0,
            },
        )

    def run(self):
        if self.verbose:
            print(
                f"starting {self.config.job_id}: "
                f"strategy=data_parallel, ranks={self.rank_count:,}, "
                "execution=aggregated cohort"
            )
        monitor = (
            self.backend.process(self._failure_monitor())
            if self.config.failure_rate > 0
            else None
        )
        for iteration in range(1, self.iterations + 1):
            self.current_iteration = iteration
            yield from self._gpu_work(
                duration=self.config.forward_seconds,
                operation="data_parallel_forward",
                iteration=iteration,
            )
            yield from self._gpu_work(
                duration=self.config.backward_seconds,
                operation="data_parallel_backward",
                iteration=iteration,
            )
            yield from self._all_reduce(iteration=iteration)
            yield from self._gpu_work(
                duration=self.config.optimizer_seconds,
                operation="data_parallel_optimizer",
                iteration=iteration,
            )
            if iteration % self.config.checkpoint_every == 0:
                checkpoint_process = self.backend.process(
                    self._checkpoint(iteration=iteration)
                )
                self.checkpoint_processes.append(checkpoint_process)
                if self.config.checkpoint_mode == "synchronous":
                    yield checkpoint_process
        if self.checkpoint_processes:
            yield self.backend.all_of(self.checkpoint_processes)
        self.active = False
        if monitor is not None:
            yield monitor
