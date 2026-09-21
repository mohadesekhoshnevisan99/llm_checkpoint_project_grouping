from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    seed: int
    iterations: int


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    node_count: int
    cpu_cores_per_node: int
    cpu_memory_gb: float
    gpus_per_node: int
    gpu_memory_gb: float
    gpu_cpu_bandwidth_gbps: float
    network_bandwidth_gbps: float
    communication_launch_seconds: float
    local_ssd_bandwidth_gbps: float
    object_store_bandwidth_gbps: float
    object_store_concurrency: int


@dataclass(frozen=True, slots=True)
class FailureSettings:
    probability_per_second: float
    process_weight: float
    node_weight: float
    spot_weight: float
    process_restart_seconds: float
    node_restart_seconds: float
    spot_restart_seconds: float
    # `reboot` (2026-07-21, reboot_failure_type_spec.md): a fourth organic type —
    # host restarts on the SAME node, so DRAM/GPU state is LOST but the local disk
    # (own SSD copy + hosted donor pieces) SURVIVES. Defaults keep every pre-reboot
    # scenario byte-identical: weight 0 => the class is never drawn, and the
    # restart value is inert while the weight is 0.
    reboot_weight: float = 0.0
    reboot_restart_seconds: float = 180.0
    # `rack` (2026-07-27, rack_failure_spec.md): correlated whole-rack loss —
    # replacement-node provisioning time for the affected cohorts. None => fall
    # back to node_restart_seconds (a rack restart IS a node replacement, just
    # for every node of the rack at once). Inert for every rack-less scenario.
    rack_restart_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class JobConfig:
    job_id: str
    strategy: str
    data_parallel_replicas: int
    pipeline_stages: int
    microbatches: int
    model_weights_gb: float
    gradient_gb: float
    activation_gb: float
    checkpoint_gb: float
    forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    checkpoint_every: int
    checkpoint_strategy: str
    checkpoint_mode: str
    checkpoint_chunk_gb: float
    checkpoint_upload_gpu_slowdown: float
    failure_rate: float
    # Validation-only fidelity knobs. Both default off so existing simulator
    # inputs preserve their historical execution order.
    async_checkpoint_backpressure: bool = False
    async_checkpoint_cleanup_seconds: float = 0.0

    @property
    def rank_count(self) -> int:
        return self.data_parallel_replicas * self.pipeline_stages

    def global_rank(self, data_parallel_rank: int, pipeline_stage: int) -> int:
        return data_parallel_rank * self.pipeline_stages + pipeline_stage


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    simulation: SimulationSettings
    cluster: ClusterConfig
    failures: FailureSettings
    jobs: tuple[JobConfig, ...]

    @property
    def required_nodes(self) -> int:
        logical_ranks = sum(job.rank_count for job in self.jobs)
        return math.ceil(logical_ranks / self.cluster.gpus_per_node)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _load_raw_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()

    if suffix == ".toml":
        with config_path.open("rb") as handle:
            return tomllib.load(handle)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config files require PyYAML. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a mapping")
        return raw

    raise ValueError(
        f"Unsupported config format {suffix!r}; use .toml, .yaml, or .yml"
    )


def _checkpoint_mode(value: Any) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"sync", "synchronous", "blocking"}:
        return "synchronous"
    if mode in {"async", "asynchronous", "background"}:
        return "asynchronous"
    raise ValueError(
        "checkpoint.mode must be 'synchronous' or 'asynchronous'"
    )


def load_config(path: str | Path) -> SimulatorConfig:
    raw = _load_raw_config(path)

    simulation_raw = raw["simulation"]
    cluster_raw = raw["cluster"]
    failures_raw = raw.get("failures", {})
    simulation = SimulationSettings(
        seed=int(simulation_raw.get("seed", 17)),
        iterations=int(simulation_raw.get("iterations", 20)),
    )
    cpu_cores_per_node = int(cluster_raw.get("cpu_cores_per_node", 1))
    cluster = ClusterConfig(
        node_count=int(cluster_raw["node_count"]),
        cpu_cores_per_node=cpu_cores_per_node,
        cpu_memory_gb=float(
            cluster_raw.get(
                "cpu_memory_gb",
                cluster_raw.get(
                    "host_memory_gb",
                    max(1.0, cpu_cores_per_node * 3.75),
                ),
            )
        ),
        gpus_per_node=int(cluster_raw.get("gpus_per_node", 1)),
        gpu_memory_gb=float(cluster_raw["gpu_memory_gb"]),
        gpu_cpu_bandwidth_gbps=float(
            cluster_raw.get("gpu_cpu_bandwidth_gbps", 31.5)
        ),
        network_bandwidth_gbps=float(
            cluster_raw.get("network_bandwidth_gbps", 25.0)
        ),
        communication_launch_seconds=float(
            cluster_raw.get("communication_launch_seconds", 0.01)
        ),
        local_ssd_bandwidth_gbps=float(
            cluster_raw.get("local_ssd_bandwidth_gbps", 4.0)
        ),
        object_store_bandwidth_gbps=float(
            cluster_raw.get("object_store_bandwidth_gbps", 8.0)
        ),
        object_store_concurrency=int(
            cluster_raw.get("object_store_concurrency", 4)
        ),
    )
    failures = FailureSettings(
        probability_per_second=float(
            failures_raw.get("probability_per_second", 0.003)
        ),
        process_weight=float(failures_raw.get("process_weight", 0.5)),
        node_weight=float(failures_raw.get("node_weight", 0.3)),
        spot_weight=float(failures_raw.get("spot_weight", 0.2)),
        process_restart_seconds=float(
            failures_raw.get("process_restart_seconds", 3.0)
        ),
        node_restart_seconds=float(
            failures_raw.get("node_restart_seconds", 8.0)
        ),
        spot_restart_seconds=float(
            failures_raw.get("spot_restart_seconds", 15.0)
        ),
        reboot_weight=float(failures_raw.get("reboot_weight", 0.0)),
        reboot_restart_seconds=float(
            failures_raw.get("reboot_restart_seconds", 180.0)
        ),
    )

    jobs: list[JobConfig] = []
    seen_ids: set[str] = set()
    for entry in raw.get("jobs", []):
        job_id = str(entry["id"])
        if job_id in seen_ids:
            raise ValueError(f"Duplicate job id: {job_id}")
        seen_ids.add(job_id)

        model = entry["model"]
        timing = entry["timing"]
        checkpoint = entry.get("checkpoint", {})
        checkpoint_mode = _checkpoint_mode(
            checkpoint.get(
                "mode",
                checkpoint.get("checkpoint_mode", "asynchronous"),
            )
        )
        requested_stages = int(entry.get("pipeline_stages", 0))
        weights_gb = float(model["weights_gb"])
        if requested_stages <= 0:
            usable_memory = cluster.gpu_memory_gb * 0.85
            requested_stages = max(2, math.ceil(weights_gb / usable_memory))

        job = JobConfig(
            job_id=job_id,
            strategy=str(entry["strategy"]),
            data_parallel_replicas=int(entry["data_parallel_replicas"]),
            pipeline_stages=requested_stages,
            microbatches=int(entry.get("microbatches", 4)),
            model_weights_gb=weights_gb,
            gradient_gb=float(model.get("gradient_gb", weights_gb)),
            activation_gb=float(model.get("activation_gb", 1.0)),
            checkpoint_gb=float(
                checkpoint.get("size_gb", weights_gb)
            ),
            forward_seconds=float(timing["forward_seconds"]),
            backward_seconds=float(timing["backward_seconds"]),
            optimizer_seconds=float(timing.get("optimizer_seconds", 1.0)),
            checkpoint_every=int(checkpoint.get("every", 1)),
            checkpoint_strategy=str(
                checkpoint.get("strategy", "object_store")
            ),
            checkpoint_mode=checkpoint_mode,
            checkpoint_chunk_gb=float(checkpoint.get("chunk_gb", 4.0)),
            checkpoint_upload_gpu_slowdown=float(
                checkpoint.get("upload_gpu_slowdown", 1.15)
            ),
            failure_rate=float(
                entry.get(
                    "failure_rate",
                    failures.probability_per_second,
                )
            ),
            async_checkpoint_backpressure=bool(
                checkpoint.get("backpressure", False)
            ),
            async_checkpoint_cleanup_seconds=float(
                checkpoint.get("cleanup_seconds", 0.0)
            ),
        )
        jobs.append(job)

    _positive("simulation.iterations", simulation.iterations)
    for field_name, value in asdict(cluster).items():
        _positive(f"cluster.{field_name}", value)
    if not jobs:
        raise ValueError("At least one [[jobs]] entry is required")
    if not 0.0 <= failures.probability_per_second <= 1.0:
        raise ValueError(
            "failures.probability_per_second must be in [0, 1]"
        )
    failure_weights = (
        failures.process_weight,
        failures.reboot_weight,
        failures.node_weight,
        failures.spot_weight,
    )
    if any(weight < 0 for weight in failure_weights) or sum(
        failure_weights
    ) <= 0:
        raise ValueError("Failure weights must be nonnegative with a positive sum")
    for name in (
        "process_restart_seconds",
        "reboot_restart_seconds",
        "node_restart_seconds",
        "spot_restart_seconds",
    ):
        _positive(f"failures.{name}", getattr(failures, name))

    for job in jobs:
        for field_name, value in asdict(job).items():
            if field_name in {
                "job_id",
                "strategy",
                "checkpoint_strategy",
                "checkpoint_mode",
                "failure_rate",
                "async_checkpoint_backpressure",
                "async_checkpoint_cleanup_seconds",
            }:
                continue
            _positive(f"jobs.{job.job_id}.{field_name}", value)
        if job.async_checkpoint_cleanup_seconds < 0:
            raise ValueError(
                f"{job.job_id}: checkpoint.cleanup_seconds must be nonnegative"
            )
        if not 0.0 <= job.failure_rate <= 1.0:
            raise ValueError(
                f"{job.job_id}: failure_rate must be in [0, 1]"
            )
        if job.strategy not in {
            "data_parallel",
            "pipeline_parallel",
            "hybrid",
        }:
            raise ValueError(
                f"{job.job_id}: unsupported strategy {job.strategy!r}"
            )
        if (
            job.strategy == "data_parallel"
            and job.pipeline_stages != 1
        ):
            raise ValueError(
                f"{job.job_id}: data_parallel requires pipeline_stages = 1"
            )
        if (
            job.strategy == "pipeline_parallel"
            and (
                job.data_parallel_replicas != 1
                or job.pipeline_stages < 2
            )
        ):
            raise ValueError(
                f"{job.job_id}: pipeline_parallel requires one data replica "
                "and at least two pipeline stages"
            )
        if (
            job.strategy == "hybrid"
            and (
                job.data_parallel_replicas < 2
                or job.pipeline_stages < 2
            )
        ):
            raise ValueError(
                f"{job.job_id}: hybrid requires at least two data replicas "
                "and two pipeline stages"
            )
        if job.checkpoint_strategy not in {
            "object_store",
            "cpu_tiered",
            "ssd_tiered",
            "local_tiered",
            "paired_tiered",
            "crossjob_peer",
        }:
            raise ValueError(
                f"{job.job_id}: unsupported checkpoint strategy "
                f"{job.checkpoint_strategy!r}"
            )
        if cluster.gpus_per_node > 1 and job.failure_rate > 0:
            raise ValueError(
                f"{job.job_id}: multi-GPU configured placement currently "
                "requires failure_rate = 0 because host-correlated failures "
                "are not yet modeled across colocated jobs"
            )
        if cluster.gpus_per_node > 1 and job.rank_count > 10_000:
            raise ValueError(
                f"{job.job_id}: multi-GPU placement is currently supported "
                "only by the detailed runtime (at most 10,000 ranks)"
            )
        if (
            job.checkpoint_strategy == "paired_tiered"
            and job.rank_count % 2 != 0
        ):
            raise ValueError(
                f"{job.job_id}: paired_tiered requires an even rank count"
            )
        stage_weights = job.model_weights_gb / job.pipeline_stages
        if stage_weights > cluster.gpu_memory_gb:
            raise ValueError(
                f"{job.job_id}: each pipeline stage needs "
                f"{stage_weights:.2f} GB but a GPU has "
                f"{cluster.gpu_memory_gb:.2f} GB"
            )

    config = SimulatorConfig(
        simulation=simulation,
        cluster=cluster,
        failures=failures,
        jobs=tuple(jobs),
    )
    if config.required_nodes > cluster.node_count:
        raise ValueError(
            f"Jobs require {config.required_nodes} nodes at "
            f"{cluster.gpus_per_node} GPU(s) per node, but the cluster "
            f"defines {cluster.node_count}"
        )
    return config
