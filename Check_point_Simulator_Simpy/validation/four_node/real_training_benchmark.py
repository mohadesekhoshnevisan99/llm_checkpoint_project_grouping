from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch
import torch.distributed as dist
from torch import nn


GB = 1024**3
MB = 1024**2
T = TypeVar("T")
MARKER_DURATION_SECONDS = 0.000001


@dataclass(slots=True)
class TraceEvent:
    event_id: int
    start: float
    end: float
    duration: float
    rank: int | None
    node: str
    category: str
    operation: str
    run_id: int | None = 0
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


class JsonlLogger:
    def __init__(self, *, rank: int) -> None:
        self.events: list[TraceEvent] = []
        self._next_event_id = rank * 1_000_000 + 1
        self._lock = threading.Lock()

    def record(
        self,
        *,
        start: float,
        end: float,
        rank: int | None,
        node: str,
        category: str,
        operation: str,
        job_id: str,
        pipeline_stage: int,
        data_parallel_rank: int,
        physical_node: str,
        resources: list[str],
        iteration: int,
        source: str | None = None,
        destination: str | None = None,
        data_gb: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.events.append(
                TraceEvent(
                    event_id=self._next_event_id,
                    start=round(start, 9),
                    end=round(end, 9),
                    duration=round(end - start, 9),
                    rank=rank,
                    node=f"rank-{rank}" if rank is not None else node,
                    category=category,
                    operation=operation,
                    job_id=job_id,
                    pipeline_stage=pipeline_stage,
                    data_parallel_rank=data_parallel_rank,
                    physical_node=physical_node,
                    resources=resources,
                    iteration=iteration,
                    source=source,
                    destination=destination,
                    data_gb=None if data_gb is None else round(data_gb, 9),
                    details=details or {},
                )
            )
            self._next_event_id += 1

    def write_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(
            self.events,
            key=lambda event: (event.start, event.end, event.event_id),
        )
        with path.open("w", encoding="utf-8") as handle:
            for event in ordered:
                handle.write(json.dumps(asdict(event), sort_keys=True))
                handle.write("\n")
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real PyTorch data/pipeline parallel validation traces."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("data_parallel", "pipeline_parallel"),
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=("synchronous", "asynchronous", "sync", "async"),
        default="synchronous",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--tensor-mb", type=float, default=8.0)
    parser.add_argument("--checkpoint-mb", type=float, default=16.0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--gpu-memory-target-percent", type=float, default=75.0)
    parser.add_argument("--gpu-memory-reserve-safety-mb", type=float, default=1024.0)
    parser.add_argument("--bucket-uri", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--node-name", default=socket.gethostname())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def normalize_checkpoint_mode(mode: str) -> str:
    if mode in {"sync", "synchronous"}:
        return "synchronous"
    if mode in {"async", "asynchronous"}:
        return "asynchronous"
    raise ValueError("--checkpoint-mode must be synchronous or asynchronous")


def checkpoint_mode_slug(mode: str) -> str:
    return "sync" if mode == "synchronous" else "async"


def job_id_for(mode: str, checkpoint_mode: str) -> str:
    parallelism = "data-parallel" if mode == "data_parallel" else "pipeline-parallel"
    return f"real-{parallelism}-{checkpoint_mode_slug(checkpoint_mode)}"


def init_distributed() -> tuple[int, int, torch.device, str]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    cuda = torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    backend = "nccl" if cuda else "gloo"
    init_kwargs: dict[str, Any] = {
        "backend": backend,
        "init_method": "env://",
        "rank": rank,
        "world_size": world_size,
        "timeout": timedelta(minutes=30),
    }
    if cuda:
        init_kwargs["device_id"] = device
    try:
        dist.init_process_group(**init_kwargs)
    except TypeError:
        init_kwargs.pop("device_id", None)
        dist.init_process_group(**init_kwargs)
    return rank, world_size, device, backend


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def barrier(device: torch.device) -> None:
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index or 0])
    else:
        dist.barrier()


def aligned_origin(rank: int, device: torch.device) -> float:
    barrier(device)
    value = time.time() + 3.0 if rank == 0 else 0.0
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.broadcast(tensor, src=0)
    origin = float(tensor.item())
    while time.time() < origin:
        time.sleep(0.001)
    barrier(device)
    return origin


def tensor_gb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / GB


def model_parameter_gb(model: nn.Module) -> float:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / GB


def gpu_memory_snapshot(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"cuda": False}
    sync_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda": True,
        "total_gb": total_bytes / GB,
        "free_gb": free_bytes / GB,
        "used_gb": (total_bytes - free_bytes) / GB,
        "torch_allocated_gb": torch.cuda.memory_allocated(device) / GB,
        "torch_reserved_gb": torch.cuda.memory_reserved(device) / GB,
        "torch_max_allocated_gb": torch.cuda.max_memory_allocated(device) / GB,
        "torch_max_reserved_gb": torch.cuda.max_memory_reserved(device) / GB,
    }


def cpu_memory_snapshot() -> dict[str, Any]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return {"available": False}
    total_bytes = int(page_size) * int(page_count)
    return {
        "available": True,
        "total_gb": total_bytes / GB,
        "cpu_count": os.cpu_count(),
    }


def batch_size_for_tensor(tensor_mb: float, hidden_size: int) -> int:
    elements = max(1, int(tensor_mb * MB / torch.empty((), dtype=torch.float32).element_size()))
    return max(1, elements // hidden_size)


def build_model(hidden_size: int, layers: int, device: torch.device) -> nn.Module:
    blocks: list[nn.Module] = []
    for _ in range(layers):
        blocks.append(nn.Linear(hidden_size, hidden_size))
        blocks.append(nn.ReLU())
    blocks.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*blocks).to(device)


def log_status(
    *,
    rank: int,
    node_name: str,
    job_id: str,
    operation: str,
    phase: str,
    iteration: int,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "source": "pytorch_validation",
        "time_unix": round(time.time(), 6),
        "rank": rank,
        "node": node_name,
        "job_id": job_id,
        "operation": operation,
        "phase": phase,
        "iteration": iteration,
    }
    if details:
        payload["details"] = details
    print(f"[simval] {json.dumps(payload, sort_keys=True)}", flush=True)


def trace_marker(
    *,
    logger: JsonlLogger,
    origin_wall: float,
    rank: int,
    node_name: str,
    job_id: str,
    pipeline_stage: int,
    data_parallel_rank: int,
    operation: str,
    category: str,
    resources: list[str],
    iteration: int,
    details: dict[str, Any] | None = None,
    source: str | None = None,
    destination: str | None = None,
    data_gb: float | None = None,
) -> None:
    event_details = {
        "source_system": "pytorch_gce",
        "marker": True,
        "base_duration": MARKER_DURATION_SECONDS,
        "effective_slowdown": 1.0,
    }
    event_details.update(details or {})
    log_status(
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        operation=operation,
        phase="marker",
        iteration=iteration,
        details=event_details,
    )
    start = time.time() - origin_wall
    logger.record(
        start=start,
        end=start + MARKER_DURATION_SECONDS,
        rank=rank,
        node=f"rank-{rank}",
        category=category,
        operation=operation,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        physical_node=node_name,
        resources=resources,
        iteration=iteration,
        source=source,
        destination=destination,
        data_gb=data_gb,
        details=event_details,
    )


def trace_marker_from_context(
    context: dict[str, Any],
    *,
    job_id: str,
    operation: str,
    category: str,
    resources: list[str],
    iteration: int,
    pipeline_stage: int,
    data_parallel_rank: int,
    details: dict[str, Any] | None = None,
    source: str | None = None,
    destination: str | None = None,
    data_gb: float | None = None,
) -> None:
    trace_marker(
        logger=context["logger"],
        origin_wall=context["origin_wall"],
        rank=context["rank"],
        node_name=context["node_name"],
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation=operation,
        category=category,
        resources=resources,
        iteration=iteration,
        details=details,
        source=source,
        destination=destination,
        data_gb=data_gb,
    )


def measure(
    fn: Callable[[], T],
    *,
    logger: JsonlLogger,
    log_event: bool,
    origin_wall: float,
    device: torch.device,
    rank: int,
    node_name: str,
    job_id: str,
    pipeline_stage: int,
    data_parallel_rank: int,
    operation: str,
    category: str,
    resources: list[str],
    iteration: int,
    source: str | None = None,
    destination: str | None = None,
    data_gb: float | None = None,
    details: dict[str, Any] | None = None,
) -> T:
    if log_event:
        log_status(
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            operation=operation,
            phase="start",
            iteration=iteration,
            details=details,
        )
    sync_device(device)
    start = time.time() - origin_wall
    perf_start = time.perf_counter()
    result = fn()
    sync_device(device)
    duration = time.perf_counter() - perf_start
    end = start + duration
    if log_event:
        event_details = {
            "source_system": "pytorch_gce",
            "base_duration": duration,
            "effective_slowdown": 1.0,
        }
        event_details.update(details or {})
        logger.record(
            start=start,
            end=end,
            rank=rank,
            node=f"rank-{rank}",
            category=category,
            operation=operation,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            physical_node=node_name,
            resources=resources,
            iteration=iteration,
            source=source,
            destination=destination,
            data_gb=data_gb,
            details=event_details,
        )
        log_status(
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            operation=operation,
            phase="end",
            iteration=iteration,
            details={
                **event_details,
                "duration": duration,
                "source": source,
                "destination": destination,
                "data_gb": data_gb,
            },
        )
    return result


def reserve_gpu_memory(
    *,
    logger: JsonlLogger,
    origin_wall: float,
    device: torch.device,
    rank: int,
    node_name: str,
    job_id: str,
    pipeline_stage: int,
    data_parallel_rank: int,
    target_percent: float,
    safety_mb: float,
) -> torch.Tensor | None:
    if device.type != "cuda":
        trace_marker(
            logger=logger,
            origin_wall=origin_wall,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            operation="gpu_memory_reserve_skipped",
            category="Setup",
            resources=["CPU"],
            iteration=0,
            details={"reason": "cuda_not_available", "target_percent": target_percent},
        )
        return None
    if target_percent <= 0:
        trace_marker(
            logger=logger,
            origin_wall=origin_wall,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            operation="gpu_memory_reserve_skipped",
            category="Setup",
            resources=["GPU"],
            iteration=0,
            details={"reason": "target_disabled", "target_percent": target_percent},
        )
        return None

    before = gpu_memory_snapshot(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    target_bytes = int(total_bytes * min(target_percent, 95.0) / 100.0)
    used_bytes = total_bytes - free_bytes
    safety_bytes = int(max(safety_mb, 0.0) * MB)
    requested_bytes = max(0, target_bytes - used_bytes)
    reserve_bytes = max(0, min(requested_bytes, free_bytes - safety_bytes))
    if reserve_bytes < MB:
        trace_marker(
            logger=logger,
            origin_wall=origin_wall,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            operation="gpu_memory_reserve_skipped",
            category="Setup",
            resources=["GPU"],
            iteration=0,
            data_gb=reserve_bytes / GB,
            details={
                "reason": "already_at_or_above_target",
                "target_percent": target_percent,
                "safety_mb": safety_mb,
                "before": before,
            },
        )
        return None

    elements = reserve_bytes // torch.empty((), dtype=torch.uint8).element_size()

    reserve = measure(
        lambda: torch.empty(elements, dtype=torch.uint8, device=device),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="gpu_memory_reserve_allocate",
        category="Setup",
        resources=["GPU"],
        iteration=0,
        data_gb=reserve_bytes / GB,
        details={
            "target_percent": target_percent,
            "safety_mb": safety_mb,
            "requested_gb": requested_bytes / GB,
            "reserve_gb": reserve_bytes / GB,
            "before": before,
        },
    )
    measure(
        lambda: reserve.fill_(1),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="gpu_memory_reserve_touch",
        category="Setup",
        resources=["GPU"],
        iteration=0,
        data_gb=reserve_bytes / GB,
        details={
            "target_percent": target_percent,
            "safety_mb": safety_mb,
            "reserve_gb": reserve_bytes / GB,
        },
    )
    trace_marker(
        logger=logger,
        origin_wall=origin_wall,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="gpu_memory_reserve_ready",
        category="Setup",
        resources=["GPU"],
        iteration=0,
        data_gb=reserve_bytes / GB,
        details={
            "target_percent": target_percent,
            "safety_mb": safety_mb,
            "after": gpu_memory_snapshot(device),
        },
    )
    return reserve


def storage_cp(source: Path, destination: str) -> None:
    if shutil.which("gcloud") is not None:
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", str(source), destination],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return
        except subprocess.CalledProcessError:
            if shutil.which("gsutil") is None:
                raise
    if shutil.which("gsutil") is not None:
        subprocess.run(
            ["gsutil", "cp", str(source), destination],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return
    raise RuntimeError("No gcloud or gsutil command was found for checkpoint upload")


def storage_rm(destination: str) -> None:
    if shutil.which("gcloud") is not None:
        subprocess.run(
            ["gcloud", "storage", "rm", destination],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return
    if shutil.which("gsutil") is not None:
        subprocess.run(
            ["gsutil", "rm", destination],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def remove_local_checkpoint(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def persist_checkpoint(
    *,
    logger: JsonlLogger,
    origin_wall: float,
    rank: int,
    node_name: str,
    job_id: str,
    pipeline_stage: int,
    data_parallel_rank: int,
    iteration: int,
    mode: str,
    checkpoint_mode: str,
    cpu_tensor: torch.Tensor,
    local_path: Path,
    shard_gb: float,
    bucket_uri: str,
) -> None:
    cpu_device = torch.device("cpu")
    measure(
        lambda: torch.save(
            {
                "mode": mode,
                "checkpoint_mode": checkpoint_mode,
                "rank": rank,
                "iteration": iteration,
                "tensor": cpu_tensor,
            },
            local_path,
        ),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=cpu_device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="checkpoint_stage_dram_to_local_disk",
        category="Checkpoint",
        resources=["CPU", "LOCAL_DISK"],
        iteration=iteration,
        source=f"rank-{rank}",
        destination=str(local_path),
        data_gb=shard_gb,
        details={
            "checkpoint_strategy": "object_store",
            "checkpoint_mode": checkpoint_mode,
            "mode": mode,
            "wait_reason": "object_store_upload_about_to_start",
        },
    )

    if not bucket_uri:
        trace_marker(
            logger=logger,
            origin_wall=origin_wall,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            operation="checkpoint_upload_skipped",
            category="Checkpoint",
            resources=["CPU"],
            iteration=iteration,
            source=str(local_path),
            destination="object-store",
            data_gb=shard_gb,
            details={
                "checkpoint_strategy": "object_store",
                "checkpoint_mode": checkpoint_mode,
                "reason": "no_bucket_uri",
            },
        )
        trace_marker(
            logger=logger,
            origin_wall=origin_wall,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            operation="checkpoint_complete",
            category="Checkpoint",
            resources=["CPU", "LOCAL_DISK"],
            iteration=iteration,
            source=str(local_path),
            destination=str(local_path),
            data_gb=shard_gb,
            details={
                "checkpoint_strategy": "object_store",
                "checkpoint_mode": checkpoint_mode,
                "mode": mode,
                "bucket_enabled": False,
            },
        )
        remove_local_checkpoint(local_path)
        return

    destination_uri = (
        f"{bucket_uri.rstrip('/')}/checkpoints/{checkpoint_mode_slug(checkpoint_mode)}/"
        f"{mode}/rank-{rank}/iter-{iteration}.pt"
    )
    trace_marker(
        logger=logger,
        origin_wall=origin_wall,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="checkpoint_upload_start",
        category="Checkpoint",
        resources=["CPU", "NETWORK"],
        iteration=iteration,
        source=str(local_path),
        destination=destination_uri,
        data_gb=shard_gb,
        details={
            "checkpoint_strategy": "object_store",
            "checkpoint_mode": checkpoint_mode,
            "mode": mode,
        },
    )
    measure(
        lambda: storage_cp(local_path, destination_uri),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=cpu_device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="checkpoint_stage_dram_to_object_store",
        category="Checkpoint",
        resources=["CPU", "NETWORK"],
        iteration=iteration,
        source=f"rank-{rank}",
        destination="object-store",
        data_gb=shard_gb,
        details={
            "checkpoint_strategy": "object_store",
            "checkpoint_mode": checkpoint_mode,
            "wait_reason": "uploading_checkpoint_to_object_store",
        },
    )
    trace_marker(
        logger=logger,
        origin_wall=origin_wall,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="checkpoint_complete",
        category="Checkpoint",
        resources=["CPU", "NETWORK"],
        iteration=iteration,
        source=str(local_path),
        destination=destination_uri,
        data_gb=shard_gb,
        details={
            "checkpoint_strategy": "object_store",
            "checkpoint_mode": checkpoint_mode,
            "mode": mode,
        },
    )
    storage_rm(destination_uri)
    remove_local_checkpoint(local_path)


def checkpoint(
    *,
    logger: JsonlLogger,
    origin_wall: float,
    device: torch.device,
    rank: int,
    node_name: str,
    job_id: str,
    pipeline_stage: int,
    data_parallel_rank: int,
    iteration: int,
    mode: str,
    checkpoint_mode: str,
    checkpoint_tensor: torch.Tensor,
    bucket_uri: str,
    output_dir: Path,
    background_threads: list[threading.Thread],
    background_errors: list[str],
) -> None:
    shard_gb = tensor_gb(checkpoint_tensor)
    trace_marker(
        logger=logger,
        origin_wall=origin_wall,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="checkpoint_start",
        category="Checkpoint",
        resources=["CPU", "GPU", "NETWORK"],
        iteration=iteration,
        source=f"rank-{rank}",
        destination="object-store" if bucket_uri else f"rank-{rank}",
        data_gb=shard_gb,
        details={
            "checkpoint_strategy": "object_store",
            "checkpoint_mode": checkpoint_mode,
            "mode": mode,
            "bucket_enabled": bool(bucket_uri),
        },
    )

    cpu_tensor = measure(
        lambda: checkpoint_tensor.detach().cpu(),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="checkpoint_stage_gpu_to_dram",
        category="Checkpoint",
        resources=["CPU", "GPU"],
        iteration=iteration,
        source=f"rank-{rank}",
        destination=f"rank-{rank}",
        data_gb=shard_gb,
        details={
            "checkpoint_strategy": "object_store",
            "checkpoint_mode": checkpoint_mode,
        },
    )

    checkpoint_root = output_dir.parent / "checkpoints"
    local_path = (
        checkpoint_root
        / checkpoint_mode_slug(checkpoint_mode)
        / mode
        / f"rank-{rank}"
        / f"iter-{iteration}.pt"
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    trace_marker(
        logger=logger,
        origin_wall=origin_wall,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        operation="checkpoint_local_path_ready",
        category="Filesystem",
        resources=["CPU", "LOCAL_DISK"],
        iteration=iteration,
        source=f"rank-{rank}",
        destination=str(local_path),
        data_gb=shard_gb,
        details={
            "checkpoint_strategy": "object_store",
            "checkpoint_mode": checkpoint_mode,
            "mode": mode,
            "path": str(local_path),
        },
    )

    def persist() -> None:
        try:
            persist_checkpoint(
                logger=logger,
                origin_wall=origin_wall,
                rank=rank,
                node_name=node_name,
                job_id=job_id,
                pipeline_stage=pipeline_stage,
                data_parallel_rank=data_parallel_rank,
                iteration=iteration,
                mode=mode,
                checkpoint_mode=checkpoint_mode,
                cpu_tensor=cpu_tensor,
                local_path=local_path,
                shard_gb=shard_gb,
                bucket_uri=bucket_uri,
            )
        except BaseException as exc:
            background_errors.append(
                f"rank {rank} iteration {iteration} async checkpoint failed: {exc!r}"
            )
            raise

    if checkpoint_mode == "asynchronous":
        trace_marker(
            logger=logger,
            origin_wall=origin_wall,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            operation="checkpoint_async_persist_enqueued",
            category="Checkpoint",
            resources=["CPU"],
            iteration=iteration,
            source=f"rank-{rank}",
            destination=str(local_path),
            data_gb=shard_gb,
            details={
                "checkpoint_strategy": "object_store",
                "checkpoint_mode": checkpoint_mode,
                "mode": mode,
                "bucket_enabled": bool(bucket_uri),
            },
        )
        thread = threading.Thread(
            target=persist,
            name=(
                f"checkpoint-{checkpoint_mode_slug(checkpoint_mode)}-"
                f"{mode}-rank-{rank}-iter-{iteration}"
            ),
        )
        background_threads.append(thread)
        thread.start()
        return

    persist()


def wait_for_background_checkpoints(
    context: dict[str, Any],
    *,
    job_id: str,
    pipeline_stage: int,
    data_parallel_rank: int,
    iteration: int,
) -> None:
    threads: list[threading.Thread] = context["checkpoint_threads"]
    if not threads:
        return
    logger: JsonlLogger = context["logger"]
    rank = context["rank"]
    node_name = context["node_name"]
    origin_wall = context["origin_wall"]
    start = time.time() - origin_wall
    for thread in threads:
        thread.join()
    end = time.time() - origin_wall
    if context["checkpoint_errors"]:
        raise RuntimeError("; ".join(context["checkpoint_errors"]))
    logger.record(
        start=start,
        end=end,
        rank=rank,
        node=f"rank-{rank}",
        category="Checkpoint",
        operation="checkpoint_async_drain",
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        physical_node=node_name,
        resources=["CPU", "NETWORK"],
        iteration=iteration,
        details={
            "checkpoint_strategy": "object_store",
            "checkpoint_mode": "asynchronous",
            "thread_count": len(threads),
            "base_duration": max(0.0, end - start),
            "effective_slowdown": 1.0,
        },
    )


def run_data_parallel(args: argparse.Namespace, context: dict[str, Any]) -> None:
    rank = context["rank"]
    world_size = context["world_size"]
    device = context["device"]
    logger = context["logger"]
    origin_wall = context["origin_wall"]
    node_name = context["node_name"]
    output_dir = context["output_dir"]

    job_id = job_id_for(args.mode, args.checkpoint_mode)
    batch = batch_size_for_tensor(args.tensor_mb, args.hidden_size)
    trace_marker_from_context(
        context,
        job_id=job_id,
        operation="data_parallel_setup_start",
        category="Setup",
        resources=["CPU", "GPU"],
        iteration=0,
        pipeline_stage=0,
        data_parallel_rank=rank,
        details={
            "mode": args.mode,
            "world_size": world_size,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "batch": batch,
        },
    )
    model = measure(
        lambda: build_model(args.hidden_size, args.layers, device),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=0,
        data_parallel_rank=rank,
        operation="data_parallel_model_build",
        category="Setup",
        resources=["CPU", "GPU"],
        iteration=0,
        details={"hidden_size": args.hidden_size, "layers": args.layers},
    )
    optimizer = measure(
        lambda: torch.optim.SGD(model.parameters(), lr=1e-3),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=0,
        data_parallel_rank=rank,
        operation="data_parallel_optimizer_build",
        category="Setup",
        resources=["CPU"],
        iteration=0,
        details={"optimizer": "SGD", "learning_rate": 1e-3},
    )
    inputs = measure(
        lambda: torch.randn(batch, args.hidden_size, device=device),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=0,
        data_parallel_rank=rank,
        operation="data_parallel_input_allocate",
        category="Setup",
        resources=["GPU"],
        iteration=0,
        data_gb=args.tensor_mb * MB / GB,
        details={"shape": [batch, args.hidden_size]},
    )
    target = measure(
        lambda: torch.zeros_like(inputs),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=0,
        data_parallel_rank=rank,
        operation="data_parallel_target_allocate",
        category="Setup",
        resources=["GPU"],
        iteration=0,
        data_gb=args.tensor_mb * MB / GB,
        details={"shape": [batch, args.hidden_size]},
    )
    gradient_gb = model_parameter_gb(model)
    checkpoint_elements = max(1, int(args.checkpoint_mb * MB / 4))
    checkpoint_tensor = measure(
        lambda: torch.randn(checkpoint_elements, device=device),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=0,
        data_parallel_rank=rank,
        operation="data_parallel_checkpoint_tensor_allocate",
        category="Setup",
        resources=["GPU"],
        iteration=0,
        data_gb=args.checkpoint_mb * MB / GB,
        details={"elements": checkpoint_elements},
    )
    gpu_memory_reserve = reserve_gpu_memory(
        logger=logger,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=0,
        data_parallel_rank=rank,
        target_percent=args.gpu_memory_target_percent,
        safety_mb=args.gpu_memory_reserve_safety_mb,
    )
    trace_marker_from_context(
        context,
        job_id=job_id,
        operation="data_parallel_setup_complete",
        category="Setup",
        resources=["CPU", "GPU"],
        iteration=0,
        pipeline_stage=0,
        data_parallel_rank=rank,
        details={
            "model_parameter_gb": model_parameter_gb(model),
            "gradient_gb": gradient_gb,
            "checkpoint_gb": args.checkpoint_mb * MB / GB,
            "tensor_gb": args.tensor_mb * MB / GB,
            "gpu_memory": gpu_memory_snapshot(device),
            "gpu_memory_reserve_active": gpu_memory_reserve is not None,
        },
    )

    total_iterations = args.warmup_iterations + args.iterations
    for loop_index in range(total_iterations):
        measured = loop_index >= args.warmup_iterations
        iteration = loop_index - args.warmup_iterations + 1
        phase = "measured" if measured else "warmup"
        trace_marker_from_context(
            context,
            job_id=job_id,
            operation="data_parallel_iteration_start",
            category="Iteration",
            resources=["CPU", "GPU", "NETWORK"],
            iteration=iteration,
            pipeline_stage=0,
            data_parallel_rank=rank,
            details={"phase": phase, "loop_index": loop_index},
        )
        measure(
            lambda: optimizer.zero_grad(set_to_none=True),
            logger=logger,
            log_event=True,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=0,
            data_parallel_rank=rank,
            operation="data_parallel_zero_grad",
            category="Computation",
            resources=["CPU", "GPU"],
            iteration=iteration,
            details={"phase": phase, "set_to_none": True},
        )

        def forward_step() -> tuple[torch.Tensor, torch.Tensor]:
            output = model(inputs)
            loss = torch.nn.functional.mse_loss(output, target)
            return output, loss

        _, loss = measure(
            forward_step,
            logger=logger,
            log_event=measured,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=0,
            data_parallel_rank=rank,
            operation="data_parallel_forward",
            category="Computation",
            resources=["GPU"],
            iteration=iteration,
        )

        measure(
            lambda: loss.backward(),
            logger=logger,
            log_event=measured,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=0,
            data_parallel_rank=rank,
            operation="data_parallel_backward",
            category="Computation",
            resources=["GPU"],
            iteration=iteration,
        )

        def all_reduce_step() -> None:
            for parameter_index, parameter in enumerate(model.parameters()):
                if parameter.grad is None:
                    trace_marker_from_context(
                        context,
                        job_id=job_id,
                        operation="data_parallel_all_reduce_parameter_skipped",
                        category="Communication",
                        resources=["GPU", "NETWORK"],
                        iteration=iteration,
                        pipeline_stage=0,
                        data_parallel_rank=rank,
                        details={
                            "parameter_index": parameter_index,
                            "reason": "missing_gradient",
                        },
                    )
                    continue
                grad_gb = tensor_gb(parameter.grad)

                def reduce_parameter() -> None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(world_size)

                measure(
                    reduce_parameter,
                    logger=logger,
                    log_event=measured,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=0,
                    data_parallel_rank=rank,
                    operation="data_parallel_all_reduce_parameter",
                    category="Communication",
                    resources=["GPU", "NETWORK"],
                    iteration=iteration,
                    source=f"rank-{rank}",
                    destination="real-data-parallel-stage-0-replicas",
                    data_gb=grad_gb,
                    details={
                        "parameter_index": parameter_index,
                        "shape": list(parameter.shape),
                        "numel": parameter.numel(),
                        "phase": phase,
                        "wait_reason": "waiting_for_data_parallel_peers",
                    },
                )

        measure(
            all_reduce_step,
            logger=logger,
            log_event=measured,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=0,
            data_parallel_rank=rank,
            operation="data_parallel_all_reduce",
            category="Communication",
            resources=["GPU", "NETWORK"],
            iteration=iteration,
            source=f"rank-{rank}",
            destination="real-data-parallel-stage-0-replicas",
            data_gb=gradient_gb,
            details={
                "link": "real-data-parallel/data-parallel/stage-0",
                "phase": phase,
                "parameter_count": sum(1 for _ in model.parameters()),
                "wait_reason": "waiting_for_data_parallel_peers",
            },
        )

        measure(
            lambda: optimizer.step(),
            logger=logger,
            log_event=measured,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=0,
            data_parallel_rank=rank,
            operation="data_parallel_optimizer",
            category="Computation",
            resources=["GPU"],
            iteration=iteration,
            details={
                "phase": phase,
                "wait_reason": "waiting_for_all_data_parallel_ranks",
            },
        )

        if not measured:
            trace_marker_from_context(
                context,
                job_id=job_id,
                operation="data_parallel_checkpoint_skipped",
                category="Checkpoint",
                resources=["CPU"],
                iteration=iteration,
                pipeline_stage=0,
                data_parallel_rank=rank,
                details={"reason": "warmup", "phase": phase},
            )
        elif args.checkpoint_every <= 0:
            trace_marker_from_context(
                context,
                job_id=job_id,
                operation="data_parallel_checkpoint_skipped",
                category="Checkpoint",
                resources=["CPU"],
                iteration=iteration,
                pipeline_stage=0,
                data_parallel_rank=rank,
                details={"reason": "checkpoint_disabled", "phase": phase},
            )
        elif iteration % args.checkpoint_every != 0:
            trace_marker_from_context(
                context,
                job_id=job_id,
                operation="data_parallel_checkpoint_skipped",
                category="Checkpoint",
                resources=["CPU"],
                iteration=iteration,
                pipeline_stage=0,
                data_parallel_rank=rank,
                details={
                    "reason": "not_due",
                    "checkpoint_every": args.checkpoint_every,
                    "phase": phase,
                },
            )
        elif rank != 0:
            trace_marker_from_context(
                context,
                job_id=job_id,
                operation="data_parallel_checkpoint_skipped",
                category="Checkpoint",
                resources=["CPU"],
                iteration=iteration,
                pipeline_stage=0,
                data_parallel_rank=rank,
                details={"reason": "rank_not_checkpoint_owner", "phase": phase},
            )
        else:
            checkpoint(
                logger=logger,
                origin_wall=origin_wall,
                device=device,
                rank=rank,
                node_name=node_name,
                job_id=job_id,
                pipeline_stage=0,
                data_parallel_rank=rank,
                iteration=iteration,
                mode=args.mode,
                checkpoint_mode=args.checkpoint_mode,
                checkpoint_tensor=checkpoint_tensor,
                bucket_uri=args.bucket_uri,
                output_dir=output_dir,
                background_threads=context["checkpoint_threads"],
                background_errors=context["checkpoint_errors"],
            )
        measure(
            lambda: barrier(device),
            logger=logger,
            log_event=True,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=0,
            data_parallel_rank=rank,
            operation="data_parallel_iteration_barrier",
            category="Synchronization",
            resources=["GPU", "NETWORK"],
            iteration=iteration,
            details={"phase": phase},
        )
        trace_marker_from_context(
            context,
            job_id=job_id,
            operation="data_parallel_iteration_complete",
            category="Iteration",
            resources=["CPU", "GPU", "NETWORK"],
            iteration=iteration,
            pipeline_stage=0,
            data_parallel_rank=rank,
            details={"phase": phase, "loop_index": loop_index},
        )


def run_pipeline_parallel(args: argparse.Namespace, context: dict[str, Any]) -> None:
    rank = context["rank"]
    world_size = context["world_size"]
    device = context["device"]
    logger = context["logger"]
    origin_wall = context["origin_wall"]
    node_name = context["node_name"]
    output_dir = context["output_dir"]

    job_id = job_id_for(args.mode, args.checkpoint_mode)
    batch = batch_size_for_tensor(args.tensor_mb, args.hidden_size)
    tensor_shape = (batch, args.hidden_size)
    activation_gb = batch * args.hidden_size * 4 / GB
    trace_marker_from_context(
        context,
        job_id=job_id,
        operation="pipeline_setup_start",
        category="Setup",
        resources=["CPU", "GPU"],
        iteration=0,
        pipeline_stage=rank,
        data_parallel_rank=0,
        details={
            "mode": args.mode,
            "world_size": world_size,
            "pipeline_stage": rank,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "microbatches": args.microbatches,
            "tensor_shape": list(tensor_shape),
        },
    )
    model = measure(
        lambda: build_model(args.hidden_size, args.layers, device),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=rank,
        data_parallel_rank=0,
        operation="pipeline_model_build",
        category="Setup",
        resources=["CPU", "GPU"],
        iteration=0,
        details={"hidden_size": args.hidden_size, "layers": args.layers},
    )
    optimizer = measure(
        lambda: torch.optim.SGD(model.parameters(), lr=1e-3),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=rank,
        data_parallel_rank=0,
        operation="pipeline_optimizer_build",
        category="Setup",
        resources=["CPU"],
        iteration=0,
        details={"optimizer": "SGD", "learning_rate": 1e-3},
    )
    checkpoint_elements = max(1, int(args.checkpoint_mb * MB / 4))
    checkpoint_tensor = measure(
        lambda: torch.randn(checkpoint_elements, device=device),
        logger=logger,
        log_event=True,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=rank,
        data_parallel_rank=0,
        operation="pipeline_checkpoint_tensor_allocate",
        category="Setup",
        resources=["GPU"],
        iteration=0,
        data_gb=args.checkpoint_mb * MB / GB,
        details={"elements": checkpoint_elements},
    )
    gpu_memory_reserve = reserve_gpu_memory(
        logger=logger,
        origin_wall=origin_wall,
        device=device,
        rank=rank,
        node_name=node_name,
        job_id=job_id,
        pipeline_stage=rank,
        data_parallel_rank=0,
        target_percent=args.gpu_memory_target_percent,
        safety_mb=args.gpu_memory_reserve_safety_mb,
    )
    trace_marker_from_context(
        context,
        job_id=job_id,
        operation="pipeline_setup_complete",
        category="Setup",
        resources=["CPU", "GPU"],
        iteration=0,
        pipeline_stage=rank,
        data_parallel_rank=0,
        details={
            "model_parameter_gb": model_parameter_gb(model),
            "gradient_gb": model_parameter_gb(model),
            "activation_gb": activation_gb,
            "checkpoint_gb": args.checkpoint_mb * MB / GB,
            "tensor_gb": activation_gb,
            "gpu_memory": gpu_memory_snapshot(device),
            "gpu_memory_reserve_active": gpu_memory_reserve is not None,
        },
    )

    total_iterations = args.warmup_iterations + args.iterations
    for loop_index in range(total_iterations):
        measured = loop_index >= args.warmup_iterations
        iteration = loop_index - args.warmup_iterations + 1
        phase = "measured" if measured else "warmup"
        trace_marker_from_context(
            context,
            job_id=job_id,
            operation="pipeline_iteration_start",
            category="Iteration",
            resources=["CPU", "GPU", "NETWORK"],
            iteration=iteration,
            pipeline_stage=rank,
            data_parallel_rank=0,
            details={"phase": phase, "loop_index": loop_index},
        )
        measure(
            lambda: optimizer.zero_grad(set_to_none=True),
            logger=logger,
            log_event=True,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=rank,
            data_parallel_rank=0,
            operation="pipeline_zero_grad",
            category="Computation",
            resources=["CPU", "GPU"],
            iteration=iteration,
            details={"phase": phase, "set_to_none": True},
        )

        for microbatch in range(args.microbatches):
            trace_marker_from_context(
                context,
                job_id=job_id,
                operation="pipeline_microbatch_start",
                category="Iteration",
                resources=["CPU", "GPU", "NETWORK"],
                iteration=iteration,
                pipeline_stage=rank,
                data_parallel_rank=0,
                details={"phase": phase, "microbatch": microbatch},
            )
            if rank == 0:
                input_tensor = measure(
                    lambda: torch.randn(*tensor_shape, device=device),
                    logger=logger,
                    log_event=True,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_input_allocate",
                    category="Setup",
                    resources=["GPU"],
                    iteration=iteration,
                    data_gb=activation_gb,
                    details={
                        "phase": phase,
                        "microbatch": microbatch,
                        "shape": list(tensor_shape),
                    },
                )
            else:
                input_tensor = measure(
                    lambda: torch.empty(*tensor_shape, device=device),
                    logger=logger,
                    log_event=True,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_receive_buffer_allocate",
                    category="Setup",
                    resources=["GPU"],
                    iteration=iteration,
                    data_gb=activation_gb,
                    details={
                        "phase": phase,
                        "microbatch": microbatch,
                        "shape": list(tensor_shape),
                    },
                )
                measure(
                    lambda: dist.recv(input_tensor, src=rank - 1),
                    logger=logger,
                    log_event=measured,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_activation_receive",
                    category="Communication",
                    resources=["GPU", "NETWORK"],
                    iteration=iteration,
                    source=f"rank-{rank - 1}",
                    destination=f"rank-{rank}",
                    data_gb=activation_gb,
                    details={
                        "phase": phase,
                        "microbatch": microbatch,
                        "wait_reason": "waiting_for_previous_pipeline_stage_activation",
                    },
                )
            measure(
                lambda: input_tensor.requires_grad_(True),
                logger=logger,
                log_event=True,
                origin_wall=origin_wall,
                device=device,
                rank=rank,
                node_name=node_name,
                job_id=job_id,
                pipeline_stage=rank,
                data_parallel_rank=0,
                operation="pipeline_input_requires_grad",
                category="Autograd",
                resources=["CPU", "GPU"],
                iteration=iteration,
                data_gb=activation_gb,
                details={"phase": phase, "microbatch": microbatch},
            )

            output = measure(
                lambda: model(input_tensor),
                logger=logger,
                log_event=measured,
                origin_wall=origin_wall,
                device=device,
                rank=rank,
                node_name=node_name,
                job_id=job_id,
                pipeline_stage=rank,
                data_parallel_rank=0,
                operation="pipeline_forward",
                category="Computation",
                resources=["GPU"],
                iteration=iteration,
                details={"phase": phase, "microbatch": microbatch},
            )

            if rank < world_size - 1:
                measure(
                    lambda: dist.send(output.detach(), dst=rank + 1),
                    logger=logger,
                    log_event=measured,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_activation_send",
                    category="Communication",
                    resources=["GPU", "NETWORK"],
                    iteration=iteration,
                    source=f"rank-{rank}",
                    destination=f"rank-{rank + 1}",
                    data_gb=activation_gb,
                    details={
                        "phase": phase,
                        "microbatch": microbatch,
                        "wait_reason": "waiting_for_next_pipeline_stage_gradient",
                    },
                )
                grad_output = measure(
                    lambda: torch.empty_like(output),
                    logger=logger,
                    log_event=True,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_gradient_buffer_allocate",
                    category="Setup",
                    resources=["GPU"],
                    iteration=iteration,
                    data_gb=activation_gb,
                    details={"phase": phase, "microbatch": microbatch},
                )
                measure(
                    lambda: dist.recv(grad_output, src=rank + 1),
                    logger=logger,
                    log_event=measured,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_gradient_receive",
                    category="Communication",
                    resources=["GPU", "NETWORK"],
                    iteration=iteration,
                    source=f"rank-{rank + 1}",
                    destination=f"rank-{rank}",
                    data_gb=activation_gb,
                    details={"phase": phase, "microbatch": microbatch},
                )
                measure(
                    lambda: output.backward(grad_output),
                    logger=logger,
                    log_event=measured,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_backward",
                    category="Computation",
                    resources=["GPU"],
                    iteration=iteration,
                    details={"phase": phase, "microbatch": microbatch},
                )
            else:
                target = measure(
                    lambda: torch.zeros_like(output),
                    logger=logger,
                    log_event=True,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_target_allocate",
                    category="Setup",
                    resources=["GPU"],
                    iteration=iteration,
                    data_gb=activation_gb,
                    details={"phase": phase, "microbatch": microbatch},
                )
                loss = measure(
                    lambda: torch.nn.functional.mse_loss(output, target),
                    logger=logger,
                    log_event=measured,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_loss_compute",
                    category="Computation",
                    resources=["GPU"],
                    iteration=iteration,
                    details={"phase": phase, "microbatch": microbatch},
                )
                measure(
                    lambda: loss.backward(),
                    logger=logger,
                    log_event=measured,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_backward",
                    category="Computation",
                    resources=["GPU"],
                    iteration=iteration,
                    details={"phase": phase, "microbatch": microbatch},
                )

            if rank > 0:
                if input_tensor.grad is None:
                    raise RuntimeError("Pipeline stage did not produce input gradients")
                grad_input = input_tensor.grad.detach()
                measure(
                    lambda: dist.send(grad_input, dst=rank - 1),
                    logger=logger,
                    log_event=measured,
                    origin_wall=origin_wall,
                    device=device,
                    rank=rank,
                    node_name=node_name,
                    job_id=job_id,
                    pipeline_stage=rank,
                    data_parallel_rank=0,
                    operation="pipeline_gradient_send",
                    category="Communication",
                    resources=["GPU", "NETWORK"],
                    iteration=iteration,
                    source=f"rank-{rank}",
                    destination=f"rank-{rank - 1}",
                    data_gb=activation_gb,
                    details={"phase": phase, "microbatch": microbatch},
                )
            trace_marker_from_context(
                context,
                job_id=job_id,
                operation="pipeline_microbatch_complete",
                category="Iteration",
                resources=["CPU", "GPU", "NETWORK"],
                iteration=iteration,
                pipeline_stage=rank,
                data_parallel_rank=0,
                details={"phase": phase, "microbatch": microbatch},
            )

        measure(
            lambda: optimizer.step(),
            logger=logger,
            log_event=measured,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=rank,
            data_parallel_rank=0,
            operation="pipeline_optimizer",
            category="Computation",
            resources=["GPU"],
            iteration=iteration,
            details={
                "phase": phase,
                "wait_reason": "waiting_for_all_pipeline_stages",
            },
        )

        if not measured:
            trace_marker_from_context(
                context,
                job_id=job_id,
                operation="pipeline_checkpoint_skipped",
                category="Checkpoint",
                resources=["CPU"],
                iteration=iteration,
                pipeline_stage=rank,
                data_parallel_rank=0,
                details={"reason": "warmup", "phase": phase},
            )
        elif args.checkpoint_every <= 0:
            trace_marker_from_context(
                context,
                job_id=job_id,
                operation="pipeline_checkpoint_skipped",
                category="Checkpoint",
                resources=["CPU"],
                iteration=iteration,
                pipeline_stage=rank,
                data_parallel_rank=0,
                details={"reason": "checkpoint_disabled", "phase": phase},
            )
        elif iteration % args.checkpoint_every != 0:
            trace_marker_from_context(
                context,
                job_id=job_id,
                operation="pipeline_checkpoint_skipped",
                category="Checkpoint",
                resources=["CPU"],
                iteration=iteration,
                pipeline_stage=rank,
                data_parallel_rank=0,
                details={
                    "reason": "not_due",
                    "checkpoint_every": args.checkpoint_every,
                    "phase": phase,
                },
            )
        else:
            checkpoint(
                logger=logger,
                origin_wall=origin_wall,
                device=device,
                rank=rank,
                node_name=node_name,
                job_id=job_id,
                pipeline_stage=rank,
                data_parallel_rank=0,
                iteration=iteration,
                mode=args.mode,
                checkpoint_mode=args.checkpoint_mode,
                checkpoint_tensor=checkpoint_tensor,
                bucket_uri=args.bucket_uri,
                output_dir=output_dir,
                background_threads=context["checkpoint_threads"],
                background_errors=context["checkpoint_errors"],
            )
        measure(
            lambda: barrier(device),
            logger=logger,
            log_event=True,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=rank,
            data_parallel_rank=0,
            operation="pipeline_iteration_barrier",
            category="Synchronization",
            resources=["GPU", "NETWORK"],
            iteration=iteration,
            details={"phase": phase},
        )
        trace_marker_from_context(
            context,
            job_id=job_id,
            operation="pipeline_iteration_complete",
            category="Iteration",
            resources=["CPU", "GPU", "NETWORK"],
            iteration=iteration,
            pipeline_stage=rank,
            data_parallel_rank=0,
            details={"phase": phase, "loop_index": loop_index},
        )


def write_summary(
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    backend: str,
    device: torch.device,
    logger: JsonlLogger,
    output_dir: Path,
    started_at: float,
    finished_at: float,
) -> None:
    model = build_model(args.hidden_size, args.layers, torch.device("cpu"))
    batch = batch_size_for_tensor(args.tensor_mb, args.hidden_size)
    summary = {
        "job_id": job_id_for(args.mode, args.checkpoint_mode),
        "mode": args.mode,
        "checkpoint_mode": args.checkpoint_mode,
        "rank": rank,
        "world_size": world_size,
        "node_name": args.node_name,
        "hostname": socket.gethostname(),
        "backend": backend,
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "microbatches": args.microbatches,
        "tensor_mb": args.tensor_mb,
        "tensor_gb": batch * args.hidden_size * 4 / GB,
        "checkpoint_mb": args.checkpoint_mb,
        "checkpoint_gb": args.checkpoint_mb * MB / GB,
        "checkpoint_every": args.checkpoint_every,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "gpu_memory_target_percent": args.gpu_memory_target_percent,
        "gpu_memory_reserve_safety_mb": args.gpu_memory_reserve_safety_mb,
        "gpu_memory": gpu_memory_snapshot(device),
        "cpu_memory": cpu_memory_snapshot(),
        "cpu_cores": os.cpu_count(),
        "model_parameter_gb": model_parameter_gb(model),
        "gradient_gb": model_parameter_gb(model),
        "event_count": len(logger.events),
        "output": str(args.output),
        "output_dir": str(output_dir),
        "run_id": args.run_id,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.checkpoint_mode = normalize_checkpoint_mode(args.checkpoint_mode)
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if args.microbatches <= 0:
        raise ValueError("--microbatches must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive")
    if args.layers <= 0:
        raise ValueError("--layers must be positive")
    if not 0.0 <= args.gpu_memory_target_percent <= 95.0:
        raise ValueError("--gpu-memory-target-percent must be between 0 and 95")
    if args.gpu_memory_reserve_safety_mb < 0:
        raise ValueError("--gpu-memory-reserve-safety-mb cannot be negative")

    started_at = time.time()
    rank, world_size, device, backend = init_distributed()
    node_name = args.node_name or socket.gethostname()
    logger = JsonlLogger(rank=rank)
    origin_wall = aligned_origin(rank, device)
    output_dir = args.output.parent
    context = {
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "backend": backend,
        "logger": logger,
        "origin_wall": origin_wall,
        "node_name": node_name,
        "output_dir": output_dir,
        "checkpoint_threads": [],
        "checkpoint_errors": [],
    }
    job_id = job_id_for(args.mode, args.checkpoint_mode)
    pipeline_stage = rank if args.mode == "pipeline_parallel" else 0
    data_parallel_rank = rank if args.mode == "data_parallel" else 0
    trace_marker_from_context(
        context,
        job_id=job_id,
        operation="benchmark_mode_start",
        category="Setup",
        resources=["CPU", "GPU", "NETWORK"],
        iteration=0,
        pipeline_stage=pipeline_stage,
        data_parallel_rank=data_parallel_rank,
        details={
            "mode": args.mode,
            "checkpoint_mode": args.checkpoint_mode,
            "backend": backend,
            "device": str(device),
            "world_size": world_size,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "microbatches": args.microbatches,
            "bucket_enabled": bool(args.bucket_uri),
        },
    )

    try:
        if args.mode == "data_parallel":
            run_data_parallel(args, context)
        else:
            run_pipeline_parallel(args, context)
        wait_for_background_checkpoints(
            context,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            iteration=args.iterations,
        )
        trace_marker_from_context(
            context,
            job_id=job_id,
            operation="benchmark_workload_complete",
            category="Iteration",
            resources=["CPU", "GPU", "NETWORK"],
            iteration=args.iterations,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            details={
                "mode": args.mode,
                "checkpoint_mode": args.checkpoint_mode,
                "wait_reason": "waiting_for_all_ranks_before_trace_write",
            },
        )
        measure(
            lambda: barrier(device),
            logger=logger,
            log_event=True,
            origin_wall=origin_wall,
            device=device,
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            operation="benchmark_final_barrier",
            category="Synchronization",
            resources=["GPU", "NETWORK"],
            iteration=args.iterations,
            details={
                "mode": args.mode,
                "checkpoint_mode": args.checkpoint_mode,
            },
        )
        trace_marker_from_context(
            context,
            job_id=job_id,
            operation="benchmark_trace_write_start",
            category="Filesystem",
            resources=["CPU", "LOCAL_DISK"],
            iteration=args.iterations,
            pipeline_stage=pipeline_stage,
            data_parallel_rank=data_parallel_rank,
            details={"output": str(args.output)},
        )
        logger.write_jsonl(args.output)
        log_status(
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            operation="benchmark_trace_write_complete",
            phase="end",
            iteration=args.iterations,
            details={"output": str(args.output), "event_count": len(logger.events)},
        )
        write_summary(
            args=args,
            rank=rank,
            world_size=world_size,
            backend=backend,
            device=device,
            logger=logger,
            output_dir=output_dir,
            started_at=started_at,
            finished_at=time.time(),
        )
        log_status(
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            operation="benchmark_summary_write_complete",
            phase="end",
            iteration=args.iterations,
            details={"output": str(args.summary_output)},
        )
    finally:
        log_status(
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            operation="benchmark_destroy_process_group",
            phase="start",
            iteration=args.iterations,
        )
        dist.destroy_process_group()
        log_status(
            rank=rank,
            node_name=node_name,
            job_id=job_id,
            operation="benchmark_destroy_process_group",
            phase="end",
            iteration=args.iterations,
        )


if __name__ == "__main__":
    main()
