from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import simpy

from nodes.node import EventLogger, Link, ObjectStore
from nodes.training_node import TrainingNode
from simulation import run_simulation

# ============================================================================
# Simulation configuration
#
# Units:
#   memory/data      GB
#   bandwidth       GB/s
#   simulated time  seconds
# ============================================================================

RANDOM_SEED = 17
RESULTS_DIR = Path("results")
EVENT_LOG_PATH = RESULTS_DIR / "simulation_log.jsonl"
CONFIG_PATH = RESULTS_DIR / "simulation_config.json"

# Distributed job
WORLD_SIZE = 4
TOTAL_ITERATIONS = 10
CHECKPOINT_EVERY = 1
FAILURE_EVERY = 2

# Arbitrary same-node CPU/GPU contention assumptions. When independent CPU
# and GPU operations overlap, each progresses more slowly by this multiplier.
CPU_SLOWDOWN_WHEN_GPU_ACTIVE = 1.25
GPU_SLOWDOWN_WHEN_CPU_ACTIVE = 1.15
CONTENTION_QUANTUM_SECONDS = 0.05

# A large BF16 model that still leaves room inside one A100 80GB for activations.
MODEL_NAME = "Dense-13B-BF16"
MODEL_PARAMETERS_BILLIONS = 13.0
BYTES_PER_PARAMETER = 2
MODEL_WEIGHTS_GB = MODEL_PARAMETERS_BILLIONS * BYTES_PER_PARAMETER
GRADIENT_SIZE_GB = MODEL_WEIGHTS_GB
CHECKPOINT_SIZE_GB = 30.0  # model weights plus compact trainer metadata

# One identical A100-class worker per rank.
CPU_CORES_PER_NODE = 1
GPUS_PER_NODE = 1
GPU_MEMORY_GB = 80.0
GPU_CPU_BANDWIDTH_GBPS = 31.5

# Host-memory and SSD values are server assumptions, not properties of A100.
HOST_DRAM_GB = 512.0
LOCAL_SSD_GB = 2_048.0
HOST_DRAM_READ_GBPS = 150.0
HOST_DRAM_WRITE_GBPS = 150.0
LOCAL_SSD_READ_GBPS = 7.0
LOCAL_SSD_WRITE_GBPS = 6.0

# 200 Gbit/s inter-node network and an 8 GB/s object-store path.
NETWORK_BANDWIDTH_GBPS = 25.0
OBJECT_STORE_BANDWIDTH_GBPS = 8.0

# Above this size, model rank 0, ranks selected to fail, and the remaining
# synchronized worker cohort explicitly. This preserves divergent timing
# without creating O(world_size**2) transfer processes.
DETAILED_WORLD_SIZE_LIMIT = 256

# Simplified per-rank compute times for one training iteration.
FORWARD_SECONDS = 5.0
BACKWARD_SECONDS = 9.0
OPTIMIZER_SECONDS = 1.0

FAILURE_RESTART_SECONDS = {
    "process": 3.0,
    "node": 8.0,
    "spot": 15.0,
}


@dataclass(frozen=True, slots=True)
class FailureSpec:
    iteration: int
    rank: int
    failure_type: str
    stage: str
    progress: float


def build_failure_schedule(
    *,
    world_size: int,
    rng: random.Random,
) -> dict[int, FailureSpec]:
    """Choose reproducible failures at varied points in local GPU work."""

    schedule: dict[int, FailureSpec] = {}
    for iteration in range(FAILURE_EVERY, TOTAL_ITERATIONS + 1, FAILURE_EVERY):
        schedule[iteration] = FailureSpec(
            iteration=iteration,
            rank=(
                rng.randrange(1, world_size)
                if world_size > 1
                else 0
            ),
            failure_type=rng.choice(("process", "node", "spot")),
            stage=rng.choice(("forward", "backward")),
            progress=round(rng.uniform(0.15, 0.85), 3),
        )
    return schedule


# ============================================================================
# Construction
# ============================================================================


def build_cluster(
    env: simpy.Environment,
    logger: EventLogger,
    object_store: ObjectStore,
    *,
    world_size: int,
    aggregate: bool,
    explicit_failure_ranks: set[int],
    verbose: bool,
) -> list[TrainingNode]:
    def create_node(rank: int) -> TrainingNode:
        return TrainingNode(
            env,
            rank=rank,
            world_size=world_size,
            cpu_cores=CPU_CORES_PER_NODE,
            gpu_count=GPUS_PER_NODE,
            gpu_memory_gb=GPU_MEMORY_GB,
            dram_gb=HOST_DRAM_GB,
            ssd_gb=LOCAL_SSD_GB,
            model_weights_gb=MODEL_WEIGHTS_GB,
            gradient_size_gb=GRADIENT_SIZE_GB,
            checkpoint_size_gb=CHECKPOINT_SIZE_GB,
            gpu_cpu_bandwidth_gbps=GPU_CPU_BANDWIDTH_GBPS,
            dram_read_gbps=HOST_DRAM_READ_GBPS,
            dram_write_gbps=HOST_DRAM_WRITE_GBPS,
            ssd_read_gbps=LOCAL_SSD_READ_GBPS,
            ssd_write_gbps=LOCAL_SSD_WRITE_GBPS,
            logger=logger,
            object_store=object_store,
            cpu_slowdown_when_gpu_active=CPU_SLOWDOWN_WHEN_GPU_ACTIVE,
            gpu_slowdown_when_cpu_active=GPU_SLOWDOWN_WHEN_CPU_ACTIVE,
            contention_quantum_seconds=CONTENTION_QUANTUM_SECONDS,
            verbose=verbose,
        )

    if aggregate:
        nodes = [create_node(0)]
        failed_workers = sorted(
            rank for rank in explicit_failure_ranks if rank != 0
        )
        cohort_count = world_size - 1 - len(failed_workers)
        if cohort_count > 0:
            cohort_rank = next(
                rank
                for rank in range(1, world_size)
                if rank not in explicit_failure_ranks
            )
            cohort = create_node(cohort_rank)
            cohort.name = f"ranks-1-{world_size - 1}"
            cohort.represented_rank_count = cohort_count
            nodes.append(cohort)
        nodes.extend(create_node(rank) for rank in failed_workers)
        return nodes

    nodes = [create_node(rank) for rank in range(world_size)]

    # Ring topology: 0--1--2--3--0.
    for rank in range(world_size):
        next_rank = (rank + 1) % world_size
        if rank == next_rank:
            continue
        node = nodes[rank]
        peer = nodes[next_rank]
        link = Link(
            env,
            node_a=node,
            node_b=peer,
            bandwidth_gbps=NETWORK_BANDWIDTH_GBPS,
            name=f"ring-{rank}-{next_rank}",
        )
        node.connect(peer, link)

    return nodes


# ============================================================================
# Orchestration helpers
# ============================================================================


def run_parallel(
    env: simpy.Environment,
    generators: Iterable,
):
    processes = [env.process(generator) for generator in generators]
    return env.all_of(processes)


def ring_all_reduce(
    env: simpy.Environment,
    nodes: list[TrainingNode],
    *,
    iteration: int,
    world_size: int,
    aggregate: bool,
):
    # Ring all-reduce sends one Nth of the gradient on each of N-1
    # reduce-scatter steps and N-1 all-gather steps.
    if world_size == 1:
        return

    shard_gb = GRADIENT_SIZE_GB / world_size

    for phase in ("reduce_scatter", "all_gather"):
        if aggregate:
            phase_duration = (
                (world_size - 1)
                * shard_gb
                / NETWORK_BANDWIDTH_GBPS
            )
            yield run_parallel(
                env,
                [
                    node.aggregate_gradient_exchange(
                        iteration=iteration,
                        phase=phase,
                        duration=phase_duration,
                        shard_gb=shard_gb,
                        logical_world_size=world_size,
                    )
                    for node in nodes
                ],
            )
            continue

        for step in range(world_size - 1):
            transfers = []
            for node in nodes:
                destination_rank = (node.rank + 1) % world_size
                transfers.append(
                    node.transfer_gradient(
                        destination_rank=destination_rank,
                        iteration=iteration,
                        phase=phase,
                        step=step,
                        data_gb=shard_gb,
                    )
                )
            yield run_parallel(env, transfers)


def run_compute_stage(
    env: simpy.Environment,
    node: TrainingNode,
    *,
    iteration: int,
    stage: str,
    duration: float,
    failure: FailureSpec | None,
    verbose: bool,
):
    operation_method = (
        node.forward_pass if stage == "forward" else node.backward_pass
    )
    should_fail = (
        failure is not None
        and failure.rank == node.rank
        and failure.stage == stage
        and node.name == f"rank-{failure.rank}"
    )
    if not should_fail:
        yield env.process(
            operation_method(iteration=iteration, duration=duration)
        )
        return

    started = env.event()
    operation = env.process(
        operation_method(
            iteration=iteration,
            duration=duration,
            started_event=started,
        )
    )
    yield started
    yield env.timeout(duration * failure.progress)
    if operation.is_alive:
        operation.interrupt(
            f"injected {failure.failure_type} failure during {stage}"
        )
    try:
        yield operation
    except simpy.Interrupt:
        pass

    if verbose:
        print(
            f"*** {failure.failure_type} failure on rank-{failure.rank} "
            f"during iteration {iteration} {stage} "
            f"({failure.progress:.0%}) ***"
        )

    yield env.process(
        node.simulate_failure(
            failure_type=failure.failure_type,
            restart_duration=FAILURE_RESTART_SECONDS[failure.failure_type],
            iteration=iteration,
            failure_stage=stage,
            failure_progress=failure.progress,
        )
    )
    yield env.process(node.recover_from_object_store(owner_rank=0))

    # Recovery loads the previous completed checkpoint. A backward failure
    # must therefore replay forward before retrying backward; a forward
    # failure only needs to restart forward. Healthy ranks may already be
    # waiting at the next all-reduce synchronization point.
    if stage == "backward":
        yield env.process(
            node.forward_pass(
                iteration=iteration,
                duration=FORWARD_SECONDS,
            )
        )
    yield env.process(
        operation_method(iteration=iteration, duration=duration)
    )


def local_compute_to_sync(
    env: simpy.Environment,
    node: TrainingNode,
    *,
    iteration: int,
    failure: FailureSpec | None,
    verbose: bool,
):
    yield env.process(
        run_compute_stage(
            env,
            node,
            iteration=iteration,
            stage="forward",
            duration=FORWARD_SECONDS,
            failure=failure,
            verbose=verbose,
        )
    )
    yield env.process(
        run_compute_stage(
            env,
            node,
            iteration=iteration,
            stage="backward",
            duration=BACKWARD_SECONDS,
            failure=failure,
            verbose=verbose,
        )
    )
    return env.now


# ============================================================================
# Main simulation policy
# ============================================================================


def training_controller(
    env: simpy.Environment,
    nodes: list[TrainingNode],
    object_store: ObjectStore,
    failure_schedule: dict[int, FailureSpec],
    *,
    world_size: int,
    aggregate: bool,
    verbose: bool,
):
    checkpoint_processes: list[simpy.Process] = []
    rank_zero = nodes[0]

    # A completed iteration-0 checkpoint guarantees that the first failure
    # always has a valid recovery point.
    initial_staged = env.event()
    initial_checkpoint = env.process(
        rank_zero.checkpoint_pipeline(iteration=0, staged_event=initial_staged)
    )
    yield initial_staged
    yield initial_checkpoint

    def optimizer_checkpoint_and_next_compute(
        node: TrainingNode,
        *,
        completed_iteration: int,
        next_iteration: int | None,
    ):
        yield env.process(
            node.optimizer_step(
                iteration=completed_iteration,
                duration=OPTIMIZER_SECONDS,
            )
        )

        # Checkpointing is rank-0-local. Starting it here lets every other
        # rank proceed immediately toward the next all-reduce dependency.
        if (
            node is rank_zero
            and completed_iteration % CHECKPOINT_EVERY == 0
        ):
            staged = env.event()
            checkpoint_process = env.process(
                rank_zero.checkpoint_pipeline(
                    iteration=completed_iteration,
                    staged_event=staged,
                )
            )
            checkpoint_processes.append(checkpoint_process)
            # This blocks only rank 0 until its GPU snapshot reaches DRAM.
            # Other rank processes are already advancing independently.
            yield staged

        if next_iteration is not None:
            yield env.process(
                local_compute_to_sync(
                    env,
                    node,
                    iteration=next_iteration,
                    failure=failure_schedule.get(next_iteration),
                    verbose=verbose,
                )
            )

    local_processes = [
        env.process(
            local_compute_to_sync(
                env,
                node,
                iteration=1,
                failure=failure_schedule.get(1),
                verbose=verbose,
            )
        )
        for node in nodes
    ]

    for iteration in range(1, TOTAL_ITERATIONS + 1):
        # Nodes arrive independently. Early arrivals remain idle until every
        # rank reaches the collective communication dependency.
        yield env.all_of(local_processes)

        if verbose:
            print(
                f"\n=== iteration {iteration}: all-reduce synchronization "
                f"(simulation time {env.now:.3f}) ==="
            )

        yield env.process(
            ring_all_reduce(
                env,
                nodes,
                iteration=iteration,
                world_size=world_size,
                aggregate=aggregate,
            )
        )

        next_iteration = (
            iteration + 1 if iteration < TOTAL_ITERATIONS else None
        )
        local_processes = [
            env.process(
                optimizer_checkpoint_and_next_compute(
                    node,
                    completed_iteration=iteration,
                    next_iteration=next_iteration,
                )
            )
            for node in nodes
        ]

    # Complete the final optimizer steps and asynchronous checkpoint writes.
    yield env.all_of(local_processes)
    if checkpoint_processes:
        yield env.all_of(checkpoint_processes)

    latest = object_store.latest_checkpoint(owner_rank=0)
    if latest is None or int(latest["iteration"]) != TOTAL_ITERATIONS:
        raise RuntimeError("Final object-store checkpoint was not completed")


# ============================================================================
# Entrypoint
# ============================================================================


def simulation_config(
    *,
    world_size: int,
    execution_mode: str,
    failure_schedule: dict[int, FailureSpec],
    simulated_node_objects: int,
) -> dict:
    return {
        "random_seed": RANDOM_SEED,
        "world_size": world_size,
        "execution_mode": execution_mode,
        "simulated_node_objects": simulated_node_objects,
        "total_iterations": TOTAL_ITERATIONS,
        "checkpoint_every": CHECKPOINT_EVERY,
        "failure_every": FAILURE_EVERY,
        "failure_schedule": [
            asdict(failure_schedule[iteration])
            for iteration in sorted(failure_schedule)
        ],
        "model": {
            "name": MODEL_NAME,
            "parameters_billions": MODEL_PARAMETERS_BILLIONS,
            "bytes_per_parameter": BYTES_PER_PARAMETER,
            "weights_gb": MODEL_WEIGHTS_GB,
            "gradient_gb": GRADIENT_SIZE_GB,
            "checkpoint_gb": CHECKPOINT_SIZE_GB,
        },
        "node": {
            "cpu_cores": CPU_CORES_PER_NODE,
            "gpus": GPUS_PER_NODE,
            "gpu_memory_gb": GPU_MEMORY_GB,
            "gpu_cpu_bandwidth_gbps": GPU_CPU_BANDWIDTH_GBPS,
            "host_dram_gb": HOST_DRAM_GB,
            "local_ssd_gb": LOCAL_SSD_GB,
        },
        "network_bandwidth_gbps": NETWORK_BANDWIDTH_GBPS,
        "object_store_bandwidth_gbps": OBJECT_STORE_BANDWIDTH_GBPS,
        "contention": {
            "model": "independent CPU/GPU overlap",
            "cpu_slowdown_when_gpu_active": CPU_SLOWDOWN_WHEN_GPU_ACTIVE,
            "gpu_slowdown_when_cpu_active": GPU_SLOWDOWN_WHEN_CPU_ACTIVE,
            "quantum_seconds": CONTENTION_QUANTUM_SECONDS,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate synchronous distributed training with SimPy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Run jobs from a TOML configuration file",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=WORLD_SIZE,
        help="Number of logical training ranks (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "detailed", "aggregate"),
        default="auto",
        help=(
            "Detailed creates every rank and transfer; aggregate models "
            "synchronized cohorts analytically (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-operation progress output",
    )
    args = parser.parse_args()
    if args.world_size <= 0:
        parser.error("--world-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.config is not None:
        configured_main(args.config, quiet=args.quiet)
        return

    execution_mode = args.mode
    if execution_mode == "auto":
        execution_mode = (
            "aggregate"
            if args.world_size > DETAILED_WORLD_SIZE_LIMIT
            else "detailed"
        )
    aggregate = execution_mode == "aggregate"
    rng = random.Random(RANDOM_SEED)
    failure_schedule = build_failure_schedule(
        world_size=args.world_size,
        rng=rng,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = EventLogger()
    env = simpy.Environment()
    object_store = ObjectStore(
        env,
        bandwidth_gbps=OBJECT_STORE_BANDWIDTH_GBPS,
        io_concurrency=args.world_size,
    )
    nodes = build_cluster(
        env,
        logger,
        object_store,
        world_size=args.world_size,
        aggregate=aggregate,
        explicit_failure_ranks={
            failure.rank for failure in failure_schedule.values()
        },
        verbose=not args.quiet,
    )

    env.process(
        training_controller(
            env,
            nodes,
            object_store,
            failure_schedule,
            world_size=args.world_size,
            aggregate=aggregate,
            verbose=not args.quiet,
        )
    )
    env.run()

    logger.write_jsonl(EVENT_LOG_PATH)
    CONFIG_PATH.write_text(
        json.dumps(
            simulation_config(
                world_size=args.world_size,
                execution_mode=execution_mode,
                failure_schedule=failure_schedule,
                simulated_node_objects=len(nodes),
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    latest = object_store.latest_checkpoint(owner_rank=0)
    print("\nSimulation complete")
    print(f"  logical ranks:  {args.world_size}")
    print(f"  mode:           {execution_mode}")
    print(f"  simulated time: {env.now:.3f} seconds")
    print(f"  events:         {len(logger.events)}")
    print(f"  event log:      {EVENT_LOG_PATH}")
    print(f"  config:         {CONFIG_PATH}")
    print(f"  final checkpoint iteration: {latest['iteration'] if latest else 'none'}")


def configured_main(config_path: Path, *, quiet: bool) -> None:
    result = run_simulation(
        config_path,
        results_dir=RESULTS_DIR,
        quiet=quiet,
        write_artifacts=True,
    )

    print("\nConfigured simulation complete")
    print(f"  jobs:           {len(result.config.jobs)}")
    print(f"  logical ranks:  {result.logical_ranks}")
    print(f"  physical nodes: {result.physical_nodes}")
    print(f"  simulated time: {result.simulated_time:.3f} seconds")
    print(f"  events:         {result.event_count}")
    print(f"  event log:      {result.event_log_path}")
    print(f"  config:         {result.config_output_path}")


if __name__ == "__main__":
    main()
