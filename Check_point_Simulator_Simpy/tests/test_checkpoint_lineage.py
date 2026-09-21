"""Regression tests for checkpoint state across a training rollback.

A checkpoint belongs to the training timeline (epoch) that created it.  Once a
job rolls back, neither a completed future copy nor a late asynchronous write
from the abandoned timeline may be considered by recovery.
"""

from collections import Counter
from types import SimpleNamespace

import pytest
import simpy

from checkpointing.crossjob import CrossJobPeerStrategy
from checkpointing.tiered import CheckpointCopy, TieredCheckpointStrategy
from nodes.node import EventLogger
from run_scenario import CohortJobRuntime, CohortWorker
from simulation import SimPyBackend


def _copy(
    iteration: int,
    *,
    tier: str = "dram",
    owner_rank: int = 0,
) -> CheckpointCopy:
    return CheckpointCopy(
        owner_rank=owner_rank,
        location_rank=owner_rank,
        tier=tier,
        iteration=iteration,
        size_gb=1.0,
        completed_at=float(iteration),
    )


def _checkpoint_harness():
    env = simpy.Environment()
    backend = SimPyBackend(env)
    logger = EventLogger()
    worker = CohortWorker(
        job_id="job-a",
        rank=0,
        data_parallel_rank=0,
        pipeline_stage=0,
        physical_node="job-a-cohort-0",
        gpu=backend.resource(1),
        cpu=backend.priority_resource(1),
        represents=1,
        current_iteration=5,
        checkpoint_epoch=4,
    )
    cluster = SimpleNamespace(
        gpu_cpu_bandwidth_gbps=10.0,
        local_ssd_bandwidth_gbps=5.0,
        network_bandwidth_gbps=5.0,
        object_store_bandwidth_gbps=5.0,
    )
    strategy = TieredCheckpointStrategy(
        run_id=0,
        cluster=cluster,
        logger=logger,
        object_store=None,
        workers={0: worker},
        paired=False,
        persist_to_ssd=False,
        backend=backend,
    )
    strategy.stats = Counter()
    job = SimpleNamespace(
        job_id="job-a",
        checkpoint_gb=1.0,
        checkpoint_chunk_gb=0.25,
        pipeline_stages=1,
    )
    return env, backend, logger, strategy, job, worker


def test_local_copy_rejects_checkpoint_beyond_current_progress() -> None:
    worker = SimpleNamespace(current_iteration=5)
    strategy = SimpleNamespace(
        workers={0: worker},
        copies={(0, 0, "dram"): _copy(4)},
        backend=SimpleNamespace(now=12.0),
    )

    TieredCheckpointStrategy._put_copy(
        strategy,
        owner_rank=0,
        location_rank=0,
        tier="dram",
        iteration=6,
        size_gb=1.0,
    )

    # A late write from the abandoned timeline must not replace the valid copy.
    assert strategy.copies[(0, 0, "dram")].iteration == 4


def test_peer_piece_rejects_checkpoint_beyond_current_progress() -> None:
    worker = SimpleNamespace(rank=0, current_iteration=5)
    strategy = SimpleNamespace(peer_copies={})
    job = SimpleNamespace(job_id="job-a")
    piece = {
        "donor": "job-b-cohort-0",
        "donor_job": "job-b",
        "bytes": 0.5,
        "meta": {"piece_idx": 0, "parity": False},
    }

    CrossJobPeerStrategy._put_peer_piece(
        strategy,
        job,
        worker,
        iteration=6,
        piece=piece,
        total=2,
        data_k=2,
    )

    # This helper is also used by detached parity stragglers after their
    # foreground checkpoint process has returned.
    assert strategy.peer_copies == {}


def test_checkpoint_created_before_invalidation_keeps_explicit_epoch() -> None:
    """Creating a generator must capture the caller's timeline, not start-time."""
    env, backend, logger, strategy, job, worker = _checkpoint_harness()
    scheduled_epoch = worker.checkpoint_epoch
    process = backend.process(
        strategy.checkpoint(
            worker,
            job,
            iteration=5,
            checkpoint_epoch=scheduled_epoch,
        )
    )

    # Invalidate after scheduling but before SimPy first enters the generator.
    worker.checkpoint_epoch += 1
    worker.current_iteration = 2
    env.run()

    assert process.value is False
    assert strategy.copies == {}
    assert worker.active_checkpoint_gb == 0.0
    assert not [event for event in logger.events if event.category == "Checkpoint"]


def test_snapshot_waiting_for_lock_cannot_land_after_invalidation() -> None:
    env, backend, logger, strategy, job, worker = _checkpoint_harness()
    lock = strategy.checkpoint_locks[worker.rank]
    held = lock.request()
    assert held.triggered
    scheduled_epoch = worker.checkpoint_epoch
    process = backend.process(
        CrossJobPeerStrategy.snapshot(
            strategy,
            worker,
            job,
            iteration=5,
            checkpoint_epoch=scheduled_epoch,
        )
    )

    # Start the process and leave it queued behind the held checkpoint lock.
    env.run(until=backend.timeout(0))
    assert lock.queue

    worker.checkpoint_epoch += 1
    worker.current_iteration = 2
    lock.release(held)
    env.run()

    assert process.value is False
    assert strategy.copies == {}
    assert strategy.stats["snapshots"] == 0
    assert not [event for event in logger.events if event.category == "Checkpoint"]


def test_older_normal_peer_and_store_copy_cannot_replace_newer_copy() -> None:
    worker = SimpleNamespace(rank=0, current_iteration=10)
    job = SimpleNamespace(job_id="job-a")
    peer_key = ("job-a", 0, "job-b-cohort-0")
    store_key = ("job-a", 0)
    strategy = SimpleNamespace(
        peer_copies={
            peer_key: {
                "iteration": 8,
                "size_gb": 0.5,
                "donor_job": "job-b",
                "k": 2,
                "data_k": 2,
                "piece_idx": 0,
                "parity": False,
            }
        },
        store_copies={store_key: {"iteration": 8, "size_gb": 1.0}},
    )
    older_piece = {
        "donor": "job-b-cohort-0",
        "donor_job": "job-b",
        "bytes": 0.5,
        "meta": {"piece_idx": 0, "parity": False},
    }

    CrossJobPeerStrategy._put_peer_piece(
        strategy,
        job,
        worker,
        iteration=4,
        piece=older_piece,
        total=2,
        data_k=2,
    )
    stored = CrossJobPeerStrategy._put_store_copy(
        strategy,
        job,
        worker,
        iteration=4,
        size_gb=1.0,
    )

    assert strategy.peer_copies[peer_key]["iteration"] == 8
    assert strategy.peer_copies[peer_key]["parity"] is False
    assert stored is False
    assert strategy.store_copies[store_key]["iteration"] == 8


def test_completed_normal_peer_flush_cannot_overwrite_newer_donor_copy() -> None:
    """Exercise the regular registry-reserved peer path, not parity helpers."""

    class NormalPeerStrategy(CrossJobPeerStrategy):
        async_persist = False
        backstop = "owner_push"
        capacity_mode = False
        engine = "event"
        kpeers_by_job = {"job-a": 1}
        rates_by_job = {}
        slot_period = None
        store_mode = False

    env = simpy.Environment()
    backend = SimPyBackend(env)
    logger = EventLogger()
    cluster = SimpleNamespace(
        gpu_cpu_bandwidth_gbps=10.0,
        local_ssd_bandwidth_gbps=5.0,
        network_bandwidth_gbps=5.0,
        object_store_bandwidth_gbps=5.0,
    )
    owner = CohortWorker(
        job_id="job-a",
        rank=0,
        data_parallel_rank=0,
        pipeline_stage=0,
        physical_node="job-a-cohort-0",
        gpu=backend.resource(1),
        cpu=backend.priority_resource(1),
        represents=1,
        current_iteration=10,
    )
    donor = CohortWorker(
        job_id="job-b",
        rank=0,
        data_parallel_rank=0,
        pipeline_stage=0,
        physical_node="job-b-cohort-0",
        gpu=backend.resource(1),
        cpu=backend.priority_resource(1),
        represents=1,
        current_iteration=10,
    )
    strategy = NormalPeerStrategy(
        run_id=947_031,
        cluster=cluster,
        logger=logger,
        object_store=None,
        workers={0: owner},
        backend=backend,
    )
    strategy.register_worker(owner, "job-a")
    strategy.register_worker(donor, "job-b")
    key = ("job-a", 0, donor.name)
    strategy.peer_copies[key] = {
        "iteration": 8,
        "size_gb": 1.0,
        "donor_job": "job-b",
        "k": 1,
        "data_k": 1,
        "piece_idx": 0,
        "parity": False,
    }
    job = SimpleNamespace(
        job_id="job-a",
        pipeline_stages=1,
        checkpoint_chunk_gb=0.25,
    )

    process = backend.process(
        strategy._persist_flow(
            owner,
            job,
            iteration=4,
            generation=owner.failure_generation,
            shard_gb=1.0,
            checkpoint_group="job-a-iteration-4",
        )
    )
    env.run()

    assert process.value is True
    assert strategy.peer_copies[key]["iteration"] == 8
    assert strategy.peer_copies[key]["parity"] is False


def test_older_l3_drain_cannot_replace_newer_piece() -> None:
    env = simpy.Environment()
    backend = SimPyBackend(env)
    worker = SimpleNamespace(rank=0, current_iteration=10, checkpoint_epoch=3)
    donor = SimpleNamespace(
        name="job-b-cohort-0",
        failed=False,
        failure_generation=0,
    )
    job = SimpleNamespace(job_id="job-a")
    key = ("job-a", 0, 0)

    class DrainHarness:
        strategy_name = "crossjob_peer"
        store_stream_gbps = 10.0

        def __init__(self):
            self.backend = backend
            self.stats = Counter()
            self.l3_pieces = {
                key: {"iteration": 8, "size_gb": 0.5, "k": 1},
            }

        def _drain_start_delay(self, donors):
            del donors
            return 0.0

        def _node_caps(self, job_id):
            del job_id
            return SimpleNamespace(disk_w=10.0, nic_out=10.0)

        def _capacity_transfer(self, *args, **kwargs):
            del args, kwargs
            yield self.backend.timeout(0)
            return True

    strategy = DrainHarness()
    process = backend.process(
        CrossJobPeerStrategy._drain_wave_to_l3(
            strategy,
            worker,
            job,
            iteration=4,
            drain_info=[(donor, "job-b", 0, 0.5)],
            checkpoint_epoch=worker.checkpoint_epoch,
        )
    )
    env.run()

    assert process.value is True
    assert strategy.l3_pieces[key]["iteration"] == 8


def test_rollback_prunes_every_future_source_before_recovery_accounting() -> None:
    worker = SimpleNamespace(
        rank=0,
        current_iteration=7,
        checkpoint_epoch=2,
        failed=False,
    )
    strategy = object.__new__(CrossJobPeerStrategy)
    strategy.workers = {0: worker}
    strategy.copies = {
        (0, 0, "dram"): _copy(2),
        (0, 0, "ssd"): _copy(8, tier="ssd"),
    }
    strategy.peer_copies = {
        ("job-a", 0, "donor-a"): {
            "iteration": 9,
            "size_gb": 0.5,
            "donor_job": "job-b",
            "k": 1,
            "piece_idx": 0,
        },
        ("other-job", 0, "donor-a"): {
            "iteration": 9,
            "size_gb": 0.5,
            "donor_job": "job-b",
            "k": 1,
            "piece_idx": 0,
        },
    }
    strategy.store_copies = {
        ("job-a", 0): {"iteration": 10, "size_gb": 1.0},
        ("other-job", 0): {"iteration": 10, "size_gb": 1.0},
    }
    strategy.l3_pieces = {
        ("job-a", 0, 0): {"iteration": 11, "size_gb": 0.5, "k": 1},
        ("other-job", 0, 0): {
            "iteration": 11,
            "size_gb": 0.5,
            "k": 1,
        },
    }
    strategy._last_flush_started = {0: 10}
    strategy.registry = SimpleNamespace(slots={})

    runtime = SimpleNamespace(
        iterations=20,
        it=7,
        workers={0: worker},
        checkpoint_strategy=strategy,
        config=SimpleNamespace(job_id="job-a"),
        last_store_it=6,
        lost_iters=[],
        lost_by_tier={},
    )

    CohortJobRuntime._invalidate_checkpoint_timeline(runtime)
    CohortJobRuntime._set_iteration(runtime, 3, rollback=True)
    restored_iteration, tier = CohortJobRuntime._peek_restore(runtime, worker)
    CohortJobRuntime._book_loss(
        runtime,
        tier,
        runtime.it - max(restored_iteration, 0),
    )

    assert runtime.it == 3
    assert worker.current_iteration == 3
    assert worker.checkpoint_epoch == 3
    assert strategy._last_flush_started[0] == 3
    assert restored_iteration == 2
    assert restored_iteration <= runtime.it
    assert runtime.lost_iters == [1]
    assert all(copy.iteration <= runtime.it for copy in strategy.copies.values())
    assert not [
        meta
        for (job_id, *_), meta in strategy.peer_copies.items()
        if job_id == "job-a" and meta["iteration"] > runtime.it
    ]
    assert ("other-job", 0, "donor-a") in strategy.peer_copies
    assert ("other-job", 0) in strategy.store_copies
    assert ("other-job", 0, 0) in strategy.l3_pieces


def test_loss_accounting_rejects_a_future_restore_iteration() -> None:
    runtime = SimpleNamespace(
        config=SimpleNamespace(job_id="job-a"),
        it=5,
        lost_iters=[],
        lost_by_tier={},
    )

    with pytest.raises(AssertionError, match="future checkpoint selected"):
        CohortJobRuntime._book_loss(runtime, "store", runtime.it - 6)

    assert runtime.lost_iters == []
    assert runtime.lost_by_tier == {}


def test_common_recovery_plan_prunes_above_slowest_cohort_target() -> None:
    workers = {
        rank: SimpleNamespace(
            rank=rank,
            current_iteration=10,
            checkpoint_epoch=1,
            failed=False,
        )
        for rank in range(2)
    }
    strategy = object.__new__(CrossJobPeerStrategy)
    strategy.workers = workers
    strategy.copies = {
        (0, 0, "ssd"): _copy(5, tier="ssd"),
        (1, 1, "dram"): _copy(5, owner_rank=1),
    }
    strategy.peer_copies = {}
    # Rank 0 initially prefers this newer durable copy. Once rank 1 forces the
    # synchronized target down to 5, it is a future-timeline copy and must no
    # longer be returned by the next planning pass.
    strategy.store_copies = {
        ("job-a", 0): {"iteration": 8, "size_gb": 1.0},
    }
    strategy.l3_pieces = {}
    strategy._last_flush_started = {}
    strategy.registry = SimpleNamespace(slots={})

    runtime = object.__new__(CohortJobRuntime)
    runtime.it = 10
    runtime.workers = workers
    runtime.checkpoint_strategy = strategy
    runtime.config = SimpleNamespace(job_id="job-a")

    target, peeks = runtime._common_recovery_plan()

    assert target == 5
    assert {iteration for iteration, _tier in peeks.values()} == {5}
    assert all(iteration <= target for iteration, _tier in peeks.values())


def test_single_failure_invalidates_job_epoch_before_restart_wait() -> None:
    """Queued checkpoints must become stale before recovery advances time."""

    class Strategy:
        capacity = None
        engine = "event"

        def drop_local_copies(self, rank, *, fraction, rng):
            del rank, fraction, rng

    env = simpy.Environment()
    workers = {
        rank: SimpleNamespace(
            rank=rank,
            name=f"job-a-cohort-{rank}",
            represents=1,
            failed=False,
            failure_generation=0,
            checkpoint_epoch=4,
        )
        for rank in range(2)
    }
    runtime = object.__new__(CohortJobRuntime)
    runtime.backend = SimPyBackend(env)
    runtime.workers = workers
    runtime.checkpoint_strategy = Strategy()
    runtime.rng = SimpleNamespace(choice=lambda choices: choices[0])
    runtime.failure_types = Counter()
    runtime.failures = SimpleNamespace(
        process_restart_seconds=1.0,
        node_restart_seconds=1.0,
        spot_restart_seconds=1.0,
    )
    runtime.all_strategies = []
    runtime.baseline = None

    failure = runtime._single_node_failure("process")
    next(failure)  # Stops at the restart timeout yielded after marking failure.

    assert [worker.checkpoint_epoch for worker in workers.values()] == [5, 5]
