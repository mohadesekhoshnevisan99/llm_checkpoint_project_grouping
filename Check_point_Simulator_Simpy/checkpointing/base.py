from __future__ import annotations

from typing import Any, Iterable, Protocol

# ---------------------------------------------------------------------------
# FABRIC IDENTITY (sim_validation_protocol.md CL-012.D S1/S2, implemented by
# CL-014). A "fabric" is one physical interconnect: a NIC + its link + its
# switch port. Two flows on the SAME fabric label share a wire; two flows on
# DIFFERENT labels are physically disjoint (distinct PCIe endpoint, distinct
# ASIC, distinct link) and cannot slow each other down through the wire.
#
# Everything is on DEFAULT_FABRIC unless a scenario says otherwise, so an
# undeclared model is exactly today's single-fabric model. Declared per class
# under `rates:`
#   nic_fabric:   the fabric this class's TRAINING COLLECTIVE rides
#   ckpt_fabric:  the fabric this class's CHECKPOINT STREAMS ride
#                 (defaults to nic_fabric: one wire unless told otherwise)
# ---------------------------------------------------------------------------
DEFAULT_FABRIC = "default"


def ckpt_fabric_of(rates_by_job: dict, job_id: str) -> str:
    """The fabric job `job_id`'s checkpoint streams ride.

    `ckpt_fabric` wins; absent, the job's checkpoint traffic shares the same
    wire as its collective (`nic_fabric`); absent both, the single default
    fabric — i.e. the pre-CL-014 world where every byte shares one wire."""
    rates = rates_by_job.get(job_id) or {}
    return str(rates.get("ckpt_fabric")
               or rates.get("nic_fabric")
               or DEFAULT_FABRIC)


def train_fabric_of(rates: dict | None) -> str:
    """The fabric a class's training collective rides (class `rates:` block)."""
    return str((rates or {}).get("nic_fabric") or DEFAULT_FABRIC)


def bump_network_tasks(workers: Iterable[Any], fabric: str, delta: int,
                       rate_gbps: float = 0.0) -> None:
    """Register/withdraw one in-flight checkpoint NETWORK transfer on `fabric`.

    Maintains BOTH counters at once so they can never disagree:
      * `checkpoint_network_tasks` — the fabric-blind aggregate the original
        count-based coupling consumes (unchanged, still authoritative when
        `model.fabric_aware_coupling` is off);
      * `checkpoint_network_tasks_by_fabric` — the same count split by fabric,
        which is what a fabric-scoped consumer reads.
    A fabric whose count returns to 0 is dropped from the dict, so an
    all-default run leaves `{}` / `{"default": n}` and nothing else.

    CL-016 (CL-012.D S3): `rate_gbps` is the transfer's OFFERED rate — the rate
    the stream would move at absent the collective, i.e. its demand cap bounded
    by the physical caps along its path (the shaped tc rate, the measured TCP
    ceiling, a donor's NIC ingress — all measured constants the scenario already
    declares; NOTHING here is fitted). It is mirrored into a third ledger,
    `checkpoint_network_demand_by_fabric` (GB/s), which the intensity-based
    coupling consumes. Pure bookkeeping: a dict update, no RNG draws, no
    timeouts — every flag-off run is byte-identical. The caller passes the SAME
    rate at +1 and -1 (both bumps read one local variable), so entries reverse
    exactly, and the entry is dropped the moment its fabric's COUNT returns to
    0, so float residue can never accumulate. Workers that predate the ledger
    (older test mocks) are tolerated: no demand dict => counts only."""
    for worker in workers:
        worker.checkpoint_network_tasks += delta
        by_fabric = worker.checkpoint_network_tasks_by_fabric
        count = by_fabric.get(fabric, 0) + delta
        if count:
            by_fabric[fabric] = count
        else:
            by_fabric.pop(fabric, None)
        demand = getattr(worker, "checkpoint_network_demand_by_fabric", None)
        if demand is None:
            continue
        if count:
            demand[fabric] = demand.get(fabric, 0.0) + delta * rate_gbps
        else:
            demand.pop(fabric, None)


class CheckpointWorker(Protocol):
    job_id: str
    rank: int
    data_parallel_rank: int
    pipeline_stage: int
    physical_node: str
    gpu: Any
    cpu: Any
    checkpoint_uploads: int
    checkpoint_network_tasks: int
    # same count as above, keyed by the fabric each transfer rides (CL-014)
    checkpoint_network_tasks_by_fabric: dict[str, int]
    # aggregate OFFERED rate (GB/s) of those same in-flight transfers, keyed by
    # fabric (CL-016 / CL-012.D S3) — consumed only by the intensity coupling
    checkpoint_network_demand_by_fabric: dict[str, float]
    training_network_tasks: int
    failed: bool
    failure_generation: int
    current_iteration: int
    checkpoint_epoch: int
    failure_type: str | None
    recovered_event: Any | None
    active_checkpoint_gb: float

    @property
    def name(self) -> str: ...
