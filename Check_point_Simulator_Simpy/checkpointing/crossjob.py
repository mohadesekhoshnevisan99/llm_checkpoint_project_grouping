"""Cross-job peer-sharing checkpoint strategy (see docs/CROSSJOB_GAPS.md).

A flushing rank spreads its checkpoint shard across k donor nodes belonging to OTHER
jobs (the real system's peerd behavior: reserve -> parallel stream -> fallback local).
Donation is endogenous: a node refuses hosting while its own job is flushing, and holds
at most `hosted_cap` concurrent shards. A run-wide DonorRegistry is shared across all
jobs' strategy instances.

v1 modeling choices (kept simple, mirrors the measured hardware behavior):
- the k parallel donor writes are modeled as one transfer at effective bandwidth
  min(network_bandwidth, k * local_ssd_bandwidth) (equal shards; slowest-donor gating
  and speed-weighted shards can come later);
- peer copies are tracked separately from the base class's local copies (ranks are
  only unique within a job, so peer copies are keyed by (job, rank, donor name));
- kpeers/phase come from class attributes set by the scenario driver (JobConfig stays
  untouched); per-job overrides via CrossJobPeerStrategy.kpeers_by_job[job_id].

Author: Sam's agent — new module only; base-class internals reused, not modified.
"""
from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .base import CheckpointWorker, bump_network_tasks
from .capacity import (BYTE_EPS, RES_DISK, RES_IN, RES_OUT, CapacityRegistry,
                       NodeCaps)
from .tiered import BilledCopies, TieredCheckpointStrategy

STORE_NODE = "__store__"

# --- Centralized peer-matching tunables (Mark meeting 2026-07-17, D8/D9) ------
# Mark: "the sharing should be done globally ... we need a peer matching
# algorithm" (D8), replacing the pre-centralized daemon's random probing.
# SPREAD_WEIGHT — refinement (a): how strongly the board avoids concentrating one
#   flusher's k shards in a single DONOR JOB. Round-robin over donor jobs is the
#   actual mechanism; this weight only tunes the frac-load fallback ordering.
# PROBE_FANOUT_MIN — the bounded random-candidate budget of the OLD daemon
#   (Mark: "random ... 8-probe"): limited visibility, no global load view. This
#   is the honest baseline the AvailabilityBoard is A/B'd against, NOT a strawman.
SPREAD_WEIGHT = 1.0
PROBE_FANOUT_MIN = 8
# D9 stage-2 phasing: a donor is "in its receive window" (safe to write to now)
# when its OWN next flush is at least WINDOW_GUARD_S away. window_bias partitions
# candidates on this first, then falls back to least-loaded — so a matched daemon
# effectively honours "write to me in this window."
WINDOW_GUARD_S = 3.0
# Background-QoS weight for donor-drain flows in the max-min allocator. A drain
# reads the donor's SSD and shares that disk (+ NIC) with FOREGROUND peer-flush
# writes landing on the same donor; at weight 1 it would take an equal share and
# slightly stretch those foreground flushes (measured on mega at high donor
# duty). A small weight makes the drain a true background flow: it yields the
# contended donor disk to foreground flushes (which keep ~1/(1+w) of the pipe)
# and only SLOWS ITSELF, while an uncontended drain still runs at its full demand
# (progressive filling lets a lone flow reach its demand cap regardless of
# weight). It never speeds a drain past its cap, so I11 stays green.
DRAIN_QOS_WEIGHT = 0.05


def resolve_engine(class_default: str | None = None) -> str:
    """Pick the capacity engine: env var SIM_ENGINE (quick A/B) wins, else the
    scenario-set class default, else 'event'. Only 'event' or 'polling'."""
    val = os.environ.get("SIM_ENGINE") or class_default or "event"
    val = str(val).strip().lower()
    if val not in ("event", "polling"):
        raise ValueError(f"SIM engine must be 'event' or 'polling', got {val!r}")
    return val


@dataclass(eq=False)          # identity eq/hash: slots are mutable and used as
class _DonorSlot:             # dict keys in the matched/probe reservation paths
    worker: Any
    job_id: str
    flushing: bool = False
    hosted: int = 0
    represents: int = 1          # ranks this slot stands for (cohort mode)
    available: bool = True       # False before the job ARRIVES (Stage 3): the
                                 # nodes exist but cannot donate yet. Default True
                                 # => no-op for every non-churn scenario.
    # L1.5 peer-DRAM tier (l15_peer_dram_spec.md): daemon-owned DRAM buffer
    # occupancy. Unlike `hosted` (a transient per-WAVE write reservation,
    # released at wave end), a DRAM reservation persists from the wave until
    # the piece DEMOTES to this host's SSD (or is lost with the host).
    dram_hosted: int = 0         # resident dram-tier pieces (shard units)
    dram_gb: float = 0.0         # resident dram-tier bytes (GB)

    @property
    def shard_capacity(self) -> int:
        return 2 * self.represents           # hosted_cap per represented rank


@dataclass
class DonorRegistry:
    """Run-wide donor pool shared by every job's strategy instance — the
    cluster-level **AvailabilityBoard** of Mark's D8. Each daemon publishes its
    state (hosting slots free/busy via ``hosted``, current disk duty implied by
    ``hosted``, job phase via ``flushing`` + the strategy's slot table) here as it
    changes; this is simulation, so publishing is a direct field update — no real
    network. The controller MATCHES a flusher's wave to donors globally instead of
    the flusher probing k random candidates:

      matching="registry"  centralized least-loaded-first with a global view, plus
                           optional refinements — (a) spread a wave across DONOR
                           JOBS (``spread_jobs``), (b) prefer donors whose own
                           next-flush window is far (``window_bias``, the D9
                           phasing hook). Refinements OFF => byte-identical to the
                           original least-loaded reserve (hand-check math intact).
      matching="probe"     the pre-centralized daemon protocol: a BOUNDED random
                           candidate probe (no global load view). Kept selectable
                           so the centralized board is measured against it (the
                           A/B IS the evaluation — Mark expects the comparison).
    """

    hosted_cap: int = 2
    slots: dict[str, _DonorSlot] = field(default_factory=dict)
    allowed_donors_by_job: dict[str, frozenset[str]] = field(default_factory=dict)
    refusals: Counter = field(default_factory=Counter)
    rng: Any = None                      # probe-mode randomness (driver-seeded)
    # RACK ANTI-AFFINITY (rack_failure_spec.md, 2026-07-27): when the driver
    # enables the rack feature (scenario `rack_size` present), the board places
    # a wave's k pieces on k RACK-DISJOINT donors, never in the flusher's own
    # rack, so one rack event can take at most one piece of any stripe.
    # racks_by_slot maps worker name -> frozenset of rack ids its nodes span
    # (empty = rack-less: relocated onto replacement nodes, never conflicts).
    # Infeasibility falls back to best-effort spread and INCREMENTS
    # rack_fallbacks + sets last_rack_fallback (per-wave flag the strategy
    # copies into the flush details) — never silent. Both default off/empty, so
    # every rack-less scenario keeps the legacy reserve paths byte-identical.
    rack_anti_affinity: bool = False
    racks_by_slot: dict[str, frozenset] = field(default_factory=dict)
    rack_fallbacks: int = 0
    last_rack_fallback: bool = False
    # L1.5 peer-DRAM capacity (l15_peer_dram_spec.md): per-represented-rank
    # daemon-owned DRAM budget — `dram_host_slots` shard-pieces (default 1) and
    # `dram_host_gb` bytes. Enforced at RESERVATION time with the same refusal
    # protocol as the SSD slots (refusals["dram_full"]). The driver sets both
    # from the scenario `model:` block; defaults keep the tier inert.
    dram_host_slots: int = 1
    dram_host_gb: float = float("inf")

    _by_run: ClassVar[dict[int, "DonorRegistry"]] = {}

    @classmethod
    def for_run(cls, run_id: int) -> "DonorRegistry":
        if run_id not in cls._by_run:
            cls._by_run[run_id] = cls()
        return cls._by_run[run_id]

    @classmethod
    def reset(cls, run_id: int) -> None:
        cls._by_run.pop(run_id, None)

    def register(self, worker: CheckpointWorker, job_id: str) -> None:
        self.slots.setdefault(worker.name, _DonorSlot(
            worker=worker, job_id=job_id,
            represents=getattr(worker, "represents", 1)))

    def set_flushing(self, worker: CheckpointWorker, value: bool) -> None:
        slot = self.slots.get(worker.name)
        if slot is not None:
            slot.flushing = value

    def set_available(self, job_id: str, value: bool) -> None:
        """Stage 3: publish a whole job's donor availability (False before its
        arrival, True once it is running)."""
        for slot in self.slots.values():
            if slot.job_id == job_id:
                slot.available = value

    # ---- L1.5 peer-DRAM capacity helpers -------------------------------------
    def _dram_room_shards(self, slot: _DonorSlot, piece_gb: float) -> int:
        """How many more dram-tier shard-pieces of `piece_gb` this donor can
        take right now (slot AND byte caps, scaled by represents)."""
        cap_slots = self.dram_host_slots * slot.represents
        cap_gb = self.dram_host_gb * slot.represents
        by_slots = cap_slots - slot.dram_hosted
        if piece_gb <= BYTE_EPS:
            return max(0, by_slots)
        by_gb = int((cap_gb - slot.dram_gb + BYTE_EPS) // piece_gb)
        return max(0, min(by_slots, by_gb))

    def _dram_take(self, slot: _DonorSlot, shards: int, piece_gb: float) -> None:
        slot.dram_hosted += shards
        slot.dram_gb += shards * piece_gb

    def release_dram(self, donor_name: str, gb: float, shards: int = 1) -> None:
        """Free a resident dram-tier piece's reservation (demoted or lost)."""
        slot = self.slots.get(donor_name)
        if slot is not None:
            slot.dram_hosted = max(0, slot.dram_hosted - shards)
            slot.dram_gb = max(0.0, slot.dram_gb - gb)

    def adjust_dram_gb(self, donor_name: str, delta_gb: float) -> None:
        """Partial-grant re-shard correction: pieces got bigger/smaller than the
        size the reservation was checked at (grant already made; bookkeeping
        follows the actual bytes so release accounting stays exact)."""
        slot = self.slots.get(donor_name)
        if slot is not None:
            slot.dram_gb = max(0.0, slot.dram_gb + delta_gb)

    # ---- matching helpers (shared by registry-refinement + probe paths) ------
    def _room_ok(self, slot: _DonorSlot, tier: str, piece_gb: float,
                 shards: int = 1) -> bool:
        """Capacity check + refusal accounting for one prospective grant.
        tier='ssd' checks the transient per-wave write slots (unchanged);
        tier='dram' checks the persistent L1.5 DRAM budget (refusal protocol
        identical, its own counter)."""
        if tier == "dram":
            if self._dram_room_shards(slot, piece_gb) < shards:
                self.refusals["dram_full"] += slot.represents
                return False
            return True
        if slot.hosted >= slot.shard_capacity:
            self.refusals["full"] += slot.represents
            return False
        return True

    def _grant(self, slot: _DonorSlot, tier: str, piece_gb: float,
               shards: int = 1) -> None:
        if tier == "dram":
            self._dram_take(slot, shards, piece_gb)
        else:
            slot.hosted += shards

    def _ungrant(self, slot: _DonorSlot, tier: str, piece_gb: float,
                 shards: int = 1) -> None:
        if tier == "dram":
            self.release_dram(slot.worker.name, shards * piece_gb, shards)
        else:
            slot.hosted = max(0, slot.hosted - shards)

    def _occupancy_key(self, tier: str, frac: bool = False):
        """Least-loaded ordering key for the tier being reserved."""
        if tier == "dram":
            if frac:
                return lambda s: s.dram_hosted / max(
                    self.dram_host_slots * s.represents, 1)
            return lambda s: s.dram_hosted
        if frac:
            return lambda s: s.hosted / s.shard_capacity
        return lambda s: s.hosted

    def _valid_donors(self, exclude_job: str, tier: str = "ssd",
                      piece_gb: float = 0.0) -> list[_DonorSlot]:
        """Board scan: donors that can host now, counting the rest as refusals
        (own_flush / donor_failed / full) — the eager accounting reserve_shards
        already uses. The legacy `reserve` path keeps its lazy early-break count."""
        valid: list[_DonorSlot] = []
        allowed = self.allowed_donors_by_job.get(exclude_job)
        for slot in self.slots.values():
            if slot.job_id == exclude_job:
                continue
            if allowed is not None and slot.job_id not in allowed:
                continue
            if not slot.available:          # not yet arrived (Stage 3)
                continue
            if slot.flushing:
                self.refusals["own_flush"] += slot.represents
                continue
            if getattr(slot.worker, "failed", False):
                self.refusals["donor_failed"] += slot.represents
                continue
            if not self._room_ok(slot, tier, piece_gb):
                continue
            valid.append(slot)
        return valid

    def _order(self, valid, window_bias, window_of, frac=False):
        """Least-loaded-first ordering (the global-view core). When window_bias is
        on (D9 stage-2), PARTITION first on the donor's availability window: donors
        whose own next flush is >= WINDOW_GUARD_S away ("write to me in this
        window") sort ahead of ones about to flush, least-loaded within each side,
        then farthest-window as the final tie-break. window_of == 0 (phasing off)
        => every donor is out-of-window equally, so this no-ops to least-loaded."""
        def key(s):
            base = (s.hosted / s.shard_capacity) if frac else float(s.hosted)
            if window_bias and window_of is not None:
                w = window_of(s)
                out_of_window = 0 if w >= WINDOW_GUARD_S else 1
                return (out_of_window, base, -w)
            return base
        return sorted(valid, key=key)

    def _reserve_probe(self, k, exclude_job, allow_partial, *,
                       tier="ssd", piece_gb=0.0):
        """Random-probe matching (the pre-centralized daemon). The flusher samples
        a BOUNDED random candidate set (no global load view) and takes the first k
        that accept — the limited-visibility baseline the board improves on."""
        rng = self.rng or random
        pool = [s for s in self.slots.values()
                if s.job_id != exclude_job and s.available]
        fanout = max(PROBE_FANOUT_MIN, 2 * k)
        probed = pool if len(pool) <= fanout else rng.sample(pool, fanout)
        granted: list[_DonorSlot] = []
        for slot in probed:
            if len(granted) == k:
                break
            if slot.flushing:
                self.refusals["own_flush"] += slot.represents
                continue
            if getattr(slot.worker, "failed", False):
                self.refusals["donor_failed"] += slot.represents
                continue
            if not self._room_ok(slot, tier, piece_gb):
                continue
            self._grant(slot, tier, piece_gb)
            granted.append(slot)
        if len(granted) < k:
            if allow_partial and granted:
                return granted
            for slot in granted:
                self._ungrant(slot, tier, piece_gb)
            return []
        return granted

    def _reserve_matched(self, k, exclude_job, allow_partial, *,
                         spread_jobs, window_bias, window_of,
                         tier="ssd", piece_gb=0.0):
        """Centralized least-loaded match with refinements. spread_jobs => round-
        robin across donor JOBS (least-loaded donor within each job), so one
        flusher's k shards don't all land in a single victim job (correlated-loss
        avoidance). window_bias => tie-break toward donors far from their own
        flush window."""
        valid = self._valid_donors(exclude_job, tier, piece_gb)
        ordered = self._order(valid, window_bias, window_of)
        granted: list[_DonorSlot] = []
        if spread_jobs:
            by_job: dict[str, list] = defaultdict(list)
            for s in ordered:
                by_job[s.job_id].append(s)
            job_order = sorted(by_job, key=lambda j: (by_job[j][0].hosted, j))
            idx = {j: 0 for j in by_job}
            while len(granted) < k:
                progressed = False
                for j in job_order:
                    if idx[j] >= len(by_job[j]):
                        continue
                    slot = by_job[j][idx[j]]
                    idx[j] += 1
                    self._grant(slot, tier, piece_gb)
                    granted.append(slot)
                    progressed = True
                    if len(granted) == k:
                        break
                if not progressed:
                    break
        else:
            for slot in ordered:
                if len(granted) == k:
                    break
                self._grant(slot, tier, piece_gb)
                granted.append(slot)
        if len(granted) < k:
            if allow_partial and granted:
                return granted
            for slot in granted:
                self._ungrant(slot, tier, piece_gb)
            return []
        return granted

    # ---- rack anti-affinity (rack_failure_spec.md, 2026-07-27) ---------------
    def _slot_racks(self, name: str | None) -> frozenset:
        if not name:
            return frozenset()
        return self.racks_by_slot.get(name) or frozenset()

    def _reserve_rack_aa(self, k, exclude_job, allow_partial, *,
                         window_bias=False, window_of=None,
                         tier="ssd", piece_gb=0.0, owner=None):
        """Centralized reserve with RACK ANTI-AFFINITY: k pieces on donors with
        pairwise-DISJOINT rack sets, none intersecting the flusher's own racks
        (m=1 slots reduce to k pieces on k distinct racks). Least-loaded-first
        ordering is kept (window_bias honoured via _order; spread_jobs is
        subsumed — one slot per rack forces the donor spread). Infeasible =>
        best-effort spread: the rack-clean grants are KEPT and the remainder is
        filled least-loaded from the deferred slots (non-owner-rack candidates
        first, the owner's rack only as last resort), with rack_fallbacks
        incremented and last_rack_fallback set — never silent."""
        owner_racks = self._slot_racks(owner)
        valid = self._valid_donors(exclude_job, tier, piece_gb)
        ordered = self._order(valid, window_bias, window_of)
        granted: list[_DonorSlot] = []
        used_racks: set = set()
        deferred: list[_DonorSlot] = []
        for slot in ordered:
            if len(granted) == k:
                break
            racks = self._slot_racks(slot.worker.name)
            if racks & owner_racks or racks & used_racks:
                deferred.append(slot)
                self.refusals["rack_conflict"] += slot.represents
                continue
            self._grant(slot, tier, piece_gb)
            granted.append(slot)
            used_racks |= racks
        if len(granted) < k and deferred:
            self.rack_fallbacks += 1
            self.last_rack_fallback = True
            deferred.sort(key=lambda s: bool(self._slot_racks(s.worker.name)
                                             & owner_racks))   # stable: owner last
            for slot in deferred:
                if len(granted) == k:
                    break
                self._grant(slot, tier, piece_gb)
                granted.append(slot)
        if len(granted) < k:
            if allow_partial and granted:
                return granted
            for slot in granted:
                self._ungrant(slot, tier, piece_gb)
            return []
        return granted

    def _reserve_shards_rack_aa(self, n_shards, exclude_job, allow_partial, *,
                                window_bias=False, window_of=None,
                                tier="ssd", piece_gb=0.0, owner=None):
        """Cohort-scale rack anti-affinity: shards fill RACK-DISJOINT donor
        slots first (each chosen slot's rack set unused and off the flusher's
        racks), then best-effort into the deferred slots with the fallback
        counter. A cohort slot spans `represents` nodes, so disjointness is
        enforced at slot-rack-set granularity — the finest the cohort model
        exposes (a take>1 bundle is one piece-group on one slot)."""
        owner_racks = self._slot_racks(owner)
        valid = self._valid_donors(exclude_job, tier, piece_gb)
        ordered = self._order(valid, window_bias, window_of, frac=True)
        granted_map: dict[_DonorSlot, int] = {}
        used_racks: set = set()
        deferred: list[_DonorSlot] = []
        remaining = n_shards
        for slot in ordered:
            if remaining <= 0:
                break
            racks = self._slot_racks(slot.worker.name)
            if racks & owner_racks or racks & used_racks:
                deferred.append(slot)
                self.refusals["rack_conflict"] += slot.represents
                continue
            take = min(remaining, self._shard_room(slot, tier, piece_gb))
            if take <= 0:
                continue
            self._grant(slot, tier, piece_gb, take)
            granted_map[slot] = take
            remaining -= take
            used_racks |= racks
        if remaining > 0 and deferred:
            self.rack_fallbacks += 1
            self.last_rack_fallback = True
            deferred.sort(key=lambda s: bool(self._slot_racks(s.worker.name)
                                             & owner_racks))   # stable: owner last
            for slot in deferred:
                if remaining <= 0:
                    break
                take = min(remaining, self._shard_room(slot, tier, piece_gb))
                if take <= 0:
                    continue
                self._grant(slot, tier, piece_gb, take)
                granted_map[slot] = granted_map.get(slot, 0) + take
                remaining -= take
        granted = list(granted_map.items())
        if remaining > 0:
            self.refusals["dram_full" if tier == "dram" else "full"] += remaining
            if not allow_partial or not granted:
                for slot, take in granted:
                    self._ungrant(slot, tier, piece_gb, take)
                return []
        return granted

    def reserve(self, k: int, exclude_job: str, allow_partial: bool = False, *,
                mode: str = "registry", spread_jobs: bool = False,
                window_bias: bool = False, window_of=None,
                tier: str = "ssd", piece_gb: float = 0.0,
                owner: str | None = None) -> list[_DonorSlot]:
        """tier='dram' (L1.5): reserve daemon-owned DRAM buffers of `piece_gb`
        instead of transient SSD write slots. DRAM grants PERSIST until the piece
        demotes to the host's SSD (or dies with the host) — the caller must NOT
        release() them at wave end; release_dram() frees them per piece.
        owner: the flushing worker's name (rack anti-affinity: its racks are
        excluded from the stripe). Ignored while rack_anti_affinity is off."""
        self.last_rack_fallback = False
        if mode == "probe":
            # the pre-centralized bounded-random daemon probe is the limited-
            # visibility A/B baseline — it stays rack-OBLIVIOUS by design (the
            # board is what gains anti-affinity; contaminating the probe would
            # poison the matching comparison).
            return self._reserve_probe(k, exclude_job, allow_partial,
                                       tier=tier, piece_gb=piece_gb)
        if self.rack_anti_affinity:
            return self._reserve_rack_aa(
                k, exclude_job, allow_partial, window_bias=window_bias,
                window_of=window_of, tier=tier, piece_gb=piece_gb, owner=owner)
        if spread_jobs or window_bias:
            return self._reserve_matched(
                k, exclude_job, allow_partial, spread_jobs=spread_jobs,
                window_bias=window_bias, window_of=window_of,
                tier=tier, piece_gb=piece_gb)
        # ---- legacy least-loaded centralized reserve (UNCHANGED for tier='ssd':
        #      hand-check pencil math and existing arm numbers rest on this path) ----
        granted: list[_DonorSlot] = []
        # least-loaded first: spreads hosted shards across the pool instead of
        # concentrating on the first donors in scan order (simultaneous reservations
        # would otherwise pile onto the same donors, whose transfers then slow each
        # other via network fair-share). The real daemons get this spreading from
        # timing skew; the controller's donor policy makes it explicit.
        for slot in sorted(self.slots.values(), key=self._occupancy_key(tier)):
            if len(granted) == k:
                break
            if slot.job_id == exclude_job:
                continue
            if not slot.available:          # not yet arrived (Stage 3); no-op
                continue                    # (available defaults True everywhere)
            if slot.flushing:
                self.refusals["own_flush"] += slot.represents
                continue
            if tier != "dram" and slot.hosted >= slot.shard_capacity:
                self.refusals["full"] += slot.represents
                continue
            if tier == "dram" and not self._room_ok(slot, tier, piece_gb):
                continue
            if getattr(slot.worker, "failed", False):
                self.refusals["donor_failed"] += slot.represents
                continue
            self._grant(slot, tier, piece_gb)
            granted.append(slot)
        if len(granted) < k:
            if allow_partial and granted:      # daemon v1.1: stream across k' < k donors
                return granted
            for slot in granted:
                self._ungrant(slot, tier, piece_gb)
            return []
        return granted

    def release(self, granted: list[_DonorSlot]) -> None:
        for slot in granted:
            slot.hosted = max(0, slot.hosted - 1)

    def _shard_room(self, slot: _DonorSlot, tier: str, piece_gb: float) -> int:
        """Free shard capacity of `slot` for the tier being reserved."""
        if tier == "dram":
            return self._dram_room_shards(slot, piece_gb)
        return slot.shard_capacity - slot.hosted

    def _reserve_shards_probe(self, n_shards, exclude_job, allow_partial, *,
                              tier="ssd", piece_gb=0.0):
        """Cohort random-probe: sample a bounded random donor set and fill shards
        greedily across it (no global occupancy view)."""
        rng = self.rng or random
        pool = [s for s in self.slots.values()
                if s.job_id != exclude_job and s.available]
        needed = (n_shards + 1) // 2               # cap>=2 => nodes needed
        fanout = max(PROBE_FANOUT_MIN, needed + needed // 2)
        probed = pool if len(pool) <= fanout else rng.sample(pool, fanout)
        granted_map: dict[_DonorSlot, int] = {}
        remaining = n_shards
        for slot in probed:
            if remaining <= 0:
                break
            if slot.flushing:
                self.refusals["own_flush"] += slot.represents
                continue
            if getattr(slot.worker, "failed", False):
                self.refusals["donor_failed"] += slot.represents
                continue
            if not self._room_ok(slot, tier, piece_gb):
                continue
            take = min(remaining, self._shard_room(slot, tier, piece_gb))
            self._grant(slot, tier, piece_gb, take)
            granted_map[slot] = take
            remaining -= take
        granted = list(granted_map.items())
        if remaining > 0:
            self.refusals["dram_full" if tier == "dram" else "full"] += remaining
            if not allow_partial or not granted:
                for slot, take in granted:
                    self._ungrant(slot, tier, piece_gb, take)
                return []
        return granted

    def _reserve_shards_matched(self, n_shards, exclude_job, allow_partial, *,
                                spread_jobs, window_bias, window_of,
                                tier="ssd", piece_gb=0.0):
        """Cohort centralized match. spread_jobs => round-robin one shard per donor
        JOB per cycle (least-occupied donor within the job), spreading a wave's
        shards across victim jobs; else least-occupied-first with optional window
        tie-break."""
        valid = self._valid_donors(exclude_job, tier, piece_gb)
        ordered = self._order(valid, window_bias, window_of, frac=True)
        granted_map: dict[_DonorSlot, int] = {}
        remaining = n_shards
        if spread_jobs:
            by_job: dict[str, list] = defaultdict(list)
            for s in ordered:
                by_job[s.job_id].append(s)
            job_order = sorted(
                by_job, key=lambda j: (by_job[j][0].hosted / by_job[j][0].shard_capacity, j))
            while remaining > 0:
                progressed = False
                for j in job_order:
                    slot = min((x for x in by_job[j]
                                if self._shard_room(x, tier, piece_gb) > 0),
                               key=lambda x: x.hosted / x.shard_capacity, default=None)
                    if slot is None:
                        continue
                    self._grant(slot, tier, piece_gb)
                    granted_map[slot] = granted_map.get(slot, 0) + 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
                if not progressed:
                    break
        else:
            for slot in ordered:
                if remaining <= 0:
                    break
                take = min(remaining, self._shard_room(slot, tier, piece_gb))
                if take <= 0:
                    continue
                self._grant(slot, tier, piece_gb, take)
                granted_map[slot] = take
                remaining -= take
        granted = list(granted_map.items())
        if remaining > 0:
            self.refusals["dram_full" if tier == "dram" else "full"] += remaining
            if not allow_partial or not granted:
                for slot, take in granted:
                    self._ungrant(slot, tier, piece_gb, take)
                return []
        return granted

    def reserve_shards(self, n_shards: int, exclude_job: str,
                       allow_partial: bool = True, *, mode: str = "registry",
                       spread_jobs: bool = False, window_bias: bool = False,
                       window_of=None, tier: str = "ssd",
                       piece_gb: float = 0.0,
                       owner: str | None = None) -> list[tuple[_DonorSlot, int]]:
        """Cohort-scale reservation: place n_shards across donors, least-loaded first
        (by fractional occupancy), respecting per-slot shard capacity. Returns
        [(slot, shards_taken)]. Refusals counted in shards. `mode`/`spread_jobs`/
        `window_bias` select the centralized-matching variants (see reserve).
        tier='dram' (L1.5): grants persist until demotion — see reserve().
        owner: flushing worker's name for rack anti-affinity (see reserve)."""
        self.last_rack_fallback = False
        if mode == "probe":
            # rack-oblivious by design — see reserve()
            return self._reserve_shards_probe(n_shards, exclude_job, allow_partial,
                                              tier=tier, piece_gb=piece_gb)
        if self.rack_anti_affinity:
            return self._reserve_shards_rack_aa(
                n_shards, exclude_job, allow_partial, window_bias=window_bias,
                window_of=window_of, tier=tier, piece_gb=piece_gb, owner=owner)
        if spread_jobs or window_bias:
            return self._reserve_shards_matched(
                n_shards, exclude_job, allow_partial, spread_jobs=spread_jobs,
                window_bias=window_bias, window_of=window_of,
                tier=tier, piece_gb=piece_gb)
        avail: list[_DonorSlot] = []
        for slot in self.slots.values():
            if slot.job_id == exclude_job:
                continue
            if not slot.available:          # not yet arrived (Stage 3)
                continue
            if slot.flushing:
                self.refusals["own_flush"] += slot.represents
                continue
            if getattr(slot.worker, "failed", False):
                self.refusals["donor_failed"] += slot.represents
                continue
            if not self._room_ok(slot, tier, piece_gb):
                continue
            avail.append(slot)
        granted: list[tuple[_DonorSlot, int]] = []
        remaining = n_shards
        for slot in sorted(avail, key=self._occupancy_key(tier, frac=True)):
            if remaining <= 0:
                break
            take = min(remaining, self._shard_room(slot, tier, piece_gb))
            if take <= 0:
                continue
            self._grant(slot, tier, piece_gb, take)
            granted.append((slot, take))
            remaining -= take
        if remaining > 0:
            self.refusals["dram_full" if tier == "dram" else "full"] += remaining
            if not allow_partial or not granted:
                for slot, take in granted:
                    self._ungrant(slot, tier, piece_gb, take)
                return []
        return granted

    def release_shards(self, granted: list[tuple[_DonorSlot, int]]) -> None:
        for slot, take in granted:
            slot.hosted = max(0, slot.hosted - take)

    def reserve_one_slot(self, min_shards: int, exclude_job: str, *,
                         exclude_names=()) -> tuple[_DonorSlot, int] | None:
        """Reserve a SINGLE donor slot with free capacity >= min_shards, for a
        cohort XOR-parity stream that must live on ONE node (so a single lost
        DATA donor is reconstructable, RAID-4 style). Least-loaded first;
        `exclude_names` keeps the parity off the wave's own data donors so its
        loss is independent. Returns (slot, min_shards) or None.
        NOTE: rack-oblivious — the parity backstops are not combined with the
        rack gates (rack_failure_spec.md scopes anti-affinity to the data
        stripe matcher; extend here if a rack x parity study ever runs)."""
        for slot in sorted(self.slots.values(),
                           key=lambda s: s.hosted / s.shard_capacity):
            if slot.job_id == exclude_job or not slot.available:
                continue
            if slot.worker.name in exclude_names:
                continue
            if slot.flushing or getattr(slot.worker, "failed", False):
                continue
            if slot.shard_capacity - slot.hosted < min_shards:
                continue
            slot.hosted += min_shards
            return (slot, min_shards)
        return None


class CrossJobPeerStrategy(TieredCheckpointStrategy):
    """Tiered capture (GPU->DRAM) + cross-job donor persist instead of rank pairing."""

    kpeers_default = 2
    kpeers_by_job: dict[str, int] = {}
    # daemon v1.1 parity: accept k' < k grants instead of falling back (the hardware
    # fleet runs --partial-grants; all-or-nothing in the sim cost -22 pts vs the
    # 32-node anchor at saturation — gap #9's real mechanism)
    partial_grants = True
    # peerd-style async flusher: checkpoint() returns after the DRAM capture and the
    # persist runs in a background process (serialized per rank, like peerd's flusher
    # thread) — so checkpoints do not stretch the training cadence. Off by default to
    # keep main.py/tests unchanged; the scenario driver turns it on.
    async_persist = False
    # slotted flushing (controller v1): flush start times are aligned to a per-job slot
    # on a CLOCK-anchored base period, instead of phases anchored to job start. Start-
    # anchored phases decay as training cadences drift (measured: fully decorrelated
    # within ~2 flush cycles at high duty); clock-anchored slots are drift-immune.
    # slot_by_job maps job_id -> offset seconds within slot_period. None = disabled.
    slot_period: float | None = None
    slot_by_job: dict[str, float] = {}
    # per-job node-class rates (CROSSJOB_GAPS #7, strategy-level so ClusterConfig stays
    # untouched): job_id -> {"nic": GB/s, "disk": GB/s} (+ optional nic_in/nic_out/disk_w
    # overrides in capacity mode). Missing entries fall back to the homogeneous cluster
    # constants. Peer transfers are gated by the SLOWEST granted donor's disk (equal
    # shards) — the sub-linearity measured on the mixed fleet.
    # CL-014 adds two OPTIONAL STRING keys to the same block: `nic_fabric` (which
    # fabric the job's training collective rides) and `ckpt_fabric` (which fabric
    # its checkpoint streams ride, default = nic_fabric). They are labels only:
    # nothing here reads them as rates, and with neither declared every transfer
    # is tagged checkpointing.base.DEFAULT_FABRIC, i.e. one wire, as before.
    rates_by_job: dict[str, dict] = {}
    # capacity mode (umbrella fix for gaps #8/#10): per-node duplex NIC + disk-write
    # resources with max-min fair sharing (checkpointing/capacity.py). Transfer rates
    # become EMERGENT (allocation across sender:out, donor:in, donor:disk) instead of
    # the min() formula + count-based slowdown. checkpoint_network_tasks counters are
    # still maintained so training<->checkpoint coupling keeps working; the count-based
    # slowdown is NOT applied to capacity-driven transfers (no double counting).
    capacity_mode = False
    # capacity engine selector: "event" (default, analytic finish-time scheduling)
    # or "polling" (legacy contention-quantum loop). None here => resolve at
    # instance init from env var SIM_ENGINE then this class attr then "event".
    # run_scenario sets this from the scenario's `simulation.engine` key.
    engine: str | None = None
    # store arm support: persist to a shared store node instead of donors. The store's
    # capacities come from the driver ("_store" policy key -> CapacityRegistry node);
    # store_stream_gbps models the measured per-stream client ceiling (a flow demand
    # cap, not a node capacity — e.g. single-stream HTTP PUT ~0.034 GB/s on the fleet).
    store_mode = False
    store_stream_gbps = 999.0
    # CENTRALIZED PEER MATCHING (Mark D8/D9). matching="registry" routes every
    # wave through the AvailabilityBoard (DonorRegistry) with a global view;
    # "probe" restores the pre-centralized bounded-random daemon probe for the
    # A/B. spread_jobs (refinement a) spreads a wave across donor JOBS;
    # window_bias (refinement b / D9 stage-2) prefers donors whose own next-flush
    # window is far. Both OFF => the original least-loaded reserve, unchanged.
    matching = "registry"
    spread_jobs = False
    window_bias = False
    # DURABILITY BACKSTOP (donor-drain L3, goal 2026-07-19). "donor_drain"
    # (default): after each peer wave the DONORS forward their hosted pieces to
    # the L3 store in the background (owner uninvolved, its DRAM already released
    # at wave end); L3 stores per-rank PIECES at the wave's iteration, so L3 is
    # as fresh as L2 and recovery can stitch k-1 live peer pieces + the missing
    # piece from L3 at the SAME iteration. "owner_push" (legacy): the owner
    # pushes the whole shard from DRAM to the store on the store_every cadence
    # (backstop_upload) — staler, and the driver schedules it, not the wave.
    # "parity" (STRETCH): stripe k data + 1 XOR parity piece across k+1 donors;
    # a single donor loss is reconstructed PEER-SIDE by XOR of the survivors, with
    # no L3 at all (no drains, no owner push). 2+ concurrent losses fall through.
    # "parity_cutoff" (Sam's straggler-cutoff design): the SAME k+1 stripe, but
    # the wave is recoverable the instant ANY k of the k+1 pieces land, so it
    # COMPLETES at the k-th order statistic instead of gating on the slowest — the
    # single slowest straggler DETACHES to a background flow (lands -> recorded;
    # donor dies or wave superseded -> dropped). Converts wave time from
    # max-of-(k+1) to the k-th order statistic (the mega tail-killer).
    backstop = "donor_drain"
    # L1.5 PEER-DRAM TIER (l15_peer_dram_spec.md, 2026-07-26). Jobs in dram_by_job
    # stripe their pieces to donor daemon-owned DRAM instead of donor SSD: the
    # wave lands at min(NIC_in, DRAM) ~= NIC_in (no donor-disk on the write path
    # — the speed win), then each piece asynchronously DEMOTES DRAM -> that
    # host's SSD in the background (RES_DISK only, low max-min weight) and
    # thereupon becomes an ORDINARY L2 piece (survives host reboot, frees the
    # DRAM slot; the L3 donor drain then proceeds from SSD as today). Volatility:
    # dram-tier pieces die with the HOST's reboot/node/spot/eviction (the driver
    # wipes them); they SURVIVE host process failure (the daemon owns them) and
    # every OWNER-side event. Enactment knob: placement `peerdram` (policy or
    # arm-global). The parity/parity_cutoff backstops do not combine with dram
    # waves (peer-side XOR lives on donor SSD); dram waves always stripe plain
    # data pieces.
    dram_by_job: dict[str, bool] = {}
    # IN-JOB SSD PLACEMENT (rack_failure_spec.md, 2026-07-27). Jobs in
    # injob_ssd_by_job write each wave to the RING-NEIGHBOR node of their OWN
    # job — one piece, k=1, on that node's SSD (_persist_injob_ssd). No donor
    # duty/slots, no DRAM rent (internalized own-job cost: local disk time +
    # intra-job NIC). The piece survives the owner's process failure and the
    # HOST's reboot (it is on disk), dies with the host node, and goes dark
    # with the host rack — the priced 2x2 point between injob (DRAM) and
    # crossjob. Recovery-source label: `injob_ssd_replica`. Enactment knob:
    # placement `injob_ssd` (policy or arm-global); dict empty => inert.
    injob_ssd_by_job: dict[str, bool] = {}

    def _rate(self, job_id: str, key: str) -> float:
        default = (self.cluster.network_bandwidth_gbps if key == "nic"
                   else self.cluster.local_ssd_bandwidth_gbps)
        return self.rates_by_job.get(job_id, {}).get(key, default)

    def _window_of(self, slot) -> float:
        """Seconds until this donor's OWN next durable-flush slot — the D9 phasing
        signal a matched daemon would publish ("write to me in this window").
        Larger => the donor is further from its own wave, so more available to host
        now. 0.0 when phasing is off (no slot table): window_bias then no-ops,
        which is correct — with no published windows there is nothing to bias on."""
        period = self.slot_period
        if not period:
            return 0.0
        offset = self.slot_by_job.get(slot.job_id)
        if offset is None:
            return 0.0
        return (offset - self.backend.now) % period

    def __init__(self, **kwargs) -> None:
        kwargs.pop("paired", None)
        kwargs.pop("persist_to_ssd", None)
        kwargs.pop("strategy_name", None)
        super().__init__(
            paired=False,
            persist_to_ssd=False,
            strategy_name="crossjob_peer",
            **kwargs,
        )
        self.registry = DonorRegistry.for_run(self.run_id)
        # base class builds peer_transfer_slots only for its pairing scheme; cross-job
        # transfers key by min rank over (flusher + donors), so backfill missing slots
        class _SlotDict(dict):
            def __init__(self, inner, backend):
                super().__init__(inner)
                self._backend = backend
            def __missing__(self, key):
                self[key] = self._backend.resource(capacity=1)
                return self[key]
        self.peer_transfer_slots = _SlotDict(self.peer_transfer_slots, self.backend)
        self.peer_copies: dict[tuple, dict] = {}
        # L1.5 write-then-rename semantics: when a NEW dram-tier piece lands on
        # a donor that still holds the previous DEMOTED (ssd-tier) piece for
        # the same (job, rank), the old SSD file is only REPLACED when the new
        # piece's demotion completes. Until then it is kept here as a shadow:
        # if the dram piece dies with a host REBOOT (DRAM wiped, disk intact),
        # the shadow is restored — recovery falls to the L2 age, exactly the
        # volatility case handcheck_l15b pins. Invariant: dram_shadow[key]
        # exists only while peer_copies[key] is a dram-tier piece.
        self.dram_shadow: dict[tuple, dict] = {}
        self.stats = Counter()
        # resource-bill accounting (2026-07-28, PURE accounting — no dynamics):
        # (a) the local/replica `copies` ledger is wrapped so every destroy/
        #     overwrite books residency GB-seconds (ckpt_dram_gb_seconds /
        #     ckpt_ssd_gb_seconds) and emits an Accounting trace event;
        # (b) hosted ssd-tier pieces (cross-job stripes, injob_ssd replicas,
        #     demoted L1.5 pieces + their write-then-rename shadows) get a
        #     residency clock keyed like peer_copies, booked into
        #     ckpt_hosted_ssd_gb_seconds at drop/overwrite/discard/horizon.
        # dram-tier hosted residency stays on the existing exactly-once
        # dram_gb_seconds ledger (_book_dram_release) — not double-counted.
        self.copies = BilledCopies(
            lambda: self.backend.now, self.stats, self._bill_copy_end_event)
        self._ssd_hosted_since: dict[tuple, float] = {}
        self.flush_locks: dict[int, Any] = {}
        self.background_flushes: list[Any] = []  # driver drains these after training ends
        self.capacity = CapacityRegistry.for_run(self.run_id)
        self.store_copies: dict[tuple, dict] = {}   # (job, rank) -> newest store copy
        # donor-drain L3: per-rank PIECE copies on the store, keyed
        # (job, rank, piece_idx) -> {iteration, size_gb, k}. A rank's L3 copy is
        # COMPLETE at iteration X when all k pieces of X are present; recovery may
        # also STITCH k-1 live peer pieces + the missing piece from L3 at X.
        self.l3_pieces: dict[tuple, dict] = {}
        self._last_flush_started: dict[int, int] = {}  # rank -> newest flushed it
        self._drain_wave_counter: dict[int, int] = {}   # rank -> waves since drain
        self.drain_every_waves: int = 1                 # set by driver from policy f2/f3
        self.engine = resolve_engine(type(self).engine)
        if self.engine == "event":
            self.capacity.ev_attach(self.backend)

    def _node_caps(self, job_id: str) -> NodeCaps:
        r = self.rates_by_job.get(job_id, {})
        nic = r.get("nic", self.cluster.network_bandwidth_gbps)
        disk = r.get("disk", self.cluster.local_ssd_bandwidth_gbps)
        return NodeCaps(nic_in=r.get("nic_in", nic), nic_out=r.get("nic_out", nic),
                        disk_w=r.get("disk_w", disk))

    def register_worker(self, worker: CheckpointWorker, job_id: str) -> None:
        """Driver entry point: donor pool + capacity node in one call. A cohort worker
        (represents=m) gets m-fold capacities: m nodes' NICs and disks in aggregate."""
        self.registry.register(worker, job_id)
        caps = self._node_caps(job_id)
        m = getattr(worker, "represents", 1)
        self.capacity.set_node(worker.name, NodeCaps(
            nic_in=caps.nic_in * m, nic_out=caps.nic_out * m, disk_w=caps.disk_w * m))

    def _capacity_transfer(self, actor, watched, generations, *, size_gb, streams,
                           iteration, operation, source, destination, details,
                           nominal_gbps, network=True, category="Checkpoint",
                           demand_per_stream=999.0, stream_weights=None,
                           hold_cpu=True):
        """Dispatch to the event or polling capacity engine (self.engine). Both
        preserve identical physics; the event engine removes quantum error."""
        impl = (self._capacity_transfer_event if self.engine == "event"
                else self._capacity_transfer_polling)
        ok = yield from impl(
            actor, watched, generations, size_gb=size_gb, streams=streams,
            iteration=iteration, operation=operation, source=source,
            destination=destination, details=details, nominal_gbps=nominal_gbps,
            network=network, category=category, demand_per_stream=demand_per_stream,
            stream_weights=stream_weights, hold_cpu=hold_cpu)
        return ok

    def _capacity_transfer_event(self, actor, watched, generations, *, size_gb,
                                 streams, iteration, operation, source, destination,
                                 details, nominal_gbps, network=True,
                                 category="Checkpoint", demand_per_stream=999.0,
                                 stream_weights=None, hold_cpu=True):
        """Event-driven twin of _capacity_transfer_polling. Preserves EXACTLY:
        per-stream byte accounting, slowest-stream completion gating, generation-
        abort semantics (via the failure path's capacity.ev_notify_worker_change),
        the effective_slowdown ceiling, CPU-hold, and network coupling. Between
        membership changes rates are constant, so bytes credit exactly — the
        transfer matches the analytic fluid model with no quantum overshoot."""
        group = object()
        demands = (list(demand_per_stream)
                   if isinstance(demand_per_stream, (list, tuple))
                   else [demand_per_stream] * len(streams))
        weights = (list(stream_weights) if stream_weights is not None
                   else [1.0] * len(streams))
        # audit fix #2 (parity): acquire the CPU BEFORE registering flows.
        cpu_request = None
        if hold_cpu:
            cpu_request = actor.cpu.request(priority=10)
            yield cpu_request
        # per-STREAM byte tracking (audit fix #0): each stream carries its
        # weight-proportional share; the transfer completes when the LAST stream
        # does; early finishers free capacity (the scheduler re-fills the rest).
        total_w = sum(weights) or 1.0
        flows = [self.capacity.make_flow(u, demand=d, group=group, weight=w)
                 for u, d, w in zip(streams, demands, weights)]
        for f, w in zip(flows, weights):
            f.remaining = size_gb * w / total_w
        open_flows = list(flows)
        # transfer CEILING (identical to polling): per stream the min physical cap
        # along its path, bounded by demand; effective_slowdown records against it.
        ceiling = 0.0
        for uses, dem in zip(streams, demands):
            per = dem
            for node, res in uses:
                caps = self.capacity.nodes.get(node)
                if caps is not None:
                    per = min(per, caps.cap(res))
            ceiling += per
        ceiling = max(ceiling, 1e-9)
        net_workers = tuple(watched) if network else ()
        # training <-> checkpoint coupling (counters). CL-014: the stream is
        # tagged with the OWNER job's checkpoint fabric — the donor's counter
        # rises on the wire the stripe actually rides, not on the donor's own.
        net_fabric = self.ckpt_fabric(actor.job_id)
        # CL-016: the transfer's OFFERED rate is `ceiling` — per stream the min
        # physical cap along its path bounded by the stream demand, summed. All
        # inputs are declared scenario constants (tc-shaped rates, measured TCP
        # ceiling, per-node NIC/disk caps); nothing fitted.
        bump_network_tasks(net_workers, net_fabric, +1, rate_gbps=ceiling)
        start = self.backend.now
        if category == "Recovery":
            def valid() -> bool:
                return tuple(w.failure_generation for w in watched) == tuple(generations)
        else:
            def valid() -> bool:
                return self._checkpoint_generation_valid(watched, generations)
        armed = False
        try:
            if not valid():
                return False
            # settle tick: siblings opened in the SAME instant must all arm before
            # the first completion is scheduled (parity with the polling engine).
            yield self.backend.timeout(0)
            if not valid():              # a failure fired during the settle tick
                return False
            done_ev, abort_ev, _partial = self.capacity.ev_arm(
                group, open_flows, watched)
            armed = True
            # wake on the transfer's own completion OR a generation-abort. Between
            # here and then, nothing polls; the scheduler advances the clock to the
            # exact per-stream finishes and to any membership change.
            yield (done_ev | abort_ev)
            if abort_ev.triggered:       # watched worker's generation changed
                return False
            base = size_gb / ceiling
            self._record(
                actor, start=start, category=category, operation=operation,
                resources=(["CPU", "NETWORK"] if network else ["CPU"]),
                iteration=iteration, source=source, destination=destination,
                data_gb=size_gb,
                details={**details, "chunk_index": 1, "chunk_count": 1,
                         "base_duration": base,
                         "effective_slowdown": ((self.backend.now - start) / base
                                                if base > 0 else 1.0),
                         "capacity_mode": True})
            return True
        finally:
            bump_network_tasks(net_workers, net_fabric, -1, rate_gbps=ceiling)
            if cpu_request is not None:
                actor.cpu.release(cpu_request)
            if armed:
                self.capacity.ev_dismiss(group)

    def _capacity_transfer_polling(self, actor, watched, generations, *, size_gb,
                                   streams, iteration, operation, source,
                                   destination, details, nominal_gbps, network=True,
                                   category="Checkpoint", demand_per_stream=999.0,
                                   stream_weights=None, hold_cpu=True):
        """One transfer whose rate is the max-min allocation of its flows, re-evaluated
        every contention quantum. Replaces the fixed-rate + count-based-slowdown path
        when capacity_mode is on. (Legacy engine; byte-identical to pre-rewrite.)"""
        group = object()
        demands = (list(demand_per_stream)
                   if isinstance(demand_per_stream, (list, tuple))
                   else [demand_per_stream] * len(streams))
        weights = (list(stream_weights) if stream_weights is not None
                   else [1.0] * len(streams))
        # audit fix #2: acquire the CPU BEFORE registering flows — a queued
        # transfer must not create phantom contention against running ones.
        # hold_cpu=False = background QoS (store backstop): a long upload must
        # not serialize with captures on the worker's core (3rd 3-tier run:
        # +1,000-2,000 s completion purely from captures queued behind uploads)
        cpu_request = None
        if hold_cpu:
            cpu_request = actor.cpu.request(priority=10)
            yield cpu_request
        flows = [self.capacity.open(u, demand=d, group=group, weight=w)
                 for u, d, w in zip(streams, demands, weights)]
        open_flows = list(flows)
        # transfer CEILING: per stream, the min physical cap along its path
        # (bounded by the stream demand); the max achievable aggregate rate.
        # effective_slowdown is recorded against THIS (guaranteed >= 1), not a
        # caller-supplied nominal — mixed-class donor sets and disk-free
        # recovery paths made nominal-based slowdowns dip impossibly below 1
        # (caught by trace_validator I3 on its first run).
        ceiling = 0.0
        for uses, dem in zip(streams, demands):
            per = dem
            for node, res in uses:
                caps = self.capacity.nodes.get(node)
                if caps is not None:
                    per = min(per, caps.cap(res))
            ceiling += per
        ceiling = max(ceiling, 1e-9)
        net_workers = tuple(watched) if network else ()
        # training <-> checkpoint coupling (counters). CL-014: the stream is
        # tagged with the OWNER job's checkpoint fabric — the donor's counter
        # rises on the wire the stripe actually rides, not on the donor's own.
        net_fabric = self.ckpt_fabric(actor.job_id)
        # CL-016: same offered-rate provenance as the event engine — `ceiling`
        # is min(physical caps along the path, stream demand), summed per stream.
        bump_network_tasks(net_workers, net_fabric, +1, rate_gbps=ceiling)
        start = self.backend.now
        # recovery transfers run WHILE worker.failed is True (cleared only after
        # success) — the checkpoint-grade validity check would reject them forever
        # (observed: zero-sim-time retry spin in the FailureController's recover loop).
        # Match the base class's recovery semantics: generation equality only.
        if category == "Recovery":
            def valid() -> bool:
                return tuple(w.failure_generation for w in watched) == tuple(generations)
        else:
            def valid() -> bool:
                return self._checkpoint_generation_valid(watched, generations)
        try:
            if not valid():
                return False

            # audit fix #0: per-STREAM byte tracking. Each stream carries its
            # weight-proportional share of the bytes and finishes at its OWN
            # allocated rate; the transfer completes when the LAST stream does
            # (equal-shard / slowest-donor gating — the semantics measured on
            # the mixed fleet). Early-finishing streams close their flows,
            # freeing capacity for the stragglers (max-min re-fills).
            total_w = sum(weights) or 1.0
            remaining = [size_gb * w / total_w for w in weights]
            # settle tick: transfers opened in the SAME sim instant must all be
            # registered before anyone samples rates, else the first opener
            # credits itself a solo-rate window (hand-check2: 2.95 s flushes
            # where fluid fair-share gives 3.45 s, with ~3% capacity overshoot)
            yield self.backend.timeout(0)
            # window cap: fine-grained for short transfers (stale-rate error is
            # relative), coarse for long ones (error amortizes); credited at the
            # MIN of window-start/end rates so capacity is never overshot
            first_rates = [self.capacity.flow_rate(f) for f in open_flows]
            est = size_gb / max(sum(first_rates), 1e-9)
            quantum = max(0.1, min(max(self.contention_quantum_seconds, 0.5),
                                   est / 10.0))
            aborted_mid = False
            while any(r > 1e-9 for r in remaining):
                if not valid():
                    aborted_mid = True
                    break
                rates = [self.capacity.flow_rate(f) if f is not None else 0.0
                         for f in open_flows]
                dt = quantum                 # step to next completion, max quantum
                for r, rem in zip(rates, remaining):
                    if rem > 1e-9 and r > 1e-12:
                        dt = min(dt, rem / r)
                yield self.backend.timeout(max(dt, 1e-6))
                rates_post = [self.capacity.flow_rate(f) if f is not None else 0.0
                              for f in open_flows]
                for i, (r0, r1) in enumerate(zip(rates, rates_post)):
                    if remaining[i] > 1e-9:
                        remaining[i] = max(0.0, remaining[i] - min(r0, r1) * dt)
                        if remaining[i] <= 1e-9 and open_flows[i] is not None:
                            self.capacity.close(open_flows[i])
                            open_flows[i] = None
            if aborted_mid:
                return False
            base = size_gb / ceiling
            self._record(
                actor, start=start, category=category, operation=operation,
                resources=(["CPU", "NETWORK"] if network else ["CPU"]),
                iteration=iteration, source=source, destination=destination,
                data_gb=size_gb,
                details={**details, "chunk_index": 1, "chunk_count": 1,
                         "base_duration": base,
                         "effective_slowdown": ((self.backend.now - start) / base
                                                if base > 0 else 1.0),
                         "capacity_mode": True})
            return True
        finally:
            bump_network_tasks(net_workers, net_fabric, -1, rate_gbps=ceiling)
            if cpu_request is not None:
                actor.cpu.release(cpu_request)
            for f in open_flows:
                if f is not None:
                    self.capacity.close(f)

    def _restore_from_store(self, worker, job, *, generation, store_it, fetch_gb,
                            shard_gb):
        """Fetch a store copy -> own DRAM -> GPU. The durability backstop: the
        store node is not part of any job, so its copies survive node loss AND
        whole-job eviction. Shared by the store arms, the ours backstop, and
        GEMINI's remote-persistent-store tier (baselines.py)."""
        caps = self._node_caps(job.job_id)
        nominal = min(caps.nic_in, self.cluster.object_store_bandwidth_gbps)
        if self.capacity_mode:
            # store fetch: store NIC-out shared with every other reader, own NIC-in
            loaded = yield from self._capacity_transfer(
                worker, (worker,), (generation,), size_gb=fetch_gb,
                streams=[[(STORE_NODE, RES_OUT), (worker.name, RES_IN)]],
                iteration=store_it,
                operation="checkpoint_store_to_dram_recovery_chunk",
                source=STORE_NODE, destination=f"{worker.name}/dram",
                details={"checkpoint_strategy": self.strategy_name,
                         "checkpoint_source_tier": "store",
                         "checkpoint_source_rank": None},
                nominal_gbps=nominal, category="Recovery")
        else:
            loaded = yield from self._recovery_chunks(
                worker, generation=generation, iteration=store_it,
                size_gb=fetch_gb, chunk_gb=job.checkpoint_chunk_gb,
                bandwidth_gbps=nominal,
                operation="checkpoint_store_to_dram_recovery_chunk",
                source=STORE_NODE, destination=f"{worker.name}/dram",
                resources=["CPU", "NETWORK"],
                source_tier="store", source_rank=None)
        if not loaded or worker.failure_generation != generation:
            return False
        ok = yield from self._finish_restore(worker, generation, shard_gb,
                                             store_it, "store")
        return ok

    def _finish_restore(self, worker, generation, shard_gb, checkpoint_iteration,
                        source_tier):
        """Shared dram->GPU restore tail (mirrors the base recover() semantics)."""
        self._put_copy(owner_rank=worker.rank, location_rank=worker.rank,
                       tier="dram", iteration=checkpoint_iteration, size_gb=shard_gb,
                       bill_gb=shard_gb * getattr(worker, "represents", 1),
                       record_begin=True)
        cpu_request = worker.cpu.request(priority=-10)
        gpu_request = worker.gpu.request()
        yield self.backend.all_of([cpu_request, gpu_request])
        if worker.failure_generation != generation:
            worker.cpu.release(cpu_request)
            worker.gpu.release(gpu_request)
            return False
        start = self.backend.now
        duration = shard_gb / self.cluster.gpu_cpu_bandwidth_gbps
        completed = yield from self._recovery_timeout(
            worker, generation=generation, duration=duration)
        if completed:
            self._record(
                worker, start=start, category="Recovery",
                operation="dram_to_gpu_restore", resources=["CPU", "GPU"],
                iteration=checkpoint_iteration, source=f"{worker.name}/dram",
                destination=worker.name, data_gb=shard_gb,
                details={"checkpoint_strategy": self.strategy_name,
                         "checkpoint_iteration": checkpoint_iteration,
                         "checkpoint_source_tier": source_tier,
                         "checkpoint_source_rank": None,
                         "base_duration": duration, "effective_slowdown": 1.0},
            )
        worker.cpu.release(cpu_request)
        worker.gpu.release(gpu_request)
        return completed

    # -- registration happens lazily: workers dict is per job ------------------
    def _ensure_registered(self, job_id: str) -> None:
        for worker in self.workers.values():
            self.registry.register(worker, job_id)

    def _persist_checkpoint(self, worker: CheckpointWorker, job, **kw):
        checkpoint_epoch = kw.pop("checkpoint_epoch", None)
        if (
            checkpoint_epoch is not None
            and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
        ):
            return False
        self._ensure_registered(job.job_id)
        if not self.async_persist:
            ok = yield from self._persist_flow(worker, job, **kw)
            if (
                checkpoint_epoch is not None
                and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
            ):
                self.discard_checkpoints_after(
                    job.job_id, getattr(worker, "current_iteration", 0)
                )
                return False
            return ok
        # async: report success at capture time; flush proceeds in the background
        # (safe while failures are decoupled — v1 pairs async mode with failure-free runs)
        self.background_flushes.append(
            self.backend.process(
                self._persist_serialized(
                    worker,
                    job,
                    self.backend.now,
                    checkpoint_epoch=checkpoint_epoch,
                    **kw,
                ))
        )
        return True

    def _persist_serialized(
        self,
        worker: CheckpointWorker,
        job,
        t_captured,
        *,
        checkpoint_epoch=None,
        **kw,
    ):
        if (
            checkpoint_epoch is not None
            and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
        ):
            return False
        if worker.rank not in self.flush_locks:
            self.flush_locks[worker.rank] = self.backend.resource(capacity=1)
        lock = self.flush_locks[worker.rank]
        with lock.request() as req:
            yield req
            if (
                checkpoint_epoch is not None
                and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
            ):
                return False
            # STALE-FLUSH SUPPRESSION: under contention the per-worker queue
            # backs up; by the time an old wave gets the lock a newer iteration
            # may already have been persisted (or be about to). Shipping the
            # stale wave wastes donor bandwidth persisting superseded state
            # (validator I8 caught waves executing ~110 s late on mini-C, and
            # backlog is a prime suspect for slots underperforming random
            # phases on the evicted class). The real daemon would do the same:
            # drop the outdated capture, persist the freshest one.
            it = kw.get("iteration")
            last = self._last_flush_started.get(worker.rank, -1)
            if it is not None and it <= last:
                self.stats["stale_flush_skipped"] = \
                    self.stats.get("stale_flush_skipped", 0) + 1
                return True
            if it is not None:
                self._last_flush_started[worker.rank] = it
            if self.slot_period and job.job_id in self.slot_by_job:
                # wait for the next occurrence of this job's clock-anchored flush slot —
                # UNLESS this persist is already >= one period late (queued behind a
                # previous flush): the real daemon paces one flush per interval tick and
                # never compounds slot waits on a backlog (gap #9, 32-node anchor:
                # compounding waits cost -22 pts at saturation vs hardware)
                if self.backend.now - t_captured < self.slot_period:
                    offset = self.slot_by_job[job.job_id]
                    wait = (offset - self.backend.now) % self.slot_period
                    if wait > 1e-9:
                        yield self.backend.timeout(wait)
            if (
                checkpoint_epoch is not None
                and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
            ):
                return False
            ok = yield from self._persist_flow(worker, job, **kw)
            if (
                checkpoint_epoch is not None
                and getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
            ):
                self.discard_checkpoints_after(
                    job.job_id, getattr(worker, "current_iteration", 0)
                )
                return False
            return ok

    def _persist_flow(
        self,
        worker: CheckpointWorker,
        job,
        *,
        iteration: int,
        generation: int,
        shard_gb: float,
        checkpoint_group: str,
    ):
        m = getattr(worker, "represents", 1)
        if self.injob_ssd_by_job.get(job.job_id):
            ok = yield from self._persist_injob_ssd(
                worker, job, m=m, iteration=iteration, generation=generation,
                shard_gb=shard_gb, checkpoint_group=checkpoint_group)
            return ok
        if m > 1:
            ok = yield from self._persist_flow_cohort(
                worker, job, m=m, iteration=iteration, generation=generation,
                shard_gb=shard_gb, checkpoint_group=checkpoint_group)
            return ok
        k = self.kpeers_by_job.get(job.job_id, self.kpeers_default)
        common_base = {
            "checkpoint_strategy": self.strategy_name,
            "checkpoint_group": checkpoint_group,
            "checkpoint_owner_rank": worker.rank,
            "shard": worker.pipeline_stage,
            "shard_count": job.pipeline_stages,
            "kpeers": k,
        }
        if self.store_mode:
            # store arm: durable upload to the shared store node (capacity-fair)
            nominal = min(self._node_caps(job.job_id).nic_out,
                          self.cluster.object_store_bandwidth_gbps)
            if self.capacity_mode:
                ok = yield from self._capacity_transfer(
                    worker, (worker,), (generation,), size_gb=shard_gb,
                    streams=[[(worker.name, RES_OUT), (STORE_NODE, RES_IN),
                              (STORE_NODE, RES_DISK)]],
                    iteration=iteration,
                    operation="checkpoint_stage_dram_to_object_store",
                    source=f"{worker.name}/dram", destination=f"{STORE_NODE}",
                    details={**common_base, "path": "store"}, nominal_gbps=nominal,
                    demand_per_stream=self.store_stream_gbps)
            else:
                ok = yield from self._checkpoint_chunks(
                    actor=worker, watched=(worker,), generations=(generation,),
                    iteration=iteration, size_gb=shard_gb,
                    chunk_gb=job.checkpoint_chunk_gb,
                    bandwidth_gbps=self.cluster.object_store_bandwidth_gbps,
                    operation="checkpoint_stage_dram_to_object_store",
                    source=f"{worker.name}/dram", destination=STORE_NODE,
                    resources=["CPU", "NETWORK"],
                    details={**common_base, "path": "store"})
            if ok:
                self.stats["store_flushes"] += 1
                self.stats["store_gb_written"] += shard_gb   # resource bill
                self._put_store_copy(job, worker, iteration, shard_gb)
            return ok
        self.registry.set_flushing(worker, True)
        try:
            # STRETCH: +1 XOR parity piece (backstop == "parity"). Stripe k data
            # pieces + 1 parity (XOR of the k) across k+1 donors all-or-nothing;
            # ANY single donor loss is then reconstructed PEER-SIDE from the other
            # k pieces, with no L3 fetch. On an incomplete k+1 grant fall back to a
            # plain k-wide data stripe (no parity this wave). data_k = number of
            # DATA pieces; the shard is split into data_k pieces of shard_gb/data_k.
            parity_wave = False
            data_k = k
            granted = None
            # L1.5: dram waves stripe plain data pieces to donor DRAM buffers;
            # parity striping never combines with them (see dram_by_job note).
            dram_wave = bool(self.dram_by_job.get(job.job_id, False)) and k > 0
            dram_piece_est = shard_gb / max(k, 1)
            parity_mode = (not dram_wave) and self.backstop in ("parity",
                                                                "parity_cutoff")
            cutoff = self.backstop == "parity_cutoff"
            if parity_mode:
                granted = self.registry.reserve(
                    k + 1, exclude_job=job.job_id, allow_partial=False,
                    mode=self.matching, spread_jobs=self.spread_jobs,
                    window_bias=self.window_bias, window_of=self._window_of,
                    owner=worker.name)
                if granted:
                    parity_wave = True        # k data + 1 parity across k+1 donors
                    data_k = k
            if not parity_wave:
                granted = self.registry.reserve(
                    k, exclude_job=job.job_id, allow_partial=self.partial_grants,
                    mode=self.matching, spread_jobs=self.spread_jobs,
                    window_bias=self.window_bias, window_of=self._window_of,
                    tier=("dram" if dram_wave else "ssd"),
                    piece_gb=(dram_piece_est if dram_wave else 0.0),
                    owner=worker.name)
            # rack anti-affinity fallback flag for THIS wave (validator I21
            # exempts flagged waves; absent everywhere while the feature is off)
            rack_fb = ({"rack_fallback": True}
                       if self.registry.last_rack_fallback else {})
            partial = bool(granted) and not parity_wave and len(granted) < k
            if partial:
                self.stats["partial_grants"] += 1
                k = len(granted)               # re-shard across the smaller donor set
                data_k = k
                if dram_wave:
                    # pieces grew (shard/k' > shard/k): true up the byte ledger so
                    # release accounting stays exact (grant itself already made)
                    delta = shard_gb / k - dram_piece_est
                    for g in granted:
                        self.registry.adjust_dram_gb(g.worker.name, delta)
            common = {
                "checkpoint_strategy": self.strategy_name,
                "checkpoint_group": checkpoint_group,
                "checkpoint_owner_rank": worker.rank,
                "shard": worker.pipeline_stage,
                "shard_count": job.pipeline_stages,
                "kpeers": k,
            }
            if not granted:
                # fallback: local SSD (on-node copy; NOT node-loss durable)
                self.stats["fallback_flushes"] += 1
                if self.capacity_mode:
                    ok = yield from self._capacity_transfer(
                        worker, (worker,), (generation,), size_gb=shard_gb,
                        streams=[[(worker.name, RES_DISK)]], iteration=iteration,
                        operation="checkpoint_dram_to_local_ssd_chunk",
                        source=f"{worker.name}/dram", destination=f"{worker.name}/ssd",
                        details={**common, "path": "local_fallback"},
                        nominal_gbps=self._rate(job.job_id, "disk"), network=False)
                else:
                    ok = yield from self._checkpoint_chunks(
                        actor=worker,
                        watched=(worker,),
                        generations=(generation,),
                        iteration=iteration,
                        size_gb=shard_gb,
                        chunk_gb=job.checkpoint_chunk_gb,
                        bandwidth_gbps=self._rate(job.job_id, "disk"),
                        operation="checkpoint_dram_to_local_ssd_chunk",
                        source=f"{worker.name}/dram",
                        destination=f"{worker.name}/ssd",
                        resources=["CPU"],
                        details={**common, "path": "local_fallback"},
                    )
                if ok:
                    self._put_copy(
                        owner_rank=worker.rank,
                        location_rank=worker.rank,
                        tier="ssd",
                        iteration=iteration,
                        size_gb=shard_gb,
                    )
                return ok

            # cross-job parallel donor write: eff bw = min(own NIC, n * slowest donor
            # disk) — equal pieces, slowest-granted-donor gating (measured on the
            # fleet). n = pieces written = data_k (+1 for a parity wave); each piece
            # is shard_gb/data_k, so a parity wave ships (data_k+1)/data_k x shard_gb.
            self.stats["peer_flushes"] += 1
            n = len(granted)
            piece_gb = shard_gb / data_k
            total_write = piece_gb * n
            if parity_wave:
                self.stats["parity_flushes"] += 1
            slowest_donor = min(self._rate(g.job_id, "disk") for g in granted)
            eff_bw = min(self._rate(job.job_id, "nic"), n * slowest_donor)
            donor_names = [g.worker.name for g in granted]
            watched = (worker, *[g.worker for g in granted])
            gens = (generation, *[g.worker.failure_generation for g in granted])
            if parity_wave and self.capacity_mode:
                # XOR-parity striping: k data + 1 parity across k+1 donors. Under
                # "parity_cutoff" the wave completes at the data_k-th (=k-th) landed
                # stream and the slowest straggler detaches; under plain "parity" it
                # waits for all k+1 (no cutoff). Per-piece release lives inside the
                # transfer (the straggler's donor stays reserved until it lands).
                pieces = []
                for i, g in enumerate(granted):
                    dw = g.worker
                    pieces.append({
                        "uses": [(worker.name, RES_OUT), (dw.name, RES_IN),
                                 (dw.name, RES_DISK)],
                        "demand": min(self._rate(g.job_id, "nic"),
                                      self._rate(g.job_id, "disk")),
                        "weight": 1.0, "bytes": piece_gb, "shards": 1,
                        "donor": dw.name, "donor_job": g.job_id,
                        "donor_worker": dw,
                        "watched": (worker, dw),
                        "gens": (generation, dw.failure_generation),
                        "release": (lambda s=g: self.registry.release([s])),
                        "meta": {"piece_idx": i, "parity": (i == data_k)},
                    })
                ok = yield from self._parity_cutoff_transfer(
                    worker, job, watched, gens, pieces=pieces,
                    data_streams=data_k, iteration=iteration, common=common,
                    cutoff=cutoff)
                if not ok:
                    self.stats["peer_flush_aborted"] += 1
                return ok
            try:
                if self.capacity_mode and dram_wave:
                    # L1.5 dram stripe: sender NIC-out shared across the k streams,
                    # each stream bounded ONLY by its donor's NIC-in — the piece
                    # lands in the daemon's DRAM buffer at min(NIC_in, DRAM) ~=
                    # NIC_in; the donor DISK is NOT on the write path (that is the
                    # whole speed win; the disk bytes are paid later at demotion).
                    streams = [[(worker.name, RES_OUT),
                                (g.worker.name, RES_IN)] for g in granted]
                    per_node = [self._node_caps(g.job_id).nic_in for g in granted]
                    eff_bw = min(self._node_caps(job.job_id).nic_out,
                                 n * min(per_node))
                    ok = yield from self._capacity_transfer(
                        worker, watched, gens, size_gb=total_write, streams=streams,
                        iteration=iteration,
                        operation="checkpoint_dram_to_crossjob_peers_chunk",
                        source=f"{worker.name}/dram",
                        destination="+".join(f"{nm}/dram" for nm in donor_names),
                        details={**common, "path": "peer_dram", "tier": "dram",
                                 "partial": partial, "donors": donor_names,
                                 "data_k": data_k, "parity": False, **rack_fb},
                        nominal_gbps=eff_bw, demand_per_stream=per_node)
                elif self.capacity_mode:
                    # one flow per donor stream: sender NIC-out shared across the k
                    # streams, each stream bounded by its donor's NIC-in and disk —
                    # the min() formula becomes an emergent allocation
                    streams = [[(worker.name, RES_OUT),
                                (g.worker.name, RES_IN),
                                (g.worker.name, RES_DISK)] for g in granted]
                    # per-stream demand = ONE physical node-pair: capped by the
                    # donor NODE's own nic-in/disk, not the donor COHORT's
                    # aggregate. Without this, a shard landing on an m>1 donor
                    # cohort could stream at the cohort-aggregated disk rate —
                    # physically one node's disk (validator I2 caught it on
                    # mini-C, 8.33 GB/s into a 6 GB/s disk; pre-existing in
                    # both engines, donors were cohort-aggregated since the
                    # cohort model's birth).
                    per_node = [min(self._rate(g.job_id, "nic"),
                                    self._rate(g.job_id, "disk"))
                                for g in granted]
                    ok = yield from self._capacity_transfer(
                        worker, watched, gens, size_gb=total_write, streams=streams,
                        iteration=iteration,
                        operation="checkpoint_dram_to_crossjob_peers_chunk",
                        source=f"{worker.name}/dram",
                        destination="+".join(f"{nm}/ssd" for nm in donor_names),
                        details={**common,
                                 "path": ("peer_parity" if parity_wave else
                                          "peer_partial" if partial else "peer"),
                                 "donors": donor_names,
                                 "data_k": data_k, "parity": parity_wave,
                                 **rack_fb},
                        nominal_gbps=eff_bw, demand_per_stream=per_node)
                else:
                    # legacy path: fixed min() rate + count-based fair share.
                    # _checkpoint_chunks serializes on peer_transfer_slots[min rank of
                    # watched]; ranks are per-job so cross-job transfers would collide on
                    # the same key — give THIS transfer a fresh capacity-1 slot (an
                    # in-flight transfer keeps its own resource reference).
                    slot_key = min(w.rank for w in watched)
                    self.peer_transfer_slots[slot_key] = self.backend.resource(capacity=1)
                    ok = yield from self._checkpoint_chunks(
                        actor=worker,
                        watched=watched,
                        generations=gens,
                        iteration=iteration,
                        size_gb=total_write,
                        chunk_gb=job.checkpoint_chunk_gb,
                        bandwidth_gbps=eff_bw,
                        operation="checkpoint_dram_to_crossjob_peers_chunk",
                        source=f"{worker.name}/dram",
                        destination="+".join(f"{nm}/ssd" for nm in donor_names),
                        resources=["CPU", "NETWORK"],
                        details={**common,
                                 "path": ("peer_parity" if parity_wave else "peer"),
                                 "donors": donor_names,
                                 "data_k": data_k, "parity": parity_wave,
                                 **rack_fb},
                    )
            finally:
                if not dram_wave:
                    self.registry.release(granted)
            if not ok:
                self.stats["peer_flush_aborted"] += 1
                if dram_wave:
                    # aborted stripe: nothing landed — free the DRAM reservations
                    for g in granted:
                        self.registry.release_dram(g.worker.name, piece_gb, 1)
                return False
            if dram_wave:
                self.stats["dram_peer_flushes"] += 1
            drain_info = []
            demote_info = []
            for i, g in enumerate(granted):
                key = (job.job_id, worker.rank, g.worker.name)
                current = self.peer_copies.get(key)
                if (
                    iteration > getattr(worker, "current_iteration", iteration)
                    or current is not None
                    and current["iteration"] > iteration
                ):
                    if dram_wave:
                        # stale piece never becomes state: daemon discards the
                        # buffer immediately (reservation freed, ~zero residency)
                        self.registry.release_dram(g.worker.name, piece_gb, 1)
                    continue
                if dram_wave and current is not None:
                    if current.get("tier") == "dram":
                        # superseding an un-demoted dram piece on the same donor:
                        # the daemon drops the old buffer (frees its reservation);
                        # any existing shadow stays — it is still the last
                        # durable SSD file on that donor.
                        self._book_dram_release(key, current, lost=False)
                        self.stats["dram_superseded"] += 1
                    else:
                        # write-then-rename: the demoted (ssd) piece's file is
                        # only replaced when the NEW piece's demotion completes
                        self.dram_shadow[key] = current
                meta = {
                    "iteration": iteration,
                    "size_gb": piece_gb,
                    "donor_job": g.job_id,
                    "k": n,                    # total pieces (data + any parity)
                    "data_k": data_k,          # DATA pieces needed to reconstruct
                    "piece_idx": i,
                    "parity": parity_wave and i == data_k,   # last piece = parity
                }
                if dram_wave:
                    meta["tier"] = "dram"
                    meta["shards"] = 1
                    meta["hosted_at"] = self.backend.now
                else:
                    # resource bill: ssd piece replaces ssd piece (close old
                    # clock) and starts its own residency clock. dram waves
                    # keep the shadow's clock running (write-then-rename).
                    if current is not None:
                        self._bill_hosted_ssd_close(key, current)
                    self._bill_hosted_ssd_open(key)
                self.peer_copies[key] = meta
                if dram_wave:
                    demote_info.append((g.worker, g.job_id, key, meta, 1))
                # parity pieces are peer-side only (reconstructed via XOR), never
                # drained; data pieces drain when backstop == donor_drain.
                elif not (parity_wave and i == data_k):
                    drain_info.append((g.worker, g.job_id, i, piece_gb))
            if dram_wave:
                # demote each piece DRAM -> host SSD in the background; the L3
                # donor drain chains AFTER the demotions (it reads donor SSD).
                self._schedule_dram_demotions(worker, job, iteration, demote_info)
            else:
                self._schedule_wave_drain(worker, job, iteration, drain_info)
            return True
        finally:
            self.registry.set_flushing(worker, False)

    def _persist_flow_cohort(self, worker, job, *, m, iteration, generation,
                             shard_gb, checkpoint_group):
        """Cohort-scale persist: `worker` stands for m ranks flushing in parallel.
        Aggregate bytes = m * shard_gb (shard_gb stays the PER-RANK measured size);
        donor grants are counted in SHARDS (k*m) via reserve_shards, one capacity
        stream per granted donor slot with demand = take * per-node ingest bound.
        The worker's capacity node carries m-fold caps (register_worker), so
        per-rank transfer physics are invariant while cross-job contention sees
        the physically-correct aggregate demand. Requires capacity_mode."""
        assert self.capacity_mode, "cohort workers (represents>1) require capacity mode"
        k = self.kpeers_by_job.get(job.job_id, self.kpeers_default)
        total_gb = m * shard_gb
        common = {
            "checkpoint_strategy": self.strategy_name,
            "checkpoint_group": checkpoint_group,
            "checkpoint_owner_rank": worker.rank,
            "shard": worker.pipeline_stage,
            "shard_count": job.pipeline_stages,
            "kpeers": k,
            "represents": m,
        }
        if self.store_mode:
            # m client streams to the store, each at the measured per-stream ceiling
            nominal = min(self._node_caps(job.job_id).nic_out * m,
                          self.cluster.object_store_bandwidth_gbps)
            ok = yield from self._capacity_transfer(
                worker, (worker,), (generation,), size_gb=total_gb,
                streams=[[(worker.name, RES_OUT), (STORE_NODE, RES_IN),
                          (STORE_NODE, RES_DISK)]],
                iteration=iteration,
                operation="checkpoint_stage_dram_to_object_store",
                source=f"{worker.name}/dram", destination=f"{STORE_NODE}",
                details={**common, "path": "store"}, nominal_gbps=nominal,
                demand_per_stream=m * self.store_stream_gbps,
                stream_weights=[float(m)])
            if ok:
                self.stats["store_flushes"] += m
                self.stats["store_gb_written"] += total_gb   # resource bill
                self._put_store_copy(job, worker, iteration, total_gb)
            return ok
        self.registry.set_flushing(worker, True)
        try:
            want = k * m
            # L1.5 dram stripe (see dram_by_job): shard-pieces land in donor
            # DRAM; per-shard piece size checked at the full-width estimate.
            dram_wave = bool(self.dram_by_job.get(job.job_id, False)) and want > 0
            dram_piece_est = shard_gb / max(k, 1)
            granted = self.registry.reserve_shards(
                want, exclude_job=job.job_id, allow_partial=self.partial_grants,
                mode=self.matching, spread_jobs=self.spread_jobs,
                window_bias=self.window_bias,
                window_of=self._window_of,
                tier=("dram" if dram_wave else "ssd"),
                piece_gb=(dram_piece_est if dram_wave else 0.0),
                owner=worker.name) if want > 0 else []
            rack_fb = ({"rack_fallback": True}
                       if self.registry.last_rack_fallback else {})
            taken = sum(t for _s, t in granted)
            per_shard_gb = total_gb / taken if taken else 0.0
            if dram_wave and granted and taken < want:
                # re-shard across fewer donors: true up the byte ledger to the
                # actual (bigger) per-shard piece so release accounting is exact
                for slot, take in granted:
                    self.registry.adjust_dram_gb(
                        slot.worker.name, take * (per_shard_gb - dram_piece_est))
            if not granted:
                # fallback: the cohort's own local SSDs (m disks; NOT node-loss durable)
                self.stats["fallback_flushes"] += m
                ok = yield from self._capacity_transfer(
                    worker, (worker,), (generation,), size_gb=total_gb,
                    streams=[[(worker.name, RES_DISK)]], iteration=iteration,
                    operation="checkpoint_dram_to_local_ssd_chunk",
                    source=f"{worker.name}/dram", destination=f"{worker.name}/ssd",
                    details={**common, "path": "local_fallback"},
                    nominal_gbps=self._rate(job.job_id, "disk") * m, network=False,
                    stream_weights=[float(m)])
                if ok:
                    self._put_copy(owner_rank=worker.rank, location_rank=worker.rank,
                                   tier="ssd", iteration=iteration, size_gb=total_gb)
                return ok
            if taken < want:
                self.stats["partial_grants"] += 1
            self.stats["peer_flushes"] += m
            parity_mode = (not dram_wave) and self.backstop in ("parity",
                                                                "parity_cutoff")
            cutoff = self.backstop == "parity_cutoff"
            if parity_mode:
                # cohort XOR parity: the data streams (reserve_shards) + 1 RAID-4
                # parity stream on a DISTINCT donor sized to the largest data
                # stream, so losing any single stream is peer-side reconstructable
                # (any n_data of n_data+1 XOR-reconstruct). Under parity_cutoff the
                # wave completes at the n_data-th landed stream and the slowest
                # straggler detaches to background; under plain parity it waits for
                # all n_data+1 (the tail-inflated middle mode).
                n_data = len(granted)
                stream_bytes = [total_gb * t / taken for _s, t in granted]
                max_take = max(t for _s, t in granted)
                exclude = {s.worker.name for s, _t in granted}
                parity_slot = self.registry.reserve_one_slot(
                    max_take, job.job_id, exclude_names=exclude)
                if parity_slot is not None:
                    self.stats["parity_flushes"] += m
                    pieces = []
                    for i, (slot, take) in enumerate(granted):
                        caps = self._node_caps(slot.job_id)
                        pieces.append({
                            "uses": [(worker.name, RES_OUT),
                                     (slot.worker.name, RES_IN),
                                     (slot.worker.name, RES_DISK)],
                            "demand": take * min(caps.nic_in, caps.disk_w),
                            "weight": float(take), "bytes": stream_bytes[i],
                            "shards": take, "donor": slot.worker.name,
                            "donor_job": slot.job_id, "donor_worker": slot.worker,
                            "watched": (worker, slot.worker),
                            "gens": (generation, slot.worker.failure_generation),
                            "release": (lambda e=(slot, take):
                                        self.registry.release_shards([e])),
                            "meta": {"piece_idx": i, "parity": False},
                        })
                    ps, pt = parity_slot
                    pcaps = self._node_caps(ps.job_id)
                    pieces.append({
                        "uses": [(worker.name, RES_OUT), (ps.worker.name, RES_IN),
                                 (ps.worker.name, RES_DISK)],
                        "demand": pt * min(pcaps.nic_in, pcaps.disk_w),
                        "weight": float(pt), "bytes": max(stream_bytes),
                        "shards": pt, "donor": ps.worker.name,
                        "donor_job": ps.job_id, "donor_worker": ps.worker,
                        "watched": (worker, ps.worker),
                        "gens": (generation, ps.worker.failure_generation),
                        "release": (lambda e=(ps, pt):
                                    self.registry.release_shards([e])),
                        "meta": {"piece_idx": n_data, "parity": True},
                    })
                    p_watched = (worker, *[p["donor_worker"] for p in pieces])
                    p_gens = (generation,
                              *[p["donor_worker"].failure_generation
                                for p in pieces])
                    ok = yield from self._parity_cutoff_transfer(
                        worker, job, p_watched, p_gens, pieces=pieces,
                        data_streams=n_data, iteration=iteration, common=common,
                        cutoff=cutoff)
                    if not ok:
                        self.stats["peer_flush_aborted"] += m
                    return ok
                # no distinct parity donor free this wave -> plain cohort wave.
            donor_names = [s.worker.name for s, _t in granted]
            watched = (worker, *[s.worker for s, _t in granted])
            gens = (generation, *[s.worker.failure_generation for s, _t in granted])
            streams, demands, sweights = [], [], []
            slowest = 999.0
            for slot, take in granted:
                caps = self._node_caps(slot.job_id)
                if dram_wave:
                    # L1.5: piece lands in donor DRAM at NIC_in — no donor disk
                    # on the write path (paid later at demotion)
                    per_shard = caps.nic_in
                    slowest = min(slowest, caps.nic_in)
                    streams.append([(worker.name, RES_OUT),
                                    (slot.worker.name, RES_IN)])
                else:
                    per_shard = min(caps.nic_in, caps.disk_w)  # per-NODE ingest bound
                    slowest = min(slowest, caps.disk_w)
                    streams.append([(worker.name, RES_OUT),
                                    (slot.worker.name, RES_IN),
                                    (slot.worker.name, RES_DISK)])
                demands.append(take * per_shard)
                sweights.append(float(take))
            eff_bw = min((self._node_caps(job.job_id).nic_out if dram_wave
                          else self._rate(job.job_id, "nic")) * m,
                         taken * slowest)
            try:
                ok = yield from self._capacity_transfer(
                    worker, watched, gens, size_gb=total_gb, streams=streams,
                    iteration=iteration,
                    operation="checkpoint_dram_to_crossjob_peers_chunk",
                    source=f"{worker.name}/dram",
                    destination="+".join(
                        f"{n}/{'dram' if dram_wave else 'ssd'}"
                        for n in donor_names),
                    details={**common,
                             "path": ("peer_dram" if dram_wave else
                                      "peer_partial" if taken < want else "peer"),
                             **({"tier": "dram", "partial": taken < want}
                                if dram_wave else {}),
                             "donors": donor_names,
                             "shards_by_donor": {s.worker.name: t
                                                 for s, t in granted},
                             **rack_fb},
                    nominal_gbps=eff_bw, demand_per_stream=demands,
                    stream_weights=sweights)
            finally:
                if not dram_wave:
                    self.registry.release_shards(granted)
            if not ok:
                self.stats["peer_flush_aborted"] += m
                if dram_wave:
                    for slot, take in granted:   # nothing landed: free the DRAM
                        self.registry.release_dram(
                            slot.worker.name, take * per_shard_gb, take)
                return False
            if dram_wave:
                self.stats["dram_peer_flushes"] += m
            drain_info = []
            demote_info = []
            for i, (slot, take) in enumerate(granted):
                piece_gb = total_gb * take / taken
                key = (job.job_id, worker.rank, slot.worker.name)
                current = self.peer_copies.get(key)
                if (
                    iteration > getattr(worker, "current_iteration", iteration)
                    or current is not None
                    and current["iteration"] > iteration
                ):
                    if dram_wave:   # stale piece: daemon discards the buffer now
                        self.registry.release_dram(slot.worker.name,
                                                   piece_gb, take)
                    continue
                if dram_wave and current is not None:
                    if current.get("tier") == "dram":
                        self._book_dram_release(key, current, lost=False)
                        self.stats["dram_superseded"] += 1
                    else:   # write-then-rename (see the m=1 path)
                        self.dram_shadow[key] = current
                meta = {
                    "iteration": iteration,
                    "size_gb": piece_gb,
                    "donor_job": slot.job_id,
                    "k": len(granted),
                    "piece_idx": i,
                }
                if dram_wave:
                    meta["tier"] = "dram"
                    meta["shards"] = take
                    meta["hosted_at"] = self.backend.now
                else:
                    # resource bill (see the m=1 path)
                    if current is not None:
                        self._bill_hosted_ssd_close(key, current)
                    self._bill_hosted_ssd_open(key)
                self.peer_copies[key] = meta
                if dram_wave:
                    demote_info.append((slot.worker, slot.job_id, key, meta, take))
                else:
                    drain_info.append((slot.worker, slot.job_id, i, piece_gb))
            if dram_wave:
                self._schedule_dram_demotions(worker, job, iteration, demote_info)
            else:
                self._schedule_wave_drain(worker, job, iteration, drain_info)
            return True
        finally:
            self.registry.set_flushing(worker, False)

    # ---- in-job SSD placement (rack_failure_spec.md, 2026-07-27) -------------
    def _persist_injob_ssd(self, worker, job, *, m, iteration, generation,
                           shard_gb, checkpoint_group):
        """Placement `injob_ssd`: the wave lands on the RING-NEIGHBOR node of
        the SAME job — one piece, k=1, on that node's SSD (Gemini's ring pick,
        but on disk instead of DRAM). No donor duty/slots and no DRAM rent; the
        piece survives the owner's process failure and the host's REBOOT (it is
        on disk), dies with the host node (driver's hosted sweep), and goes
        dark with the host rack — the deliberate same-rack-by-packing bet the
        rack 2x2 prices. Cohort workers (m>1) ship their m per-rank shards to
        the neighbor cohort's m nodes in aggregate (one bundled stream, per-
        node ingest bound per shard). Recovery-source label: injob_ssd_replica
        (recover() below)."""
        assert self.capacity_mode, "injob_ssd placement requires capacity mode"
        total_gb = m * shard_gb
        common = {
            "checkpoint_strategy": self.strategy_name,
            "checkpoint_group": checkpoint_group,
            "checkpoint_owner_rank": worker.rank,
            "shard": worker.pipeline_stage,
            "shard_count": job.pipeline_stages,
            "kpeers": 1,
            "represents": m,
        }
        others = sorted((w for w in self.workers.values()
                         if w.rank != worker.rank), key=lambda w: w.rank)
        neighbor = others[worker.rank % len(others)] if others else None
        if neighbor is None or getattr(neighbor, "failed", False):
            # degenerate 1-cohort job / neighbor down: own local SSD — the same
            # not-node-loss-durable fallback a refused cross-job wave takes.
            self.stats["fallback_flushes"] += m
            ok = yield from self._capacity_transfer(
                worker, (worker,), (generation,), size_gb=total_gb,
                streams=[[(worker.name, RES_DISK)]], iteration=iteration,
                operation="checkpoint_dram_to_local_ssd_chunk",
                source=f"{worker.name}/dram", destination=f"{worker.name}/ssd",
                details={**common, "path": "local_fallback"},
                nominal_gbps=self._rate(job.job_id, "disk") * m, network=False,
                stream_weights=[float(m)])
            if ok:
                self._put_copy(owner_rank=worker.rank, location_rank=worker.rank,
                               tier="ssd", iteration=iteration, size_gb=total_gb)
            return ok
        caps = self._node_caps(job.job_id)
        per = min(caps.nic_in, caps.disk_w)          # one node-pair per shard
        watched = (worker, neighbor)
        gens = (generation, neighbor.failure_generation)
        ok = yield from self._capacity_transfer(
            worker, watched, gens, size_gb=total_gb,
            streams=[[(worker.name, RES_OUT), (neighbor.name, RES_IN),
                      (neighbor.name, RES_DISK)]],
            iteration=iteration,
            operation="checkpoint_dram_to_injob_ssd_chunk",
            source=f"{worker.name}/dram", destination=f"{neighbor.name}/ssd",
            details={**common, "path": "injob_ssd", "donors": [neighbor.name],
                     "shards_by_donor": {neighbor.name: m}},
            nominal_gbps=min(caps.nic_out * m, m * per),
            demand_per_stream=[m * per], stream_weights=[float(m)])
        if not ok:
            self.stats["peer_flush_aborted"] += m
            return False
        self.stats["injob_ssd_flushes"] += m
        key = (job.job_id, worker.rank, neighbor.name)
        current = self.peer_copies.get(key)
        if (iteration > getattr(worker, "current_iteration", iteration)
                or (current is not None and current["iteration"] > iteration)):
            return True
        if current is not None:                     # resource bill: overwrite
            self._bill_hosted_ssd_close(key, current)
        self._bill_hosted_ssd_open(key)
        self.peer_copies[key] = {
            "iteration": iteration, "size_gb": total_gb,
            "donor_job": job.job_id, "k": 1, "data_k": 1,
            "piece_idx": 0, "injob_ssd": True,
        }
        return True

    # ---- straggler-cutoff XOR parity (Sam's design, 2026-07-20) --------------
    # A parity wave stripes n data pieces + 1 XOR parity piece across n+1 donors
    # (m=1: n=k pieces of shard_gb/k; cohort: the reserve_shards data streams + 1
    # RAID-4 parity stream). The wave is RECOVERABLE the instant ANY n of the n+1
    # pieces land (peer-side XOR, _best_parity_set / _parity_recover — no L3).
    #
    #   backstop == "parity_cutoff": COMPLETE at the n-th finished stream (the
    #       n-th order statistic) — the single slowest straggler DETACHES to a
    #       background flow. If it lands its piece is recorded too (n+1 pieces =
    #       single-loss-tolerant); if its donor dies or a newer wave supersedes
    #       it, it is dropped silently. Wave time: max-of-(n+1) -> n-th order stat.
    #   backstop == "parity": the SAME stripe, but waits for ALL n+1 (no cutoff) —
    #       the tail-inflated middle mode kept for the three-way comparison.
    #
    # Pieces record in peer_copies AS THEY LAND (not only at wave end), so a
    # straggler that arrives after the wave "completed" still becomes recoverable
    # state. Both capacity engines support the threshold; the event engine fires a
    # partial completion at the k-th stream (capacity._GroupState.threshold), the
    # polling engine counts finished streams in its own loop.

    def _stream_ceiling(self, uses, demand):
        per = demand
        for node, res in uses:
            caps = self.capacity.nodes.get(node)
            if caps is not None:
                per = min(per, caps.cap(res))
        return max(per, 1e-9)

    def _put_store_copy(self, job, worker, iteration, size_gb) -> bool:
        if iteration > getattr(worker, "current_iteration", iteration):
            return False
        key = (job.job_id, worker.rank)
        current = self.store_copies.get(key)
        if current is not None and current["iteration"] > iteration:
            return False
        self.store_copies[key] = {"iteration": iteration, "size_gb": size_gb}
        return True

    def _put_peer_piece(self, job, worker, iteration, piece, *, total, data_k):
        if iteration > getattr(worker, "current_iteration", iteration):
            return
        key = (job.job_id, worker.rank, piece["donor"])
        cur = self.peer_copies.get(key)
        # a background straggler can land AFTER a newer wave already placed a
        # piece on this same donor — never overwrite a newer iteration backwards
        # (mirrors _put_copy's guard; without it a late straggler corrupts the
        # donor's newest-piece record and poisons _best_peer_set accounting).
        if cur is not None and cur["iteration"] > iteration:
            return
        if cur is not None:                         # resource bill: overwrite
            self._bill_hosted_ssd_close(key, cur)
        self._bill_hosted_ssd_open(key)
        self.peer_copies[key] = {
            "iteration": iteration, "size_gb": piece["bytes"],
            "donor_job": piece["donor_job"], "k": total, "data_k": data_k,
            "piece_idx": piece["meta"]["piece_idx"],
            "parity": piece["meta"]["parity"]}

    def _wave_superseded(self, job_id, rank, iteration):
        """A strictly-newer durable peer wave for (job, rank) already exists — a
        detached straggler landing this late is stale (its iteration is already
        rolled forward), so drop it silently (task: 'superseded -> drop')."""
        for (j, r, _d), meta in self.peer_copies.items():
            if j == job_id and r == rank and meta["iteration"] > iteration:
                return True
        return False

    def _record_cutoff_wave(self, worker, iteration, pieces, landed, straggler, *,
                            start, common, data_streams, cutoff):
        """One flush event over the streams that LANDED by the cutoff (data_gb =
        their bytes, duration = the n-th order statistic). Understated per-donor
        rate (<= actual) keeps validator I2 green; the straggler is named."""
        bytes_landed = sum(pieces[i]["bytes"] for i in landed)
        ceil = max(sum(self._stream_ceiling(pieces[i]["uses"], pieces[i]["demand"])
                       for i in landed), 1e-9)
        base = bytes_landed / ceil
        now = self.backend.now
        donor_names = [pieces[i]["donor"] for i in landed]
        self._record(
            worker, start=start, category="Checkpoint",
            operation="checkpoint_dram_to_crossjob_peers_chunk",
            resources=["CPU", "NETWORK"], iteration=iteration,
            source=f"{worker.name}/dram",
            destination="+".join(f"{n}/ssd" for n in donor_names),
            data_gb=bytes_landed,
            details={**common, "path": "peer_parity", "donors": donor_names,
                     "shards_by_donor": {pieces[i]["donor"]: pieces[i]["shards"]
                                         for i in landed},
                     "data_k": data_streams, "parity": True, "cutoff": cutoff,
                     "straggler": (pieces[straggler[0]]["donor"]
                                   if straggler else None),
                     "base_duration": base,
                     "effective_slowdown": ((now - start) / base
                                            if base > 0 else 1.0),
                     "capacity_mode": True})

    def _record_straggler(self, worker, job, iteration, piece, *, start, common,
                          total, data_streams):
        """The detached straggler landed (background): record its own single-donor
        flush event and its peer_copy (now n+1 pieces = single-loss-tolerant)."""
        now = self.backend.now
        base = piece["bytes"] / self._stream_ceiling(piece["uses"], piece["demand"])
        self._record(
            worker, start=start, category="Checkpoint",
            operation="checkpoint_dram_to_crossjob_peers_chunk",
            resources=["CPU", "NETWORK"], iteration=iteration,
            source=f"{worker.name}/dram", destination=f'{piece["donor"]}/ssd',
            data_gb=piece["bytes"],
            details={**common, "path": "peer_parity_straggler",
                     "donors": [piece["donor"]],
                     "shards_by_donor": {piece["donor"]: piece["shards"]},
                     "data_k": data_streams, "parity": piece["meta"]["parity"],
                     "cutoff": True, "straggler_landed": True,
                     "base_duration": base,
                     "effective_slowdown": ((now - start) / base
                                            if base > 0 else 1.0),
                     "capacity_mode": True})
        self._put_peer_piece(job, worker, iteration, piece,
                             total=total, data_k=data_streams)

    def _parity_cutoff_transfer(self, worker, job, watched, gens, *, pieces,
                                data_streams, iteration, common, cutoff):
        assert self.capacity_mode, "parity striping requires capacity mode"
        impl = (self._parity_cutoff_event if self.engine == "event"
                else self._parity_cutoff_polling)
        ok = yield from impl(worker, job, watched, gens, pieces=pieces,
                             data_streams=data_streams, iteration=iteration,
                             common=common, cutoff=cutoff)
        return ok

    def _parity_cutoff_event(self, worker, job, watched, gens, *, pieces,
                             data_streams, iteration, common, cutoff):
        group = object()
        flows = [self.capacity.make_flow(p["uses"], demand=p["demand"],
                                         group=group, weight=p["weight"])
                 for p in pieces]
        for f, p in zip(flows, pieces):
            f.remaining = p["bytes"]
        total = len(pieces)
        threshold = data_streams if cutoff else total
        net_workers = tuple(watched)
        net_fabric = self.ckpt_fabric(worker.job_id)   # CL-014: owner's fabric
        # CL-016: offered rate of the whole parity wave — per piece the min
        # physical cap along its path bounded by the piece demand, summed (the
        # same provenance as _capacity_transfer's `ceiling`: declared scenario
        # constants only, nothing fitted).
        wave_rate = 0.0
        for p in pieces:
            per_piece = p["demand"]
            for node, res in p["uses"]:
                caps = self.capacity.nodes.get(node)
                if caps is not None:
                    per_piece = min(per_piece, caps.cap(res))
            wave_rate += per_piece
        cpu_request = worker.cpu.request(priority=10)
        yield cpu_request
        bump_network_tasks(net_workers, net_fabric, +1, rate_gbps=wave_rate)
        released = {"fg": False}

        def _release_fg():
            if not released["fg"]:
                bump_network_tasks(net_workers, net_fabric, -1,
                                   rate_gbps=wave_rate)
                worker.cpu.release(cpu_request)
                released["fg"] = True

        start = self.backend.now
        armed = False
        try:
            if not self._checkpoint_generation_valid(watched, gens):
                _release_fg()
                for p in pieces:
                    p["release"]()
                return False
            yield self.backend.timeout(0)          # settle tick (engine parity)
            if not self._checkpoint_generation_valid(watched, gens):
                _release_fg()
                for p in pieces:
                    p["release"]()
                return False
            done_ev, abort_ev, partial_ev = self.capacity.ev_arm(
                group, flows, watched, threshold=threshold)
            armed = True
            yield (partial_ev | abort_ev)
            if abort_ev.triggered and not partial_ev.triggered:
                _release_fg()
                self.capacity.ev_dismiss(group)
                armed = False
                for p in pieces:
                    p["release"]()
                return False
            landed = [i for i, f in enumerate(flows) if f.remaining <= BYTE_EPS]
            straggler = [i for i, f in enumerate(flows) if f.remaining > BYTE_EPS]
            self._record_cutoff_wave(worker, iteration, pieces, landed, straggler,
                                     start=start, common=common,
                                     data_streams=data_streams, cutoff=cutoff)
            for i in landed:
                self._put_peer_piece(job, worker, iteration, pieces[i],
                                     total=total, data_k=data_streams)
                pieces[i]["release"]()
            _release_fg()
            if not straggler:
                self.capacity.ev_dismiss(group)     # all landed at once; empty
                armed = False
                return True
            si = straggler[0]                        # the single slowest stream
            self.stats["cutoff_waves"] += 1

            def _bg():
                try:
                    yield (done_ev | abort_ev)
                    if (done_ev.triggered and not abort_ev.triggered
                            and flows[si].remaining <= BYTE_EPS
                            and not self._wave_superseded(
                                job.job_id, worker.rank, iteration)):
                        self._record_straggler(
                            worker, job, iteration, pieces[si], start=start,
                            common=common, total=total, data_streams=data_streams)
                        self.stats["cutoff_straggler_landed"] += 1
                    else:
                        self.stats["cutoff_straggler_dropped"] += 1
                finally:
                    pieces[si]["release"]()
                    self.capacity.ev_dismiss(group)
            self.background_flushes.append(self.backend.process(_bg()))
            armed = False          # the group's teardown now belongs to _bg
            return True
        finally:
            if armed:
                self.capacity.ev_dismiss(group)

    def _parity_cutoff_polling(self, worker, job, watched, gens, *, pieces,
                               data_streams, iteration, common, cutoff):
        group = object()
        flows = [self.capacity.open(p["uses"], demand=p["demand"], group=group,
                                    weight=p["weight"]) for p in pieces]
        open_flows = list(flows)
        remaining = [p["bytes"] for p in pieces]
        total = len(pieces)
        threshold = data_streams if cutoff else total
        net_workers = tuple(watched)
        net_fabric = self.ckpt_fabric(worker.job_id)   # CL-014: owner's fabric
        # CL-016: offered rate of the whole parity wave — per piece the min
        # physical cap along its path bounded by the piece demand, summed (the
        # same provenance as _capacity_transfer's `ceiling`: declared scenario
        # constants only, nothing fitted).
        wave_rate = 0.0
        for p in pieces:
            per_piece = p["demand"]
            for node, res in p["uses"]:
                caps = self.capacity.nodes.get(node)
                if caps is not None:
                    per_piece = min(per_piece, caps.cap(res))
            wave_rate += per_piece
        cpu_request = worker.cpu.request(priority=10)
        yield cpu_request
        bump_network_tasks(net_workers, net_fabric, +1, rate_gbps=wave_rate)
        released = {"fg": False}

        def _release_fg():
            if not released["fg"]:
                bump_network_tasks(net_workers, net_fabric, -1,
                                   rate_gbps=wave_rate)
                worker.cpu.release(cpu_request)
                released["fg"] = True

        def _close(i):
            if open_flows[i] is not None:
                self.capacity.close(open_flows[i])
                open_flows[i] = None

        start = self.backend.now
        if not self._checkpoint_generation_valid(watched, gens):
            _release_fg()
            for i in range(total):
                _close(i)
            for p in pieces:
                p["release"]()
            return False
        yield self.backend.timeout(0)               # settle tick
        first_rates = [self.capacity.flow_rate(f) for f in open_flows]
        est = sum(remaining) / max(sum(first_rates), 1e-9)
        quantum = max(0.1, min(max(self.contention_quantum_seconds, 0.5),
                               est / 10.0))
        finished = 0
        aborted = False
        while finished < threshold:
            if not self._checkpoint_generation_valid(watched, gens):
                aborted = True
                break
            rates = [self.capacity.flow_rate(f) if f is not None else 0.0
                     for f in open_flows]
            dt = quantum
            for i, r in enumerate(rates):
                if remaining[i] > 1e-9 and r > 1e-12:
                    dt = min(dt, remaining[i] / r)
            yield self.backend.timeout(max(dt, 1e-6))
            rates_post = [self.capacity.flow_rate(f) if f is not None else 0.0
                          for f in open_flows]
            for i, (r0, r1) in enumerate(zip(rates, rates_post)):
                if remaining[i] > 1e-9:
                    remaining[i] = max(0.0, remaining[i] - min(r0, r1) * dt)
                    if remaining[i] <= 1e-9 and open_flows[i] is not None:
                        _close(i)
                        finished += 1
        if aborted:
            _release_fg()
            for i in range(total):
                _close(i)
            for p in pieces:
                p["release"]()
            return False
        landed = [i for i in range(total) if remaining[i] <= 1e-9]
        straggler = [i for i in range(total) if remaining[i] > 1e-9]
        self._record_cutoff_wave(worker, iteration, pieces, landed, straggler,
                                 start=start, common=common,
                                 data_streams=data_streams, cutoff=cutoff)
        for i in landed:
            self._put_peer_piece(job, worker, iteration, pieces[i],
                                 total=total, data_k=data_streams)
            pieces[i]["release"]()
        _release_fg()
        if not straggler:
            return True
        si = straggler[0]
        self.stats["cutoff_waves"] += 1

        def _bg():
            aborted2 = False
            try:
                while remaining[si] > 1e-9:
                    if not self._checkpoint_generation_valid(
                            pieces[si]["watched"], pieces[si]["gens"]):
                        aborted2 = True
                        break
                    r0 = self.capacity.flow_rate(open_flows[si])
                    dt = quantum
                    if r0 > 1e-12:
                        dt = min(dt, remaining[si] / r0)
                    yield self.backend.timeout(max(dt, 1e-6))
                    r1 = self.capacity.flow_rate(open_flows[si])
                    remaining[si] = max(0.0, remaining[si] - min(r0, r1) * dt)
                if (not aborted2 and not self._wave_superseded(
                        job.job_id, worker.rank, iteration)):
                    self._record_straggler(
                        worker, job, iteration, pieces[si], start=start,
                        common=common, total=total, data_streams=data_streams)
                    self.stats["cutoff_straggler_landed"] += 1
                else:
                    self.stats["cutoff_straggler_dropped"] += 1
            finally:
                _close(si)
                pieces[si]["release"]()
        self.background_flushes.append(self.backend.process(_bg()))
        return True

    def snapshot(
        self,
        worker: CheckpointWorker,
        job,
        *,
        iteration: int,
        checkpoint_epoch: int | None = None,
    ):
        """f1 enactment: GPU->DRAM capture WITHOUT persisting — the cost
        model's cheap high-frequency tier (process failures, 40% of the mix,
        restore from host DRAM; freshness = this cadence, not f2's)."""
        if checkpoint_epoch is None:
            checkpoint_epoch = getattr(worker, "checkpoint_epoch", 0)
        shard_gb = job.checkpoint_gb / job.pipeline_stages
        lock = self.checkpoint_locks[worker.rank]
        with lock.request() as lock_request:
            yield lock_request
            generation = worker.failure_generation
            # CL-017: same nonblocking-capture semantics as the persist-wave
            # capture in tiered.checkpoint() — no GPU lock when the flag is
            # on; capture duration and DRAM-copy placement unchanged.
            if self.nonblocking_capture:
                gpu_request = None
            else:
                gpu_request = worker.gpu.request()
                yield gpu_request
            cpu_request = worker.cpu.request(priority=0)
            yield cpu_request
            try:
                if (
                    worker.failed
                    or worker.failure_generation != generation
                    or getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch
                ):
                    return False
                start = self.backend.now
                duration = shard_gb / self.cluster.gpu_cpu_bandwidth_gbps
                completed = yield from self._checkpoint_timeout(
                    (worker,), generations=(generation,), duration=duration)
                if not completed:
                    return False
                if getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch:
                    return False
                self._put_copy(owner_rank=worker.rank, location_rank=worker.rank,
                               tier="dram", iteration=iteration, size_gb=shard_gb,
                               bill_gb=shard_gb * getattr(worker, "represents", 1))
                self._record(
                    worker, start=start, category="Checkpoint",
                    operation="checkpoint_stage_gpu_to_dram",
                    resources=["CPU", "GPU"], iteration=iteration,
                    data_gb=shard_gb,
                    details={"checkpoint_strategy": self.strategy_name,
                             "checkpoint_group": f"{job.job_id}-snap-{iteration}",
                             "snapshot_only": True,
                             "represents": getattr(worker, "represents", 1)})
                self.stats["snapshots"] += getattr(worker, "represents", 1)
                return True
            finally:
                if gpu_request is not None:
                    worker.gpu.release(gpu_request)
                worker.cpu.release(cpu_request)

    def backstop_upload(
        self,
        worker: CheckpointWorker,
        job,
        *,
        iteration: int,
        checkpoint_epoch: int | None = None,
    ):
        """f3 enactment: rare async store upload as the durability BACKSTOP
        (bounded loss even when donors and local copies are all gone —
        e.g. eviction wiped hosted shards). Runs in ours arms; store_mode
        arms route every persist to the store and don't need this."""
        if checkpoint_epoch is None:
            checkpoint_epoch = getattr(worker, "checkpoint_epoch", 0)
        m = getattr(worker, "represents", 1)
        shard_gb = job.checkpoint_gb / job.pipeline_stages
        total = m * shard_gb
        generation = worker.failure_generation
        ok = yield from self._capacity_transfer(
            worker, (worker,), (generation,), size_gb=total,
            streams=[[(worker.name, RES_OUT), (STORE_NODE, RES_IN),
                      (STORE_NODE, RES_DISK)]],
            iteration=iteration,
            operation="checkpoint_stage_dram_to_object_store",
            source=f"{worker.name}/dram", destination=STORE_NODE,
            details={"checkpoint_strategy": self.strategy_name,
                     "checkpoint_group": f"{job.job_id}-iteration-{iteration}",
                     "checkpoint_owner_rank": worker.rank,
                     "path": "store_backstop", "represents": m},
            nominal_gbps=self.cluster.object_store_bandwidth_gbps,
            demand_per_stream=m * self.store_stream_gbps,
            stream_weights=[float(m)],
            hold_cpu=False,
            network=False)   # background QoS: shares capacity, but does not
                             # throttle the job's all-reduce like foreground
                             # flushes do (first 3-tier run: +21% completion
                             # purely from backstop-held coupling counters)
        epoch_valid = getattr(worker, "checkpoint_epoch", 0) == checkpoint_epoch
        if ok:
            # resource bill: bytes PHYSICALLY shipped to the store are charged
            # even when an epoch bump refuses registration (the PUT happened;
            # the trace-side extractor sees the completed transfer either way).
            self.stats["store_gb_written"] += total
        if ok and epoch_valid:
            self.stats["backstop_uploads"] += m
            self._put_store_copy(job, worker, iteration, total)
        return ok and epoch_valid

    # ---- donor-drain L3 backstop (goal 2026-07-19) --------------------------
    def _schedule_wave_drain(self, worker, job, iteration, drain_info):
        """Hand the donors' pieces of THIS wave to the L3 drain (background).
        No-op under owner_push (the legacy backstop_upload path).
        drain_info: [(donor_worker, donor_job_id, piece_idx, piece_gb)].

        GATED AT THE SOLVED f3 CADENCE (drain_every_waves, from the policy's
        store_every / checkpoint_every ratio): draining EVERY wave would feed
        the store each job's full state at L2 frequency — the ~1.5 TB/s
        aggregate a 500 GB/s store cannot absorb (Sam caught this via the
        wall-clock: every-wave drains also tripled event volume, 5h->7.8h).
        Same-iteration piece consistency and stitch recovery are unaffected;
        only the L3 refresh rate returns to the physically honest f3."""
        if self.backstop != "donor_drain" or not drain_info:
            return
        gate = max(1, int(getattr(self, "drain_every_waves", 1)))
        n = self._drain_wave_counter.get(worker.rank, 0) + 1
        self._drain_wave_counter[worker.rank] = n
        if n % gate != 0:
            return
        proc = self.backend.process(
            self._drain_wave_to_l3(worker, job, iteration=iteration,
                                   drain_info=drain_info,
                                   checkpoint_epoch=getattr(
                                       worker, "checkpoint_epoch", 0
                                   )))
        self.background_flushes.append(proc)

    def _drain_start_delay(self, donors) -> float:
        """Best-effort idle-window nudge — REUSES the D9 donor-window machinery
        (_window_of). If a donor's OWN durable flush is imminent (< WINDOW_GUARD_S
        away) the background drain's donor-disk read would collide with the donor
        writing its own wave, so wait until just past that window. No-op when
        phasing is off (slot_period None) -> drains start immediately; background
        QoS (below) is the hard guarantee, this only reduces disk contention."""
        if not self.slot_period:
            return 0.0
        delay = 0.0
        for d in donors:
            slot = self.registry.slots.get(d.name)
            if slot is None:
                continue
            w = self._window_of(slot)        # secs until donor's next flush slot
            if w < WINDOW_GUARD_S:           # donor about to flush -> wait past it
                delay = max(delay, w + WINDOW_GUARD_S)
        return min(delay, self.slot_period)

    def _drain_wave_to_l3(
        self,
        worker,
        job,
        *,
        iteration,
        drain_info,
        checkpoint_epoch=None,
    ):
        """One donor forwards its hosted piece to L3 per stream: donor SSD-read +
        donor NIC-out -> store in/disk, BACKGROUND QoS (no CPU hold, NOT coupled
        to training). Per-stream demand = min(donor disk, donor nic, store
        single-stream ceiling). On success each landed piece is recorded in
        l3_pieces at the WAVE's iteration and the donor's drained bytes +
        NIC-seconds are booked (donor_drain_nic_pct in the results JSON)."""
        if checkpoint_epoch is None:
            checkpoint_epoch = getattr(worker, "checkpoint_epoch", 0)
        if getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch:
            return False
        if iteration > getattr(worker, "current_iteration", iteration):
            return False
        donors0 = [d for d, _dj, _pi, _pg in drain_info]
        delay = self._drain_start_delay(donors0)
        if delay > 0:
            yield self.backend.timeout(delay)
        if getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch:
            return False
        # a donor already lost before its drain ran -> its piece is gone (the
        # driver's donor-loss path already deleted the peer copy); drop it.
        live = [(d, dj, pi, pg) for (d, dj, pi, pg) in drain_info
                if not getattr(d, "failed", False)]
        if not live:
            self.stats["l3_drain_aborted"] += len(drain_info)
            return False
        donors = [d for d, _dj, _pi, _pg in live]
        gens = tuple(d.failure_generation for d in donors)
        total_gb = sum(pg for _d, _dj, _pi, pg in live)
        streams, demands, weights = [], [], []
        drain_pieces, nic_seconds = {}, 0.0
        for d, dj, pi, pg in live:
            caps = self._node_caps(dj)
            # per-stream demand caps at the store's single-stream ingest ceiling
            # (store_stream_gbps); the DONOR's disk-read + NIC-out are the shared
            # capacities (contend with hosting writes on that donor). Every drain
            # also consumes the shared store ingress and disk capacities, so all
            # concurrent waves obey the configured aggregate durability limit.
            demands.append(min(caps.disk_w, caps.nic_out, self.store_stream_gbps))
            streams.append([
                (d.name, RES_DISK),
                (d.name, RES_OUT),
                (STORE_NODE, RES_IN),
                (STORE_NODE, RES_DISK),
            ])
            # background-QoS weight (yields the shared donor disk to foreground
            # flushes); intra-transfer byte apportioning is preserved because the
            # constant factor cancels in weight/sum(weights).
            weights.append(pg * DRAIN_QOS_WEIGHT)
            drain_pieces[d.name] = pg
            nic_seconds += pg / max(caps.nic_out, 1e-9)
        ok = yield from self._capacity_transfer(
            worker, tuple(donors), gens, size_gb=total_gb, streams=streams,
            iteration=iteration,
            operation="checkpoint_donor_ssd_to_l3_drain_chunk",
            source="+".join(f"{d.name}/ssd" for d in donors),
            destination=STORE_NODE,
            details={"checkpoint_strategy": self.strategy_name,
                     "checkpoint_group": f"{job.job_id}-iteration-{iteration}",
                     "checkpoint_owner_rank": worker.rank,
                     "path": "l3_drain", "drain": True,
                     "donors": [d.name for d in donors],
                     "drain_pieces": drain_pieces},
            nominal_gbps=max(sum(demands), 1e-9),
            demand_per_stream=demands, stream_weights=weights,
            hold_cpu=False, network=False)
        if not ok:
            self.stats["l3_drain_aborted"] += len(live)
            return False
        # resource bill: the drain's bytes PHYSICALLY landed on the store even
        # when the epoch check below refuses registration (drain_bytes_gb /
        # l3_drains keep their archived registered-only semantics unchanged).
        self.stats["store_gb_written"] += total_gb
        if getattr(worker, "checkpoint_epoch", 0) != checkpoint_epoch:
            return False
        for d, dj, pi, pg in live:
            key = (job.job_id, worker.rank, pi)
            current = self.l3_pieces.get(key)
            if current is not None and current["iteration"] > iteration:
                continue
            self.l3_pieces[key] = {
                "iteration": iteration, "size_gb": pg, "k": len(drain_info)}
        self.stats["l3_drains"] += len(live)
        self.stats["drain_bytes_gb"] += total_gb
        self.stats["drain_nic_seconds"] += nic_seconds
        return True

    # ---- resource-bill accounting (2026-07-28, PURE accounting) --------------
    # ckpt_dram_gbh / ckpt_ssd_gbh / store_gb_written for the priced bill.
    # None of these helpers touch RNG, capacities, or the event queue.
    def _bill_copy_end_event(self, key, copy, gb) -> None:
        """BilledCopies end hook -> zero-duration Accounting trace event (the
        extractor's death signal for L1 snapshots / local-SSD copies / in-job
        DRAM replicas). details carries NO `path` / `checkpoint_strategy`
        keys, so checkpoint_paths counting and the closed-loop flush-span
        accumulator (slot repacking) never see it."""
        w = self.workers.get(key[1])
        self.logger.record(
            start=self.backend.now, end=self.backend.now, run_id=self.run_id,
            job_id=getattr(w, "job_id", None), rank=key[0],
            node=getattr(w, "name", f"rank-{key[1]}"),
            category="Accounting", operation="ckpt_copy_end",
            iteration=copy.iteration,
            details={"tier": copy.tier, "location_rank": key[1]})

    def _bill_hosted_ssd_open(self, key) -> None:
        """Start the SSD residency clock for one hosted piece (cross-job
        stripe piece, injob_ssd replica, or a just-demoted L1.5 piece)."""
        clocks = getattr(self, "_ssd_hosted_since", None)
        if clocks is None:                 # bare test shells skip __init__
            return
        clocks[key] = self.backend.now

    def _bill_hosted_ssd_close(self, key, meta) -> None:
        """Book one hosted piece's SSD residency GB-seconds and emit the
        Accounting death event. Idempotent per open (clock is popped); callers
        pass the meta whose bytes actually occupied the disk (for the L1.5
        write-then-rename case that is the SHADOW, whose file stays on disk
        while a newer dram-tier piece is hosted — the clock deliberately keeps
        running across that replacement)."""
        clocks = getattr(self, "_ssd_hosted_since", None)
        if clocks is None:                 # bare test shells skip __init__
            return
        t0 = clocks.pop(key, None)
        if t0 is None:
            return
        self.stats["ckpt_hosted_ssd_gb_seconds"] += float(
            (meta or {}).get("size_gb", 0.0)) * max(0.0, self.backend.now - t0)
        self.logger.record(
            start=self.backend.now, end=self.backend.now, run_id=self.run_id,
            job_id=key[0], rank=key[1], node=key[2],
            category="Accounting", operation="ckpt_hosted_end",
            iteration=int((meta or {}).get("iteration", -1)),
            details={"tier": "ssd"})

    def finalize_bill_accounting(self) -> None:
        """End of run: close the residency integrals at the horizon (mirrors
        finalize_dram_accounting). Idempotent."""
        if isinstance(self.copies, BilledCopies):
            self.copies.finalize()
        now = self.backend.now
        for key, t0 in list(getattr(self, "_ssd_hosted_since", {}).items()):
            meta = self.peer_copies.get(key)
            if meta is None or meta.get("tier") == "dram":
                # clock belongs to the write-then-rename shadow (or is orphaned)
                meta = self.dram_shadow.get(key, None if meta is None else meta)
            gb = float((meta or {}).get("size_gb", 0.0))
            if meta is not None and meta.get("tier") == "dram":
                gb = 0.0                       # defensive: no ssd bytes resident
            self.stats["ckpt_hosted_ssd_gb_seconds"] += gb * max(0.0, now - t0)
            self._ssd_hosted_since[key] = now

    # ---- L1.5 peer-DRAM demotion + accounting (l15_peer_dram_spec.md) --------
    def _book_dram_release(self, key, meta, lost: bool = False) -> None:
        """Free the DRAM reservation of one resident dram-tier piece and book its
        residency (hosted GB-seconds — the memory bill) + loss counters. Callers
        must invoke this EXACTLY once per resident dram piece, at whichever end
        comes first: demotion, supersede, host loss, or timeline discard."""
        if meta.get("tier") != "dram":
            return
        gb = float(meta.get("size_gb", 0.0))
        shards = int(meta.get("shards", 1))
        self.registry.release_dram(key[2], gb, shards)
        now = self.backend.now
        self.stats["dram_gb_seconds"] += gb * max(
            0.0, now - float(meta.get("hosted_at", now)))
        if lost:
            self.stats["dram_pieces_lost"] += shards

    def drop_hosted_piece(self, key, lost: bool = True,
                          disk_survives: bool = False) -> None:
        """Delete ONE hosted piece (any tier), freeing its DRAM reservation if it
        was dram-tier. The driver's host-loss paths route through this so the
        DRAM ledger stays exact; for ssd-tier pieces it is exactly `del`.

        disk_survives (host REBOOT): the DRAM buffer dies but the host DISK is
        intact — if the dropped piece was dram-tier and the donor still held
        the previous demoted piece's SSD file (dram_shadow), that file is
        restored as the live L2 piece (write-then-rename: the old file is only
        replaced when a demotion COMPLETES). node/spot/eviction pass False:
        the disk is gone, shadow included."""
        shadows = getattr(self, "dram_shadow", None)   # bare test shells lack it
        meta = self.peer_copies.pop(key, None)
        if meta is None:
            if shadows is not None and not disk_survives:
                _sh = shadows.pop(key, None)
                if _sh is not None:                 # resource bill: file gone
                    self._bill_hosted_ssd_close(key, _sh)
            return
        self._book_dram_release(key, meta, lost=lost)
        if meta.get("tier") != "dram":              # resource bill: ssd piece
            self._bill_hosted_ssd_close(key, meta)
        shadow = shadows.pop(key, None) if shadows is not None else None
        if disk_survives and shadow is not None and meta.get("tier") == "dram":
            self.peer_copies[key] = shadow      # old SSD file was never touched
            self.stats["dram_shadow_restored"] += 1
        elif shadow is not None:                    # resource bill: file gone
            self._bill_hosted_ssd_close(key, shadow)

    def finalize_dram_accounting(self) -> None:
        """End of run: book residency for still-resident dram pieces so the
        hosted-DRAM GB-h bill covers the whole horizon (no release — over)."""
        now = self.backend.now
        for meta in self.peer_copies.values():
            if isinstance(meta, dict) and meta.get("tier") == "dram":
                gb = float(meta.get("size_gb", 0.0))
                self.stats["dram_gb_seconds"] += gb * max(
                    0.0, now - float(meta.get("hosted_at", now)))
                meta["hosted_at"] = now          # idempotent if called again

    def _schedule_dram_demotions(self, worker, job, iteration, demote_info):
        """Kick one background demotion per landed dram piece, then chain the
        ordinary L3 donor drain AFTER every demotion settles (the drain reads
        the donor's SSD, which only holds the piece once demoted).
        demote_info: [(donor_worker, donor_job_id, key, meta, take)]."""
        if not demote_info:
            return
        procs = []
        for d, dj, key, meta, take in demote_info:
            p = self.backend.process(self._demote_piece(
                worker, job, iteration=iteration, donor=d, donor_job=dj,
                key=key, meta=meta, take=take))
            procs.append(p)
            self.background_flushes.append(p)
        if self.backstop != "donor_drain":
            return

        def _chain():
            yield self.backend.all_of(procs)
            drain_info = []
            for d, dj, key, meta, take in demote_info:
                cur = self.peer_copies.get(key)
                if (cur is meta and meta.get("tier") == "ssd"
                        and meta["iteration"] == iteration):
                    drain_info.append((d, dj, meta["piece_idx"],
                                       meta["size_gb"]))
            self._schedule_wave_drain(worker, job, iteration, drain_info)
        self.background_flushes.append(self.backend.process(_chain()))

    def _demote_piece(self, worker, job, *, iteration, donor, donor_job, key,
                      meta, take=1):
        """Background DRAM -> host-SSD demotion of one hosted piece: RES_DISK
        only (no NIC — the copy is host-local), donor-drain QoS weight so it
        yields the host's disk to foreground flushes. On completion the piece
        BECOMES an ordinary L2 piece (tier -> ssd; survives host reboot) and its
        DRAM reservation is freed. A host event mid-demotion aborts the
        transfer: reboot/node/spot delete the piece (loop exit); a host PROCESS
        failure leaves the daemon's buffer intact, so the demotion resumes after
        the restart."""
        caps = self._node_caps(donor_job)
        piece_gb = float(meta.get("size_gb", 0.0))
        while True:
            if self.peer_copies.get(key) is not meta or meta.get("tier") != "dram":
                self.stats["dram_demote_aborted"] += take
                return False                 # piece lost or superseded
            if getattr(donor, "failed", False):
                # host down: daemon resumes after the restart window
                yield self.backend.timeout(1.0)
                continue
            gen = donor.failure_generation
            ok = yield from self._capacity_transfer(
                worker, (donor,), (gen,), size_gb=piece_gb,
                streams=[[(donor.name, RES_DISK)]], iteration=iteration,
                operation="checkpoint_peer_dram_demote_to_ssd_chunk",
                source=f"{donor.name}/dram", destination=f"{donor.name}/ssd",
                details={"checkpoint_strategy": self.strategy_name,
                         "checkpoint_group": f"{job.job_id}-iteration-{iteration}",
                         "checkpoint_owner_rank": worker.rank,
                         "path": "dram_demote", "demote": True,
                         "donor": donor.name, "donor_job": donor_job,
                         "piece_idx": meta.get("piece_idx"),
                         "piece_shards": take},
                nominal_gbps=max(caps.disk_w * take, 1e-9),
                demand_per_stream=caps.disk_w * take,
                stream_weights=[max(take, 1) * DRAIN_QOS_WEIGHT],
                hold_cpu=False, network=False)
            if ok:
                break
            # aborted mid-transfer: loop re-checks whether the piece survived
        if self.peer_copies.get(key) is not meta or meta.get("tier") != "dram":
            self.stats["dram_demote_aborted"] += take
            return False
        # demoted piece is byte-identical to its dram parent (same size_gb /
        # iteration / piece_idx — validator I15 asserts this on the trace);
        # it now survives host reboot and frees the DRAM slot. The demotion's
        # completed write REPLACES the previous piece's SSD file (rename) —
        # its shadow, if any, is gone now.
        self._book_dram_release(key, meta, lost=False)
        meta["tier"] = "ssd"
        meta.pop("hosted_at", None)
        _sh = self.dram_shadow.pop(key, None)
        # resource bill: the rename replaces the shadow's SSD file NOW (close
        # its clock, which has been running since ITS demotion) and starts the
        # demoted piece's own SSD residency clock.
        if _sh is not None:
            self._bill_hosted_ssd_close(key, _sh)
        self._bill_hosted_ssd_open(key)
        self.stats["dram_demotions"] += take
        return True

    def discard_checkpoints_after(self, job_id: str, iteration: int) -> int:
        """Remove checkpoint state from an abandoned post-rollback timeline."""
        removed = 0
        for key, copy in list(self.copies.items()):
            if copy.iteration > iteration:
                del self.copies[key]
                removed += 1
        shadows = getattr(self, "dram_shadow", None)   # bare test shells lack it
        for key, meta in list(self.peer_copies.items()):
            if key[0] == job_id and int(meta.get("iteration", -1)) > iteration:
                self._book_dram_release(key, meta, lost=False)
                if meta.get("tier") != "dram":      # resource bill: ssd piece
                    self._bill_hosted_ssd_close(key, meta)
                del self.peer_copies[key]
                # a discarded (future-timeline) dram piece never replaced the
                # previous demoted piece's SSD file — restore its shadow when
                # that shadow is still on the surviving timeline
                if shadows is not None:
                    sh = shadows.pop(key, None)
                    if sh is not None and int(sh.get("iteration", -1)) <= iteration:
                        self.peer_copies[key] = sh  # bill clock keeps running
                    elif sh is not None:            # resource bill: file gone
                        self._bill_hosted_ssd_close(key, sh)
                removed += 1
        if shadows is not None:
            for key, sh in list(shadows.items()):
                if key[0] == job_id and int(sh.get("iteration", -1)) > iteration:
                    self._bill_hosted_ssd_close(key, sh)   # resource bill
                    del shadows[key]
        for key, meta in list(self.store_copies.items()):
            if key[0] == job_id and int(meta.get("iteration", -1)) > iteration:
                del self.store_copies[key]
                removed += 1
        for key, meta in list(self.l3_pieces.items()):
            if key[0] == job_id and int(meta.get("iteration", -1)) > iteration:
                del self.l3_pieces[key]
                removed += 1
        for rank, last in list(self._last_flush_started.items()):
            if last > iteration:
                self._last_flush_started[rank] = iteration
        return removed

    def drop_hosted_copies(self, donor_name: str) -> int:
        """Audit fix #13: a lost donor node destroys the shards it HOSTS for
        other jobs (they must not resurrect when the donor restarts). The
        driver calls this across ALL strategies on donor node/spot loss.
        L1.5: dram-tier pieces free their DRAM reservation and count as lost."""
        doomed = [key for key in self.peer_copies if key[2] == donor_name]
        for key in doomed:
            self.drop_hosted_piece(key, lost=True)      # disk gone: no shadow
        shadows = getattr(self, "dram_shadow", None)    # bare test shells lack it
        if shadows is not None:
            for key in [k for k in shadows if k[2] == donor_name]:
                self._bill_hosted_ssd_close(key, shadows[key])  # resource bill
                del shadows[key]                        # belt: orphan shadows
        return len(doomed)

    def drop_local_copies(self, rank: int, fraction: float = 1.0,
                          rng=None, tier: str | None = None) -> None:
        """Node-loss semantics for driver-side cohort runtimes: DRAM/SSD copies
        located on this worker's lost node(s) are gone. fraction<1 = ONE node
        of an m-node cohort died, so each located copy dies with p=1/m
        (fairness fix: wholesale cohort wipes over-punished in-job replication
        baselines like Gemini at mega scale).

        tier (2026-07-21, reboot_failure_type_spec.md): if set, only copies in
        that tier drop — a `reboot` passes tier='dram' so DRAM snapshots on the
        rebooted host are lost while the LOCAL SSD copy (key[2]=='ssd') survives
        the host restart. node/spot pass tier=None (every located tier gone).
        The key is (owner_rank, location_rank, tier); key[1] is the node the copy
        PHYSICALLY sits on, key[2] its tier."""
        for key in [key for key in self.copies
                    if key[1] == rank and (tier is None or key[2] == tier)]:
            if fraction >= 1.0 or rng is None or rng.random() < fraction:
                del self.copies[key]

    # ---- recovery: peer copies survive node loss (that is the point) --------------
    def _best_peer_set(self, job_id: str, rank: int):
        """Newest checkpoint iteration for (job, rank) whose k shards ALL sit on
        currently-alive donors (k-sharding needs every shard)."""
        by_iter: dict[int, list] = {}
        owner = getattr(self, "workers", {}).get(rank)
        max_iteration = getattr(owner, "current_iteration", None)
        for (j, r, donor), meta in self.peer_copies.items():
            if j == job_id and r == rank:
                if max_iteration is not None and meta["iteration"] > max_iteration:
                    continue
                by_iter.setdefault(meta["iteration"], []).append((donor, meta))
        best = None
        for it, entries in by_iter.items():
            k = entries[0][1].get("k", len(entries))
            alive = [d for d, _m in entries
                     if not getattr(self.registry.slots[d].worker, "failed", False)]
            if len(alive) >= k and (best is None or it > best[0]):
                best = (it, entries)
        return best

    def _best_stitch_set(self, job_id: str, rank: int):
        """Newest checkpoint iteration X for (job, rank) whose EVERY piece is
        recoverable from a currently-alive peer donor OR from an L3 piece landed
        at X — using at least one L3 piece (the all-peer case is _best_peer_set,
        left unchanged). This is the new piece-level recovery (goal point 2): a
        donor lost between waves leaves k-1 live peer pieces at X, and the missing
        piece is fetched individually from L3 at the SAME X. Returns
        (X, k, {piece_idx: ("peer", donor_name) | ("l3", None)}) or None."""
        peer_by_iter: dict[int, dict] = defaultdict(dict)   # it -> {piece: donor}
        l3_by_iter: dict[int, dict] = defaultdict(dict)     # it -> {piece: True}
        k_by_iter: dict[int, int] = {}
        owner = getattr(self, "workers", {}).get(rank)
        max_iteration = getattr(owner, "current_iteration", None)
        for (j, r, donor), meta in self.peer_copies.items():
            if j != job_id or r != rank:
                continue
            if max_iteration is not None and meta["iteration"] > max_iteration:
                continue
            pi = meta.get("piece_idx")
            if pi is None:
                continue
            it = meta["iteration"]
            k_by_iter[it] = max(k_by_iter.get(it, 0), meta.get("k", 0))
            slot = self.registry.slots.get(donor)
            if slot is not None and not getattr(slot.worker, "failed", False):
                peer_by_iter[it][pi] = donor
        for (j, r, pi), meta in self.l3_pieces.items():
            if j != job_id or r != rank:
                continue
            if max_iteration is not None and meta["iteration"] > max_iteration:
                continue
            it = meta["iteration"]
            k_by_iter[it] = max(k_by_iter.get(it, 0), meta.get("k", 0))
            l3_by_iter[it][pi] = True
        for it in sorted(set(peer_by_iter) | set(l3_by_iter), reverse=True):
            k = k_by_iter.get(it, 0)
            if k <= 0:
                continue
            sources, used_l3, ok = {}, False, True
            for pi in range(k):
                if pi in peer_by_iter[it]:
                    sources[pi] = ("peer", peer_by_iter[it][pi])
                elif pi in l3_by_iter[it]:
                    sources[pi] = ("l3", None)
                    used_l3 = True
                else:
                    ok = False
                    break
            if ok and used_l3:                # newest fully-coverable, needs L3
                return (it, k, sources)
        return None

    def _stitched_recover(self, worker, job, *, generation, stitch, fetch_gb,
                          shard_gb):
        """Reassemble a shard from a mix of live peer pieces and L3 pieces at ONE
        iteration (the stitch). Peer pieces stream donor NIC-out -> own NIC-in;
        L3 pieces stream store NIC-out -> own NIC-in. Both source tiers and the
        single consistent iteration are recorded in the recovery event details."""
        it, k, sources = stitch
        streams, demands, weights, srcnames, tiers = [], [], [], [], []
        donor_workers = []
        for pi in range(k):
            kind, donor = sources[pi]
            if kind == "peer":
                dslot = self.registry.slots[donor]
                donor_workers.append(dslot.worker)
                streams.append([(donor, RES_OUT), (worker.name, RES_IN)])
                demands.append(min(self._rate(dslot.job_id, "nic"),
                                   self._rate(job.job_id, "nic")))
                # L1.5: a not-yet-demoted dram-tier piece may participate in a
                # stitch; the causality label carries its tier (validator I16).
                pmeta = self.peer_copies.get((job.job_id, worker.rank, donor))
                p_tier = (pmeta or {}).get("tier", "ssd")
                srcnames.append(f"{donor}/{'dram' if p_tier == 'dram' else 'ssd'}")
                tiers.append("crossjob_peer_dram" if p_tier == "dram"
                             else "crossjob_peer")
            else:
                streams.append([(STORE_NODE, RES_OUT), (worker.name, RES_IN)])
                demands.append(min(self._rate(job.job_id, "nic"),
                                   self.cluster.object_store_bandwidth_gbps))
                srcnames.append(f"{STORE_NODE}/l3")
                tiers.append("l3")
            weights.append(1.0)
        watched = (worker, *donor_workers)
        gens = (generation, *[d.failure_generation for d in donor_workers])
        loaded = yield from self._capacity_transfer(
            worker, watched, gens, size_gb=fetch_gb, streams=streams,
            iteration=it,
            operation="checkpoint_crossjob_stitch_to_dram_recovery_chunk",
            source="+".join(srcnames), destination=f"{worker.name}/dram",
            details={"checkpoint_strategy": self.strategy_name,
                     "checkpoint_source_tier": "crossjob_peer_l3_stitch",
                     "checkpoint_source_rank": None,
                     "stitch_iteration": it,
                     "stitch_sources": tiers,
                     "peer_pieces": (tiers.count("crossjob_peer")
                                     + tiers.count("crossjob_peer_dram")),
                     "dram_pieces": tiers.count("crossjob_peer_dram"),
                     "l3_pieces": tiers.count("l3")},
            nominal_gbps=max(sum(demands), 1e-9),
            demand_per_stream=demands, stream_weights=weights, category="Recovery")
        if not loaded or worker.failure_generation != generation:
            return False
        ok = yield from self._finish_restore(worker, generation, shard_gb, it,
                                             "crossjob_peer_l3_stitch")
        return ok

    def _best_parity_set(self, job_id: str, rank: int):
        """STRETCH (backstop == parity): newest PARITY wave for (job, rank) with
        EXACTLY one piece lost — data_k of (data_k+1) donors alive — so the missing
        piece is reconstructed PEER-SIDE by XOR of the survivors, with NO L3 fetch.
        All-alive parity waves take the normal peer path; 2+ losses are not
        reconstructable here. Returns (it, data_k, [(donor, meta)]) or None."""
        by_iter: dict[int, dict] = defaultdict(
            lambda: {"alive": [], "n": 0, "data_k": 0, "parity": False})
        owner = getattr(self, "workers", {}).get(rank)
        max_iteration = getattr(owner, "current_iteration", None)
        for (j, r, donor), meta in self.peer_copies.items():
            if j != job_id or r != rank:
                continue
            if max_iteration is not None and meta["iteration"] > max_iteration:
                continue
            it = meta["iteration"]
            rec = by_iter[it]
            nn = meta.get("k", 0)
            dk = meta.get("data_k", nn)
            rec["n"] = max(rec["n"], nn)
            rec["data_k"] = dk
            if dk < nn:
                rec["parity"] = True
            slot = self.registry.slots.get(donor)
            if slot is not None and not getattr(slot.worker, "failed", False):
                rec["alive"].append((donor, meta))
        for it in sorted(by_iter, reverse=True):
            rec = by_iter[it]
            if rec["parity"] and len(rec["alive"]) == rec["data_k"]:
                return (it, rec["data_k"], rec["alive"])
        return None

    def _parity_recover(self, worker, job, *, generation, parity, fetch_gb,
                        shard_gb):
        """Reconstruct a shard at ONE iteration from data_k surviving pieces (the
        XOR of any data_k of the data_k+1 pieces recovers the missing one). Fetch
        the survivors donor NIC-out -> own NIC-in; the XOR itself is CPU-cheap and
        not separately modeled (it overlaps the fetch)."""
        it, data_k, alive = parity
        use = alive[:data_k]                      # any data_k survivors suffice
        donors = [d for d, _m in use]
        streams = [[(d, RES_OUT), (worker.name, RES_IN)] for d in donors]
        demands = [min(self._rate(self.registry.slots[d].job_id, "nic"),
                       self._rate(job.job_id, "nic")) for d in donors]
        donor_workers = [self.registry.slots[d].worker for d in donors]
        watched = (worker, *donor_workers)
        gens = (generation, *[d.failure_generation for d in donor_workers])
        loaded = yield from self._capacity_transfer(
            worker, watched, gens, size_gb=fetch_gb, streams=streams,
            iteration=it,
            operation="checkpoint_crossjob_parity_reconstruct_recovery_chunk",
            source="+".join(f"{d}/ssd" for d in donors),
            destination=f"{worker.name}/dram",
            details={"checkpoint_strategy": self.strategy_name,
                     "checkpoint_source_tier": "crossjob_peer_parity",
                     "checkpoint_source_rank": None,
                     "parity_iteration": it, "data_k": data_k,
                     "reconstructed_pieces": 1},
            nominal_gbps=max(sum(demands), 1e-9),
            demand_per_stream=demands, category="Recovery")
        if not loaded or worker.failure_generation != generation:
            return False
        ok = yield from self._finish_restore(worker, generation, shard_gb, it,
                                             "crossjob_peer_parity")
        return ok

    def recover(self, worker: CheckpointWorker, job, size_scale: float = 1.0):
        """size_scale > 1 (cohort runtimes): the NETWORK fetch carries scale x the
        per-rank shard (whole-cohort refetch after eviction); the dram->GPU restore
        tail stays per-rank (it runs in parallel on every node of the cohort)."""
        generation = worker.failure_generation
        shard_gb = job.checkpoint_gb / job.pipeline_stages
        fetch_gb = shard_gb * size_scale
        local = self._latest_available_copy(worker)
        peer = self._best_peer_set(job.job_id, worker.rank)
        store = self.store_copies.get((job.job_id, worker.rank))
        if (
            store is not None
            and store["iteration"] > getattr(worker, "current_iteration", store["iteration"])
        ):
            store = None
        local_it = local.iteration if local is not None else -1
        peer_it = peer[0] if peer is not None else -1
        store_it = store["iteration"] if store is not None else -1

        # piece-level L3 stitch (goal point 2): only when it recovers a STRICTLY
        # newer iteration than any complete source — i.e. a donor died between
        # waves and L3 fills the gap. Equal newest peer set stays on the (cheaper,
        # store-free) peer path below; hand-check math there is unchanged.
        stitch = self._best_stitch_set(job.job_id, worker.rank)
        stitch_it = stitch[0] if stitch is not None else -1
        # STRETCH: XOR parity reconstruction (peer-side, no L3) for a single donor
        # loss on a parity wave. Preferred whenever it recovers a strictly newer
        # iteration than any complete/stitched source.
        parity = self._best_parity_set(job.job_id, worker.rank)
        parity_it = parity[0] if parity is not None else -1
        if parity_it > max(local_it, peer_it, store_it, stitch_it):
            ok = yield from self._parity_recover(
                worker, job, generation=generation, parity=parity,
                fetch_gb=fetch_gb, shard_gb=shard_gb)
            return ok
        if stitch_it > max(local_it, peer_it, store_it):
            ok = yield from self._stitched_recover(
                worker, job, generation=generation, stitch=stitch,
                fetch_gb=fetch_gb, shard_gb=shard_gb)
            return ok

        if store_it > max(local_it, peer_it):
            ok = yield from self._restore_from_store(
                worker, job, generation=generation, store_it=store_it,
                fetch_gb=fetch_gb, shard_gb=shard_gb)
            return ok

        if peer is None or local_it >= peer_it:
            ok = yield from super().recover(worker, job)
            return ok

        it, entries = peer
        k = entries[0][1].get("k", len(entries))
        donors = [d for d, _m in entries]
        # L1.5 tier labels: a piece still resident in the donor daemon's DRAM
        # (not yet demoted) recovers as `crossjob_peer_dram`; demoted pieces are
        # ordinary L2 (`crossjob_peer`); a mixed-tier wave (some pieces demoted,
        # some not) is the dram stitch variant. Recovery physics are identical
        # (donor NIC-out -> own NIC-in; source medium read not modeled) — only
        # the causality label differs, so the validator can check volatility.
        piece_tiers = [m.get("tier", "ssd") for _d, m in entries]
        injob_wave = any(m.get("injob_ssd") for _d, m in entries)
        if injob_wave:
            # in-job SSD replica (rack_failure_spec.md): same fetch physics
            # (holder NIC-out -> own NIC-in), its own causality label so the
            # 2x2 placement outcome is auditable per recovery.
            src_tier = "injob_ssd_replica"
        elif all(t == "dram" for t in piece_tiers):
            src_tier = "crossjob_peer_dram"
        elif "dram" in piece_tiers:
            src_tier = "crossjob_peer_dram_stitch"
        else:
            src_tier = "crossjob_peer"
        srcnames = [f"{d}/{'dram' if t == 'dram' else 'ssd'}"
                    for d, t in zip(donors, piece_tiers)]
        recovery_op = ("checkpoint_injob_ssd_to_dram_recovery_chunk" if injob_wave
                       else "checkpoint_crossjob_peers_to_dram_recovery_chunk")
        slowest = min(self._rate(m["donor_job"], "disk") for _d, m in entries)
        bw = min(self._rate(job.job_id, "nic"), k * slowest)
        if self.capacity_mode:
            # one flow per donor: donor NIC-out -> own NIC-in (disk read not modeled)
            streams = [[(d, RES_OUT), (worker.name, RES_IN)] for d in donors]
            loaded = yield from self._capacity_transfer(
                worker, (worker,), (generation,), size_gb=fetch_gb, streams=streams,
                iteration=it,
                operation=recovery_op,
                source="+".join(srcnames),
                destination=f"{worker.name}/dram",
                details={"checkpoint_strategy": self.strategy_name,
                         "checkpoint_source_tier": src_tier,
                         "checkpoint_source_rank": None,
                         **({"dram_donors": [d for d, t in zip(donors, piece_tiers)
                                             if t == "dram"]}
                            if "dram" in piece_tiers else {})},
                nominal_gbps=bw, category="Recovery")
        else:
            loaded = yield from self._recovery_chunks(
                worker, generation=generation, iteration=it, size_gb=fetch_gb,
                chunk_gb=job.checkpoint_chunk_gb, bandwidth_gbps=bw,
                operation=recovery_op,
                source="+".join(srcnames),
                destination=f"{worker.name}/dram",
                resources=["CPU", "NETWORK"],
                source_tier=src_tier, source_rank=None,
            )
        if not loaded or worker.failure_generation != generation:
            return False
        ok = yield from self._finish_restore(worker, generation, shard_gb,
                                             it, src_tier)
        return ok

    def summary(self) -> dict:
        return {
            **{f"stat_{k}": v for k, v in self.stats.items()},
            **{f"refused_{k}": v for k, v in self.registry.refusals.items()},
            "peer_copies": len(self.peer_copies),
            "l3_pieces": len(self.l3_pieces),
            "backstop": self.backstop,
            "dram_jobs": sorted(j for j, on in self.dram_by_job.items() if on),
        }
