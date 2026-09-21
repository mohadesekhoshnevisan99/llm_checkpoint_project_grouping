"""Per-node capacity model: duplex NIC + disk-write resources with max-min fairness.

Fixes the two systemic contention inaccuracies found by hardware cross-validation
(CROSSJOB_GAPS #8/#10 and their common root):
  - the base model shares bandwidth by TASK COUNT at the busiest endpoint (half-duplex,
    capacity-blind) — so flows slow each other even when the pipe has room, and never
    saturate a resource that is actually full;
  - disk write bandwidth is a rate parameter, not a resource — concurrent writers to
    one disk (hosted shards, local fallbacks, store ingest) never divide it.

Here every node has three independent capacities (GB/s): nic_in, nic_out, disk_w.
A transfer registers one Flow per stream, listing the (node, resource) pairs it
traverses, e.g. a k-donor peer flush = k flows, each [sender:out, donor:in, donor:disk].
Allocations are max-min fair (progressive filling): all active flows rise together;
a flow freezes when it meets its demand cap or when any resource it uses saturates.
Recomputed lazily whenever the flow set changes; O(flows x resources) per epoch.

Strategies consume this via a slowdown callable for simulation/progress.advance_work:
factor = base_rate / current_allocation, re-evaluated every contention quantum. The
base model's checkpoint_network_tasks counters are still maintained by callers so
training <-> checkpoint coupling keeps working; capacity replaces only the transfer-
rate arithmetic (no double counting: capacity-driven transfers must not ALSO apply
the count-based slowdown).

New module only; nothing in the existing engine changes. Author: Sam's agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from typing import Any, ClassVar, Iterable

RES_IN, RES_OUT, RES_DISK = "in", "out", "disk"

# Event-engine tolerances. BYTE_EPS: a flow within this many GB of its demand is
# treated as finished (mirrors the polling loop's 1e-9 remaining threshold).
# TIME_EPS: any flow that would finish within this many sim-seconds is reaped in
# the SAME reschedule instead of scheduling a vanishing timeout (prevents a
# same-instant wake spin when now + tiny underflows to now in float64; the lost
# bytes are <= allocation * TIME_EPS, far below the polling engine's own
# max(dt, 1e-6) step floor).
BYTE_EPS = 1e-9
TIME_EPS = 1e-9


@dataclass
class NodeCaps:
    nic_in: float
    nic_out: float
    disk_w: float

    def cap(self, res: str) -> float:
        return {RES_IN: self.nic_in, RES_OUT: self.nic_out, RES_DISK: self.disk_w}[res]


@dataclass(eq=False)
class Flow:
    """One stream bundle with a demand cap (GB/s) traversing (node, resource)
    pairs. `weight` = number of PHYSICAL unit-streams this flow stands for
    (cohort mode bundles m ranks' streams into one flow); fair sharing is
    per unit-stream, so a saturated resource splits proportionally to weight.
    weight=1 reproduces the original per-flow max-min exactly (audit fix #1/#5)."""

    uses: tuple[tuple[str, str], ...]
    demand: float
    group: object = None          # opaque tag: flows of one logical transfer
    weight: float = 1.0
    allocation: float = 0.0
    remaining: float = 0.0        # bytes (GB) left to send — event engine only
    standing: bool = False        # persistent background demand (Route C): never
                                  # depletes, carries no group/done, and is never
                                  # scheduled on the completion heap — it only
                                  # perturbs the max-min solve (see _install_background)
    _ev_last: float = 0.0         # sim-time this flow was last credited (event)
    _ev_tok: int = 0              # scheduling generation (stale-heap-entry token)
    _birth: int = 0               # DETERMINISM: monotonic creation order (stamped by
                                  # the registry). Flow is eq=False -> identity-hashed,
                                  # so `set(flows)` iterates in id()/address order, which
                                  # is NOT stable run-to-run even under PYTHONHASHSEED=0.
                                  # `_birth` is the run-independent TOTAL ORDER used to
                                  # break equal-finish-time ties (see _ev_touch / the
                                  # completion heap) so simultaneous events — the norm
                                  # under clock-anchored slot flushing — are processed in
                                  # the same order every run. Creation order is itself
                                  # deterministic (fixed up to the very tie this breaks),
                                  # so the order is consistent across runs. Physical FIFO
                                  # tie-break; carries no policy.


@dataclass(eq=False)
class _GroupState:
    """Event-engine bookkeeping for one in-flight transfer (a Flow `group`).
    `done` fires when every stream has finished; `abort` fires when a watched
    worker's generation changes (failure/eviction) — exactly the two ways the
    polling loop could leave its `while any(remaining)` block.

    THRESHOLD-COMPLETION (straggler cutoff, XOR parity): when `threshold` is set
    (< len(flows)), `partial` fires as soon as `threshold` streams have landed —
    the wave is then recoverable (any k of k+1 XOR-reconstruct) and the caller
    returns, detaching the still-in-flight straggler stream(s) to a background
    process that keeps watching `done`/`abort`. `done` still fires only when ALL
    streams (including the straggler) finish, so the background wait is unchanged;
    `threshold=None` => `partial` is never armed and behaviour is identical."""

    done: Any
    abort: Any
    flows: set
    watched: tuple
    aborted: bool = False
    finished: bool = False
    threshold: Any = None            # int: partial fires at this many landed
    partial: Any = None              # event fired at the threshold (cutoff)
    landed: int = 0                  # streams finished so far (for the threshold)


@dataclass
class CapacityRegistry:
    """Run-wide node capacities + active flows with max-min fair allocation."""

    nodes: dict[str, NodeCaps] = field(default_factory=dict)
    flows: set = field(default_factory=set)
    _dirty: bool = True

    # Route C — per-node persistent BACKGROUND front-end traffic (GB/s). Models the
    # front-end/storage-fabric load that is NOT our checkpoints: a GPU node's own
    # training-data reads, plus other tenants' storage I/O on the shared storage
    # fabric (workload_sources_dossier.md "Dual-fabric evidence" §D + caveat 2;
    # Mark meeting D3). > 0 installs one STANDING ingress demand of this magnitude on
    # every node's NIC as it is registered; 0.0 (default) installs nothing, so every
    # existing scenario is byte-identical. See _install_background for the direction
    # rationale (data loading = ingress) and why it adds no simulator events.
    background_nic_gbps: float = 0.0
    _background_flows: dict = field(default_factory=dict)   # node -> standing Flow

    # -- event engine state (unused in polling mode) ---------------------------
    backend: Any = None
    _ev_groups: dict = field(default_factory=dict)          # group -> _GroupState
    _ev_worker_groups: dict = field(default_factory=dict)   # id(worker) -> {group}
    _ev_users: dict = field(default_factory=dict)           # (node,res) -> {flow}
    _ev_heap: list = field(default_factory=list)            # (finish_t, seq, flow, tok)
    _ev_seq: int = 0            # unique per heap push; assigned in _birth order
    _flow_seq: int = 0          # monotonic source for Flow._birth (deterministic order)
    _ev_wake_time: float = None

    _by_run: ClassVar[dict[int, "CapacityRegistry"]] = {}

    @classmethod
    def for_run(cls, run_id: int) -> "CapacityRegistry":
        if run_id not in cls._by_run:
            cls._by_run[run_id] = cls()
        return cls._by_run[run_id]

    @classmethod
    def reset(cls, run_id: int) -> None:
        cls._by_run.pop(run_id, None)

    def set_node(self, name: str, caps: NodeCaps) -> None:
        if name in self.nodes:
            return
        self.nodes[name] = caps
        self._install_background(name)

    def _install_background(self, name: str) -> None:
        """Route C: install the node's STANDING background ingress demand.

        Direction (physically sensible choice, documented per the FEATURE note):
        data loading is a READ *into* the node, so the background rides RES_IN
        (ingress). On a full-duplex NIC (in/out are independent capacities here)
        an ingress demand contends with checkpoint traffic the node *receives* —
        hosted peer shards and restore reads, plus, on the store node, incoming
        checkpoint writes and other tenants' storage I/O — but NOT with the node's
        own outgoing checkpoint writes (egress). That matches real deployments:
        the front-end/storage fabric absorbs data-loading reads and checkpoint
        writes together, and Route A already keeps both off the NCCL compute
        fabric (dossier §D, caveat 2).

        It is a STANDING demand, never a discrete event: the flow carries no group
        and no done-event, its `remaining` is infinite so it never depletes or
        completes, and (event engine) it is skipped when survivors are scheduled
        on the completion heap — so it changes max-min allocations WITHOUT adding
        any events. B <= 0 installs nothing (existing scenarios unchanged)."""
        b = self.background_nic_gbps
        if b <= 0.0:
            return
        f = Flow(uses=((name, RES_IN),), demand=b, group=None, weight=1.0,
                 standing=True)
        self._flow_seq += 1
        f._birth = self._flow_seq
        f.remaining = float("inf")
        self.flows.add(f)
        # index it for the event engine's component walk (harmless in polling mode,
        # which ignores _ev_users and just re-solves the whole live flow set).
        self._ev_users.setdefault((name, RES_IN), set()).add(f)
        self._background_flows[name] = f
        self._dirty = True

    def open(self, uses: Iterable[tuple[str, str]], demand: float,
             group: object = None, weight: float = 1.0) -> Flow:
        flow = Flow(uses=tuple(uses), demand=max(demand, 1e-12), group=group,
                    weight=max(weight, 1e-12))
        self._flow_seq += 1
        flow._birth = self._flow_seq
        self.flows.add(flow)
        self._dirty = True
        return flow

    def close(self, flow: Flow) -> None:
        self.flows.discard(flow)
        self._dirty = True

    def _reallocate(self) -> None:
        """WEIGHTED progressive-filling max-min fairness over ALL live flows —
        the polling engine's entry point, numerically UNCHANGED (it just calls
        the extracted kernel on the whole flow set)."""
        self._reallocate_subset(self.flows)
        self._dirty = False

    def _reallocate_subset(self, flows) -> None:
        """WEIGHTED progressive-filling max-min fairness for `flows`: the fill
        LEVEL is the per-unit-stream rate; flow allocation = level * weight. All
        weights 1 reduces to classic per-flow max-min (bit-identical to the old
        code). The result is independent of flow iteration order, so calling this
        on the whole set (polling) or per connected component (event) yields the
        SAME allocations — max-min is separable across disjoint resource sets.

        DETERMINISM (audit 2026-07-25, same class as the fixed _birth bug):
        `flows` arrives as a SET of identity-hashed Flow objects, so iterating
        it — and the per-resource user sets — runs in id()/address order, which
        is NOT stable run-to-run even under PYTHONHASHSEED=0. The weight SUMS
        below are floating-point, so summation order can flip ULPs and, at a
        near-tie, freeze a different flow first. All iteration that feeds
        arithmetic is therefore pinned to Flow._birth order (physical FIFO,
        carries no policy): `ordered` fixes dict insertion order, and users[u]
        are LISTS built in that order so every sum accumulates identically."""
        ordered = sorted(flows, key=lambda f: f._birth)
        active = {f: 0.0 for f in ordered}         # allocation (level * weight)
        if not active:
            return
        remaining: dict[tuple[str, str], float] = {}
        users: dict[tuple[str, str], list] = {}
        for f in ordered:
            for u in f.uses:
                node, res = u
                if node in self.nodes:
                    remaining.setdefault(u, self.nodes[node].cap(res))
                    users.setdefault(u, []).append(f)
        unfrozen = set(active)
        while unfrozen:
            # largest uniform LEVEL increment every unfrozen flow can take
            # (min over exact per-flow ratios: order-independent)
            inc = min((f.demand - active[f]) / f.weight for f in unfrozen)
            for u, cap_left in remaining.items():
                w = sum(f.weight for f in users[u] if f in unfrozen)
                if w > 0:
                    inc = min(inc, cap_left / w)
            if inc <= 1e-15:
                inc = 0.0
            for f in unfrozen:
                active[f] += inc * f.weight
            for u in list(remaining):
                w = sum(f.weight for f in users[u] if f in unfrozen)
                if w > 0:
                    remaining[u] -= inc * w
            # freeze: flows at demand, and all flows on saturated resources
            newly_frozen = {f for f in unfrozen
                            if f.demand - active[f] <= 1e-12 * f.weight}
            for u, cap_left in remaining.items():
                if cap_left <= 1e-12:
                    newly_frozen |= {f for f in users[u] if f in unfrozen}
            if not newly_frozen:      # numerical guard: freeze everything
                newly_frozen = set(unfrozen)
            unfrozen -= newly_frozen
        for f, a in active.items():
            f.allocation = a

    def flow_rate(self, flow: Flow) -> float:
        """Current allocation of ONE flow (GB/s) — per-stream advancement."""
        if self._dirty:
            self._reallocate()
        return flow.allocation

    def rate(self, group: object) -> float:
        """Current aggregate allocation (GB/s) of all flows tagged with `group`."""
        if self._dirty:
            self._reallocate()
        return sum(f.allocation for f in self.flows if f.group is group)

    # ==================================================================
    # EVENT ENGINE
    #
    # Same physics as the polling consumer (crossjob._capacity_transfer), but
    # shares recompute ONLY when a flow starts, finishes, or dies — and only for
    # the CONNECTED COMPONENT the change touches (flows sharing a capacity
    # resource). Independent transfers (e.g. gemini's in-job replication, one
    # component per job) never enter each other's work: crediting, the max-min
    # solve, and finish scheduling are all O(component), not O(fleet). That is
    # the fix the profiling note (control-plane/design/mega_striping_tail_note)
    # is after — the old poll loop paid O(fleet) per contention quantum.
    #
    # Per-flow invariant: a flow's `allocation` is constant between the moments
    # its component is (re)solved (the only place `_reallocate_subset` runs on
    # it), and `_ev_last` records that moment — so crediting `allocation *
    # (now - _ev_last)` against `remaining` is EXACT (no quantum error). A lazy
    # min-heap of (finish_time, flow, token) drives wakes; a bumped token
    # invalidates a flow's stale entries after any re-solve (the vendored simpy
    # shim cannot cancel a scheduled timeout).
    # ==================================================================

    def ev_attach(self, backend: Any) -> None:
        """Bind the sim backend so the scheduler can time completions."""
        self.backend = backend

    def make_flow(self, uses: Iterable[tuple[str, str]], demand: float,
                  group: object = None, weight: float = 1.0) -> Flow:
        """Construct a Flow WITHOUT registering it (event mode). The flow joins
        the live set only in `ev_arm`, atomically with `remaining` set and its
        group armed — so a concurrent solve can never see a half-built,
        zero-remaining flow and reap it prematurely."""
        flow = Flow(uses=tuple(uses), demand=max(demand, 1e-12), group=group,
                    weight=max(weight, 1e-12))
        self._flow_seq += 1
        flow._birth = self._flow_seq
        return flow

    def ev_arm(self, group: object, flows: list, watched: Iterable,
               threshold: int | None = None) -> tuple:
        """Register a transfer's streams (each Flow already carries its
        `remaining`), then re-solve the affected component once. Returns
        (done_event, abort_event, partial_event). `partial_event` is None unless
        `threshold` is given, in which case it fires once `threshold` streams have
        landed (the straggler-cutoff hook — see _GroupState)."""
        partial = self.backend.event() if threshold is not None else None
        st = _GroupState(done=self.backend.event(), abort=self.backend.event(),
                         flows=set(flows), watched=tuple(watched),
                         threshold=threshold, partial=partial)
        self._ev_groups[group] = st
        for w in st.watched:
            self._ev_worker_groups.setdefault(id(w), set()).add(group)
        now = float(self.backend.now)
        for f in flows:
            f._ev_last = now
            self._ev_add_flow(f)
        self._ev_touch(flows)
        self._ev_ensure_wake()
        return st.done, st.abort, st.partial

    def ev_notify_worker_change(self, worker: Any) -> None:
        """A watched worker's generation changed (failure/eviction): abort every
        in-flight transfer that watches it. Called from the failure path so the
        abort is prompt — the observable equivalent of the polling loop's
        per-quantum `valid()` re-check. Flows are freed by the transfer's own
        `finally` (ev_dismiss), matching the polling engine's finally block.

        DETERMINISM (audit 2026-07-25): `groups` is a set of opaque group
        objects (id()-order iteration). When one failure aborts SEVERAL
        transfers, the abort events fire in iteration order — pin it to the
        groups' oldest member flow (_birth, physical FIFO) so downstream
        wakeups run in a run-independent order."""
        groups = self._ev_worker_groups.get(id(worker))
        if not groups:
            return

        def _group_key(g):
            st = self._ev_groups.get(g)
            if st is None or not st.flows:
                return (1 << 62)
            return min(f._birth for f in st.flows)

        for group in sorted(groups, key=_group_key):
            st = self._ev_groups.get(group)
            if st is not None and not st.aborted and not st.finished:
                st.aborted = True
                if not st.abort.triggered:
                    st.abort.succeed()

    def ev_dismiss(self, group: object) -> None:
        """Tear down a transfer's group (from its finally). On normal completion
        the group is already gone (reaped in `_ev_touch`), so this is a no-op; on
        abort it removes the still-open flows and re-solves their neighbours so
        the freed capacity refills."""
        st = self._ev_groups.pop(group, None)
        if st is None:
            return
        for w in st.watched:
            s = self._ev_worker_groups.get(id(w))
            if s is not None:
                s.discard(group)
                if not s:
                    self._ev_worker_groups.pop(id(w), None)
        flows = list(st.flows)
        st.flows.clear()
        if not flows:
            return
        # surviving neighbours (still credited at the OLD allocation until now)
        neighbours = []
        drop = set(flows)
        for f in flows:
            for u in f.uses:
                if u[0] in self.nodes:
                    for g in self._ev_users.get(u, ()):
                        if g not in drop:
                            neighbours.append(g)
        for f in flows:
            self._ev_remove_flow(f)
        self._ev_touch(neighbours)
        self._ev_ensure_wake()

    # -- internals -------------------------------------------------------------

    def _ev_add_flow(self, f: Flow) -> None:
        self.flows.add(f)
        for u in f.uses:
            if u[0] in self.nodes:
                self._ev_users.setdefault(u, set()).add(f)

    def _ev_remove_flow(self, f: Flow) -> None:
        self.flows.discard(f)
        for u in f.uses:
            if u[0] in self.nodes:
                s = self._ev_users.get(u)
                if s is not None:
                    s.discard(f)
                    if not s:
                        self._ev_users.pop(u, None)
        st = self._ev_groups.get(f.group)
        if st is not None:
            st.flows.discard(f)

    def _ev_component(self, seed) -> set:
        """Flows reachable from `seed` through shared capacity resources. Marks
        each flow seen at PUSH time (not pop) so every flow and edge is visited
        once — O(component); a pop-time guard re-pushes duplicates and degrades
        to O(component^2) on large cross-job-coupled components (store/donor)."""
        comp: set = set()
        stack = []
        for f in seed:
            if f in self.flows and f not in comp:
                comp.add(f)
                stack.append(f)
        while stack:
            f = stack.pop()
            for u in f.uses:
                if u[0] in self.nodes:
                    for g in self._ev_users.get(u, ()):
                        if g not in comp:
                            comp.add(g)
                            stack.append(g)
        return comp

    def _ev_fire_completed(self, finished) -> None:
        # threshold (cutoff): credit each landed stream and fire `partial` the
        # moment `threshold` of a group's streams have finished — the wave is
        # recoverable and the caller detaches the remaining straggler.
        for f in finished:
            st = self._ev_groups.get(f.group)
            if st is None or st.aborted or st.threshold is None:
                continue
            st.landed += 1
            if (st.partial is not None and not st.partial.triggered
                    and st.landed >= st.threshold):
                st.partial.succeed()
        seen = set()
        for f in finished:
            g = f.group
            if g is None or g in seen:
                continue
            seen.add(g)
            st = self._ev_groups.get(g)
            if st is None or st.aborted or st.finished or st.flows:
                continue
            st.finished = True
            self._ev_groups.pop(g, None)
            for w in st.watched:
                s = self._ev_worker_groups.get(id(w))
                if s is not None:
                    s.discard(g)
                    if not s:
                        self._ev_worker_groups.pop(id(w), None)
            if not st.done.triggered:
                st.done.succeed()

    def _ev_touch(self, seed) -> None:
        """Re-solve the connected component(s) touched by `seed`: credit elapsed
        bytes, reap finished streams (firing their groups' done events),
        re-solve max-min for the survivors, and (re)schedule their finishes.
        Cost is O(component)."""
        comp = self._ev_component(seed)
        if not comp:
            return
        now = float(self.backend.now)
        for f in comp:
            dt = now - f._ev_last
            if dt > 0.0 and not f.standing:      # standing demand never depletes
                f.remaining -= f.allocation * dt
            f._ev_last = now
        while True:
            # DETERMINISM: sort by _birth. `comp` is a set of identity-hashed Flows,
            # so its iteration order is id()/address order (unstable run-to-run even
            # with PYTHONHASHSEED=0). `finished` order drives group-done firing in
            # _ev_fire_completed, so it must be run-independent.
            finished = sorted((f for f in comp if f.remaining <= BYTE_EPS),
                              key=lambda f: f._birth)
            if finished:
                for f in finished:
                    comp.discard(f)
                    self._ev_remove_flow(f)
                self._ev_fire_completed(finished)
            self._reallocate_subset(comp)
            instant = [f for f in comp
                       if f.allocation > 1e-15 and f.remaining > BYTE_EPS
                       and f.remaining / f.allocation <= TIME_EPS]
            if not finished and not instant:
                break
            for f in instant:                    # collapse sub-TIME_EPS finishes
                f.remaining = 0.0
        # (re)schedule survivors in _birth order so `_ev_seq` — the completion
        # heap's tie-break for equal finish times (the norm under clock-anchored
        # slot flushing) — is assigned in deterministic FIFO-by-creation order
        # rather than set-iteration (id) order. `_ev_seq` stays unique per push
        # (so the heap never has to compare two Flow objects), but WHICH flow gets
        # the smaller seq within a batch is now run-independent.
        for f in sorted(comp, key=lambda f: f._birth):
            if f.standing:                       # standing demand never completes:
                continue                         # keep it off the heap (no events)
            f._ev_tok += 1
            a = f.allocation
            if a > 1e-15 and f.remaining > BYTE_EPS:
                self._ev_seq += 1
                heappush(self._ev_heap,
                         (now + f.remaining / a, self._ev_seq, f, f._ev_tok))

    def _ev_stale(self, entry) -> bool:
        _ft, _seq, f, tok = entry
        return (f not in self.flows or f._ev_tok != tok
                or f.remaining <= BYTE_EPS or f.allocation <= 1e-15)

    def _ev_compact(self) -> None:
        """Drop buried stale heap entries. A flow re-solved K times leaves K-1
        stale entries that lazy top-cleaning never reaches if their finish time
        is in the future — over a long, churny run that leak degrades every heap
        op to O(leak) and the clock appears to freeze. Rebuild when the heap has
        grown well past the live-flow count (amortized O(1) per push)."""
        heap = self._ev_heap
        if len(heap) > 2 * len(self.flows) + 16:
            self._ev_heap = [e for e in heap if not self._ev_stale(e)]
            heapify(self._ev_heap)

    def _ev_ensure_wake(self) -> None:
        """Schedule a pump at the earliest pending finish, unless an equal-or-
        earlier wake is already scheduled. Discards stale heap tops first."""
        self._ev_compact()
        heap = self._ev_heap
        while heap:
            if self._ev_stale(heap[0]):
                heappop(heap)
                continue
            ft = heap[0][0]
            if self._ev_wake_time is None or ft < self._ev_wake_time - 1e-12:
                self._ev_wake_time = ft
                now = float(self.backend.now)
                ev = self.backend.timeout(max(ft - now, 0.0))
                ev.add_callback(self._ev_pump)
            return

    def _ev_pump(self, _event: Any = None) -> None:
        """Wake handler: process every stream whose finish has arrived (each
        re-solves its component), then re-arm the next wake."""
        self._ev_wake_time = None
        heap = self._ev_heap
        now = float(self.backend.now)
        while heap:
            if self._ev_stale(heap[0]):
                heappop(heap)
                continue
            if heap[0][0] > now + TIME_EPS:
                break                            # earliest real finish is future
            _ft, _seq, f, _tok = heappop(heap)
            self._ev_touch([f])
            now = float(self.backend.now)
        self._ev_ensure_wake()
