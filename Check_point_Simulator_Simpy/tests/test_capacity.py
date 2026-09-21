"""Unit tests for the per-node capacity model (checkpointing/capacity.py)."""
import pytest

from checkpointing.capacity import CapacityRegistry, NodeCaps, RES_DISK, RES_IN, RES_OUT


def reg(**nodes):
    r = CapacityRegistry()
    for name, (i, o, d) in nodes.items():
        r.set_node(name, NodeCaps(nic_in=i, nic_out=o, disk_w=d))
    return r


def test_single_flow_bottlenecked_by_disk():
    r = reg(a=(1.25, 1.25, 9), b=(1.25, 1.25, 0.17))
    f = r.open([("a", RES_OUT), ("b", RES_IN), ("b", RES_DISK)], demand=99, group="g")
    assert r.rate("g") == pytest.approx(0.17)
    r.close(f)


def test_duplex_full_rate_send_and_receive():
    # node b sends one stream and receives another: full duplex = both at line rate
    r = reg(a=(1.0, 1.0, 9), b=(1.0, 1.0, 9), c=(1.0, 1.0, 9))
    r.open([("a", RES_OUT), ("b", RES_IN)], demand=99, group="in")
    r.open([("b", RES_OUT), ("c", RES_IN)], demand=99, group="out")
    assert r.rate("in") == pytest.approx(1.0)
    assert r.rate("out") == pytest.approx(1.0)


def test_maxmin_unequal_demands():
    # two flows share a 1.0 pipe; one only wants 0.2 -> other gets 0.8
    r = reg(a=(9, 1.0, 9), b=(9, 9, 9), c=(9, 9, 9))
    r.open([("a", RES_OUT), ("b", RES_IN)], demand=0.2, group="small")
    r.open([("a", RES_OUT), ("c", RES_IN)], demand=99, group="big")
    assert r.rate("small") == pytest.approx(0.2)
    assert r.rate("big") == pytest.approx(0.8)


def test_k_subflows_share_sender_ceiling():
    # 4 donor streams, each disk-capped 0.173, sender out capped 0.52
    r = reg(s=(9, 0.52, 9), **{f"d{i}": (1.25, 9, 0.173) for i in range(4)})
    for i in range(4):
        r.open([("s", RES_OUT), (f"d{i}", RES_IN), (f"d{i}", RES_DISK)],
               demand=99, group="xfer")
    assert r.rate("xfer") == pytest.approx(0.52)


def test_hosted_write_shares_disk_with_local_write():
    r = reg(s=(9, 9, 9), d=(9, 9, 0.2))
    r.open([("s", RES_OUT), ("d", RES_IN), ("d", RES_DISK)], demand=99, group="hosted")
    r.open([("d", RES_DISK)], demand=99, group="local")
    assert r.rate("hosted") == pytest.approx(0.1)
    assert r.rate("local") == pytest.approx(0.1)


def test_reallocation_on_close():
    r = reg(a=(9, 1.0, 9), b=(9, 9, 9), c=(9, 9, 9))
    f1 = r.open([("a", RES_OUT), ("b", RES_IN)], demand=99, group="f1")
    r.open([("a", RES_OUT), ("c", RES_IN)], demand=99, group="f2")
    assert r.rate("f1") == pytest.approx(0.5)
    r.close(f1)
    assert r.rate("f2") == pytest.approx(1.0)


def test_weighted_shares_on_saturated_resource():
    """Audit fix #1/#5: a flow bundling m unit-streams gets m x the share of a
    saturated resource (weighted max-min), not one flow-share."""
    reg = CapacityRegistry()
    reg.set_node("store", NodeCaps(nic_in=10.0, nic_out=999.0, disk_w=999.0))
    big = reg.open([("store", RES_IN)], demand=999.0, weight=8.0)
    small = reg.open([("store", RES_IN)], demand=999.0, weight=2.0)
    assert abs(reg.flow_rate(big) - 8.0) < 1e-9
    assert abs(reg.flow_rate(small) - 2.0) < 1e-9


def test_weighted_demand_caps_per_unit_stream():
    """Weight interacts with demand caps: a bundle freezes at its demand and
    the leftover goes to others (still max-min)."""
    reg = CapacityRegistry()
    reg.set_node("store", NodeCaps(nic_in=10.0, nic_out=999.0, disk_w=999.0))
    capped = reg.open([("store", RES_IN)], demand=2.0, weight=8.0)   # wants little
    hungry = reg.open([("store", RES_IN)], demand=999.0, weight=2.0)
    assert abs(reg.flow_rate(capped) - 2.0) < 1e-9
    assert abs(reg.flow_rate(hungry) - 8.0) < 1e-9


# ---------------------------------------------------------------------------
# Route C — per-node persistent BACKGROUND front-end (ingress) traffic.
# The knob is cluster.background_nic_gbps; the engine installs one STANDING
# ingress demand of that magnitude on every node as it is registered. These
# pencil-math hand-checks pin the physics to the digit: a checkpoint stream
# arriving at a node whose NIC already carries background B gets EXACTLY the
# reduced max-min share (nic_in - B), and default-off installs nothing.
# ---------------------------------------------------------------------------

def test_background_default_off_installs_nothing():
    """Default background_nic_gbps = 0.0 => no standing flow, existing scenarios
    are byte-identical (the whole point of the default)."""
    r = CapacityRegistry()
    r.set_node("a", NodeCaps(nic_in=50.0, nic_out=50.0, disk_w=6.0))
    assert r.flows == set()
    assert r._background_flows == {}


def test_background_reduces_single_stream_to_exact_maxmin_share():
    """PENCIL MATH (assert to the digit): background B on a node's ingress NIC +
    one checkpoint stream => the stream's rate is EXACTLY the max-min share
    nic_in - B. B is a competing standing demand capped at B, so it freezes at B
    and the lone hungry stream takes the entire remainder."""
    r = CapacityRegistry()
    r.background_nic_gbps = 5.0                      # 10% of a 50 GB/s front-end NIC
    r.set_node("s", NodeCaps(nic_in=50.0, nic_out=50.0, disk_w=50.0))   # sender
    r.set_node("d", NodeCaps(nic_in=50.0, nic_out=50.0, disk_w=50.0))   # receiver
    # a checkpoint stream s.out -> d.in (disk not limiting): the only contention is
    # d's ingress, which the standing background already loads at B = 5.0.
    r.open([("s", RES_OUT), ("d", RES_IN)], demand=999.0, group="ckpt")
    assert r.rate("ckpt") == pytest.approx(50.0 - 5.0, abs=1e-12)   # == 45.0 exactly


def test_background_small_rate_is_the_exact_leftover():
    """The sourced data-loading magnitude (~5 MB/s = 0.005 GB/s) leaves the stream
    at nic_in - 0.005 to the digit — the delta-vs-control the experiment expects
    to be indistinguishable."""
    r = CapacityRegistry()
    r.background_nic_gbps = 0.005
    r.set_node("s", NodeCaps(nic_in=50.0, nic_out=50.0, disk_w=50.0))
    r.set_node("d", NodeCaps(nic_in=50.0, nic_out=50.0, disk_w=50.0))
    r.open([("s", RES_OUT), ("d", RES_IN)], demand=999.0, group="ckpt")
    assert r.rate("ckpt") == pytest.approx(50.0 - 0.005, abs=1e-12)  # == 49.995


def test_background_shared_reduced_pipe_two_streams():
    """Two checkpoint streams into one node share the pipe LEFT OVER after the
    standing background: each gets (nic_in - B)/2 exactly (still max-min)."""
    r = CapacityRegistry()
    r.background_nic_gbps = 4.0
    r.set_node("s1", NodeCaps(nic_in=50.0, nic_out=50.0, disk_w=50.0))
    r.set_node("s2", NodeCaps(nic_in=50.0, nic_out=50.0, disk_w=50.0))
    r.set_node("d", NodeCaps(nic_in=20.0, nic_out=50.0, disk_w=50.0))   # ingress 20
    r.open([("s1", RES_OUT), ("d", RES_IN)], demand=999.0, group="c1")
    r.open([("s2", RES_OUT), ("d", RES_IN)], demand=999.0, group="c2")
    # standing background 4.0 on d.in, then 16.0 shared two ways = 8.0 each.
    assert r.rate("c1") == pytest.approx((20.0 - 4.0) / 2, abs=1e-12)   # == 8.0
    assert r.rate("c2") == pytest.approx((20.0 - 4.0) / 2, abs=1e-12)   # == 8.0
    # the standing demand itself sits at exactly B on d's ingress.
    bg = r._background_flows["d"]
    assert r.flow_rate(bg) == pytest.approx(4.0, abs=1e-12)


def test_background_egress_unaffected_full_duplex():
    """Data loading = ingress: a node SENDING a checkpoint (egress) is NOT slowed
    by its own background ingress demand — full-duplex, in/out independent."""
    r = CapacityRegistry()
    r.background_nic_gbps = 10.0
    r.set_node("s", NodeCaps(nic_in=50.0, nic_out=50.0, disk_w=50.0))
    r.set_node("d", NodeCaps(nic_in=999.0, nic_out=50.0, disk_w=999.0))
    # stream s.out -> d.in; d.in is 999 (uncontended), so the bottleneck is s.out.
    # s also carries background on s.IN, which must not touch s.out.
    r.open([("s", RES_OUT), ("d", RES_IN)], demand=999.0, group="ckpt")
    assert r.rate("ckpt") == pytest.approx(50.0, abs=1e-12)   # full s.out line rate
