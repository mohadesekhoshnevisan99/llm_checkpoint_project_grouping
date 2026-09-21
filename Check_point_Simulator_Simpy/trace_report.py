"""trace_report.py — who checkpoints where: donor-placement inspection for crossjob traces.

Answers, from any run_crossjob event log (plain or .gz):
  --job JOB     placement timeline for one job: every checkpoint, which donor nodes got
                its shards, path (peer/partial/fallback/store), size and duration, plus
                any recoveries and where they read from
  --matrix PNG  donation heatmap: donor-class x host-class shard counts (who stores on whom)
  --donors N    the N busiest donor nodes (hosted shards, GB, from which jobs)
  (no flags)    run-wide summary: paths, per-class donation totals, refusal-free placement stats

Works on results/crossjob_log.jsonl and the archived results/traces/*.jsonl.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f]


def job_of(node_name: str) -> str:
    return node_name.split("-rank-")[0] if "-rank-" in node_name else node_name


def job_class(job: str) -> str:
    for prefix in ("fxl", "frontier", "standard", "be"):
        if job.startswith(prefix):
            return {"fxl": "frontier_xl"}.get(prefix, prefix)
    return "other"


def flush_events(events):
    for e in events:
        det = e.get("details") or {}
        if det.get("checkpoint_strategy") == "crossjob_peer" and "path" in det:
            yield e, det


def print_job_timeline(events, job: str) -> None:
    print(f"=== checkpoint placements for {job} ===")
    rows = []
    for e, det in flush_events(events):
        if e.get("job_id") != job:
            continue
        rows.append((e["start"], det.get("checkpoint_group", "?"),
                     det.get("checkpoint_owner_rank"), det["path"],
                     det.get("donors") or [], e.get("data_gb"), e["end"] - e["start"]))
    for t, grp, rank, path, donors, gb, dur in sorted(rows):
        it = grp.split("-iteration-")[-1]
        if donors:
            each = gb / len(donors)
            where = " + ".join(donors) + f"   ({each:.2f} GB to each)"
        else:
            where = "__store__" if path == "store" else "LOCAL ONLY (not node-loss durable)"
        print(f"  t={t:7.1f}s  it{it:>3} rank{rank}  {path:13s} {gb:.2f} GB "
              f"in {dur:5.1f}s  ->  {where}")
    print(f"\n=== recoveries for {job} ===")
    any_rec = False
    for e in events:
        det = e.get("details") or {}
        if e.get("job_id") == job and e.get("operation") == "dram_to_gpu_restore":
            any_rec = True
            print(f"  t={e['start']:7.1f}s  rank{e.get('rank')}  restored from "
                  f"{det.get('checkpoint_source_tier')} @ iteration "
                  f"{det.get('checkpoint_iteration')}")
    if not any_rec:
        print("  (none)")


def donation_matrix(events):
    """(owner class, host class) -> [shard placements, GB moved]. Rows = whose
    CHECKPOINT it is; columns = whose nodes HOST it (the donors, in peerd terms)."""
    m = defaultdict(lambda: [0, 0.0])
    sizes = defaultdict(list)
    for e, det in flush_events(events):
        oc = job_class(e.get("job_id") or "?")
        if e.get("data_gb"):
            sizes[oc].append(e["data_gb"])
        if not det.get("donors"):
            continue
        per_donor_gb = (e.get("data_gb") or 0.0) / len(det["donors"])
        for d in det["donors"]:
            cell = m[(oc, job_class(job_of(d)))]
            cell[0] += 1
            cell[1] += per_donor_gb
    mean_size = {c: (sum(v) / len(v) if v else 0.0) for c, v in sizes.items()}
    return m, mean_size


def print_summary(events) -> None:
    paths = Counter()
    per_donor = Counter()
    gb_per_donor = Counter()
    donor_sources = defaultdict(Counter)
    for e, det in flush_events(events):
        paths[det["path"]] += 1
        for d in det.get("donors") or []:
            per_donor[d] += 1
            gb_per_donor[d] += (e.get("data_gb") or 0) / max(len(det["donors"]), 1)
            donor_sources[d][e.get("job_id")] += 1
    print("=== run-wide flush paths ===")
    for p, n in paths.most_common():
        print(f"  {p:15s} {n}")
    print("\n=== placement matrix: rows = checkpoint OWNER, columns = HOSTING nodes ===")
    print("    (cell: shard placements / GB moved)")
    m, mean_size = donation_matrix(events)
    classes = ["frontier_xl", "frontier", "standard", "be"]
    print(f"  {'':22s}" + "".join(f"{c:>18s}" for c in classes))
    for oc in classes:
        label = f"{oc} ({mean_size.get(oc, 0):.1f} GB/ckpt)"
        row = "".join(f"{m[(oc, hc)][0]:>8d}/{m[(oc, hc)][1]:>7.0f}GB" for hc in classes)
        print(f"  {label:22s}" + row)
    # peers used per owner class
    ks = defaultdict(list)
    uniq = defaultdict(set)
    for e, det in flush_events(events):
        if det.get("donors"):
            oc = job_class(e.get("job_id") or "?")
            ks[oc].append(len(det["donors"]))
            uniq[oc].update(det["donors"])
    print("\n=== peers used, by owner class ===")
    for oc in classes:
        if ks[oc]:
            print(f"  {oc:12s} mean k = {sum(ks[oc])/len(ks[oc]):.2f} donors/ckpt, "
                  f"{len(uniq[oc])} distinct donor nodes used over the run")
    return per_donor, gb_per_donor, donor_sources


def print_top_donors(per_donor, gb, sources, n: int) -> None:
    print(f"\n=== top {n} donor nodes ===")
    for name, cnt in per_donor.most_common(n):
        top = ", ".join(f"{j}:{c}" for j, c in sources[name].most_common(3))
        print(f"  {name:22s} hosted {cnt:3d} shards ({gb[name]:6.1f} GB)  from: {top}")


def render_matrix(events, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    m, mean_size = donation_matrix(events)
    classes = ["frontier_xl", "frontier", "standard", "be"]
    counts = np.array([[m[(o, h)][0] for h in classes] for o in classes], float)
    gbs = np.array([[m[(o, h)][1] for h in classes] for o in classes], float)
    ylabels = [f"{c}\n({mean_size.get(c, 0):.1f} GB/ckpt)" for c in classes]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), layout="constrained")
    for ax, grid, title, unit in ((axes[0], counts, "shard placements (count)", "%d"),
                                  (axes[1], gbs, "checkpoint traffic hosted (GB)", "%.0f")):
        ax.imshow(grid, cmap="Greens")
        ax.set_xticks(range(4), classes, fontsize=9)
        ax.set_yticks(range(4), ylabels, fontsize=8.5)
        ax.set_xlabel("HOSTING class — whose nodes lend their disks (the donors)")
        if ax is axes[0]:
            ax.set_ylabel("OWNER class — whose checkpoint it is")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, unit % grid[i, j], ha="center", va="center", fontsize=9.5,
                        color="white" if grid[i, j] > grid.max() * 0.6 else "#1a1a19")
        ax.set_title(title, fontsize=11, fontweight="bold")
    fig.suptitle("Who checkpoints onto whom — read rows -> columns "
                 "(e.g. standard-row x frontier-col: standard jobs' shards stored on frontier nodes)",
                 fontsize=10)
    fig.savefig(out, dpi=150)
    print(f"matrix -> {out}")


def traffic_series(events, bucket: float = 5.0):
    """Aggregate checkpoint-network traffic (GB/s) per time bucket: peer + store
    flushes and network recoveries, each event's bytes spread over its span."""
    end_max = max((e["end"] for e in events), default=0.0)
    n = int(end_max / bucket) + 1
    rate = [0.0] * n
    for e in events:
        det = e.get("details") or {}
        op = e.get("operation") or ""
        is_net_flush = det.get("path") in ("peer", "peer_partial", "store")
        is_net_recovery = "recovery_chunk" in op and "NETWORK" in (e.get("resources") or [])
        if not (is_net_flush or is_net_recovery):
            continue
        gb, s, t = e.get("data_gb") or 0.0, e["start"], e["end"]
        if t <= s or gb <= 0:
            continue
        r = gb / (t - s)
        b0, b1 = int(s / bucket), int(t / bucket)
        for b in range(b0, min(b1, n - 1) + 1):
            lo, hi = max(s, b * bucket), min(t, (b + 1) * bucket)
            rate[b] += r * max(0.0, hi - lo) / bucket
    return rate, bucket


def render_traffic(events, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rate, bucket = traffic_series(events)
    xs = [i * bucket for i in range(len(rate))]
    fig, ax = plt.subplots(figsize=(9.5, 3.4), layout="constrained")
    ax.fill_between(xs, rate, color="#008300", alpha=0.75, lw=0)
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("checkpoint network traffic (GB/s)")
    total = sum(r * bucket for r in rate)
    ax.set_title(f"Checkpoint network traffic over time (total {total:.0f} GB)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=150)
    print(f"traffic -> {out}")


def render_flows(events, out: Path, job: str | None = None, window: float = 60.0) -> None:
    """Job-to-job checkpoint traffic over time. With job=X: rows = the jobs RECEIVING
    X's shards, columns = time windows, cell = GB received in that window. Without job:
    rows = class->class flows. Answers "who sent how much to whom, when"."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    end_max = max((e["end"] for e in events), default=0.0)
    nwin = int(end_max / window) + 1
    flows = defaultdict(lambda: np.zeros(nwin))
    for e, det in flush_events(events):
        if not det.get("donors"):
            continue
        if job is not None and e.get("job_id") != job:
            continue
        w = min(int(e["end"] / window), nwin - 1)
        per = (e.get("data_gb") or 0.0) / len(det["donors"])
        for d in det["donors"]:
            key = job_of(d) if job is not None else \
                f"{job_class(e.get('job_id') or '?')} -> {job_class(job_of(d))}"
            flows[key][w] += per
    if not flows:
        print("no peer flows to render")
        return
    totals = {k: v.sum() for k, v in flows.items()}
    keys = sorted(totals, key=totals.get, reverse=True)
    if job is not None and len(keys) > 14:
        head, tail = keys[:14], keys[14:]
        other = np.zeros(nwin)
        for k in tail:
            other += flows[k]
        flows["(other jobs)"] = other
        keys = head + ["(other jobs)"]
    grid = np.array([flows[k] for k in keys])
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(keys) + 2.2), layout="constrained")
    im = ax.imshow(grid, aspect="auto", cmap="Greens", interpolation="nearest")
    ax.set_yticks(range(len(keys)),
                  [f"{k}  ({totals.get(k, flows[k].sum()):.0f} GB)" for k in keys],
                  fontsize=8)
    step = max(1, nwin // 12)
    ax.set_xticks(range(0, nwin, step), [f"{int(i * window)}" for i in range(0, nwin, step)],
                  fontsize=8)
    ax.set_xlabel(f"time (s), {window:.0f}-s windows")
    src_label = f"where {job}'s checkpoint data went" if job else \
        "class-to-class checkpoint flows"
    ax.set_title(f"{src_label} — GB per window", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85, label="GB received in window")
    fig.savefig(out, dpi=150)
    print(f"flows -> {out}")


def _sequence_fig(events):
    """UML-style transfer sequence: one lifeline per job (grouped by class), an arrow
    per shard transfer at its time. Hover gives exact bytes/donors/duration. Toggle
    event kinds via the legend. Interactive HTML (plotly), zoom + range-slider."""
    import plotly.graph_objects as go

    jobs = sorted({e.get("job_id") for e in events if e.get("job_id")},
                  key=lambda j: ({"frontier_xl": 0, "frontier": 1,
                                  "standard": 2, "be": 3}.get(job_class(j), 4), j))
    row = {j: i for i, j in enumerate(jobs)}
    row["__store__"] = len(jobs) + 1

    KINDS = {
        "peer flush": dict(color="#008300", dash="solid"),
        "store flush": dict(color="#e34948", dash="solid"),
        "recovery": dict(color="#4a3aa7", dash="dot"),
    }
    segs = {k: dict(x=[], y=[], hx=[], hy=[], ht=[]) for k in KINDS}

    def add(kind, t, src_row, dst_row, text):
        s = segs[kind]
        s["x"] += [t, t, None]
        s["y"] += [src_row, dst_row, None]
        s["hx"].append(t)
        s["hy"].append((src_row + dst_row) / 2)
        s["ht"].append(text)

    for e in events:
        det = e.get("details") or {}
        op = e.get("operation") or ""
        j = e.get("job_id")
        if det.get("path") in ("peer", "peer_partial") and det.get("donors"):
            per = (e.get("data_gb") or 0) / len(det["donors"])
            for d in det["donors"]:
                dj = job_of(d)
                add("peer flush", e["start"], row.get(j, 0), row.get(dj, 0),
                    f"t={e['start']:.0f}s  {j} → {dj}<br>{per:.2f} GB shard "
                    f"({det.get('checkpoint_group')})<br>transfer {e['end']-e['start']:.1f}s")
        elif det.get("path") == "store":
            add("store flush", e["start"], row.get(j, 0), row["__store__"],
                f"t={e['start']:.0f}s  {j} → store<br>{e.get('data_gb') or 0:.2f} GB "
                f"({det.get('checkpoint_group')})")
        elif "recovery_chunk" in op and "crossjob" in op:
            srcs = (e.get("source") or "").split("+")
            for s_ in srcs:
                dj = job_of(s_.split("/")[0])
                add("recovery", e["start"], row.get(dj, 0), row.get(j, 0),
                    f"t={e['start']:.0f}s  RECOVERY {dj} → {j}<br>"
                    f"{e.get('data_gb') or 0:.2f} GB from donors")

    fig = go.Figure()
    for kind, style in KINDS.items():
        s = segs[kind]
        if not s["hx"]:
            continue
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="lines", name=kind,
                                 line=dict(color=style["color"], width=1.1,
                                           dash=style["dash"]),
                                 hoverinfo="skip", legendgroup=kind))
        fig.add_trace(go.Scatter(x=s["hx"], y=s["hy"], mode="markers", name=kind,
                                 marker=dict(color=style["color"], size=5),
                                 hovertext=s["ht"], hoverinfo="text",
                                 legendgroup=kind, showlegend=False))
    names = jobs + ["", "__store__"]
    fig.update_layout(
        title="Transfer sequence — every arrow is one shard moving between jobs "
              "(hover for bytes; drag to zoom; legend toggles kinds)",
        yaxis=dict(tickvals=list(range(len(names))), ticktext=names,
                   tickfont=dict(size=7), autorange="reversed"),
        xaxis=dict(title="simulation time (s)", rangeslider=dict(visible=True)),
        height=max(600, 11 * len(names)), template="plotly_white",
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def render_sequence(events, out: Path) -> None:
    fig = _sequence_fig(events)
    fig.write_html(out, include_plotlyjs=True)
    print(f"sequence -> {out}")


def render_report(events, out: Path, title: str) -> None:
    """ONE self-contained HTML with everything: stats, sequence diagram, traffic,
    donation matrices, class flows over time, top donors, per-job ledgers."""
    import plotly.graph_objects as go
    import numpy as np

    classes = ["frontier_xl", "frontier", "standard", "be"]

    # ---- stats ----
    paths = Counter()
    total_gb = 0.0
    per_donor = Counter()
    gb_per_donor = Counter()
    donor_sources = defaultdict(Counter)
    jobs = sorted({e.get("job_id") for e in events if e.get("job_id")})
    for e, det in flush_events(events):
        paths[det["path"]] += 1
        if det["path"].startswith("peer") or det["path"] == "store":
            total_gb += e.get("data_gb") or 0
        for d in det.get("donors") or []:
            per_donor[d] += 1
            gb_per_donor[d] += (e.get("data_gb") or 0) / max(len(det["donors"]), 1)
            donor_sources[d][e.get("job_id")] += 1
    restores = [e for e in events if e.get("operation") == "dram_to_gpu_restore"]
    durable = sum(v for k, v in paths.items() if k.startswith("peer") or k == "store")
    tot = sum(paths.values())
    end = max((e["end"] for e in events), default=0)
    cards = [
        ("jobs", len(jobs)), ("sim time", f"{end:,.0f} s"),
        ("flushes", tot), ("durable", f"{durable/tot:.1%}" if tot else "-"),
        ("durable data", f"{total_gb:,.0f} GB"), ("recoveries", len(restores)),
    ]
    card_html = "".join(
        f'<div style="background:#f4f6f4;border:1px solid #dfe3df;border-radius:10px;'
        f'padding:12px 18px;min-width:120px"><div style="color:#666;font-size:12px">{k}</div>'
        f'<div style="font-size:22px;font-weight:700">{v}</div></div>'
        for k, v in cards)
    path_html = " · ".join(f"{k}: <b>{v}</b>" for k, v in paths.most_common())

    figs = []

    # big traces: browsers give up rendering multi-hundred-MB pages (a full
    # realwidth trace embeds millions of shapes -> 240 MB HTML). Downsample the
    # per-event charts to a representative WINDOW and say so in the title;
    # aggregate charts (traffic, matrices, donors) still use ALL events.
    MAX_EVT = 15_000
    win_events, win_note = events, ""
    if len(events) > MAX_EVT:
        end_all = max(e["end"] for e in events)
        w0, w1 = 0.30 * end_all, 0.30 * end_all
        while True:
            win_events = [e for e in events if e["end"] > w0 and e["start"] < w1]
            if len(win_events) >= MAX_EVT or w1 >= end_all:
                break
            w1 = min(end_all, w1 + 0.05 * end_all)
        win_events = win_events[:MAX_EVT]
        win_note = (f" — DOWNSAMPLED window t={w0:,.0f}–{w1:,.0f}s "
                    f"({len(win_events):,} of {len(events):,} events; "
                    f"aggregate charts below use the full trace)")

    # ---- the colored per-rank occupancy timeline (from the team's visualiser) ----
    try:
        import visualise as V
        occ = V.make_resource_timeline(win_events, V.build_operation_styles(win_events))
        figs.append(("GPU/CPU occupancy timeline — colored by operation "
                     "(training, captures, peer transfers, recoveries)" + win_note, occ))
    except Exception as ex:      # visualiser optional; report still renders
        print(f"occupancy timeline skipped: {ex}")

    # ---- sequence ----
    figs.append(("Transfer sequence — every arrow is one shard between jobs "
                 "(hover for bytes, drag to zoom, legend toggles kinds)" + win_note,
                 _sequence_fig(win_events)))

    # ---- traffic over time ----
    rate, bucket = traffic_series(events)
    f = go.Figure(go.Scatter(x=[i * bucket for i in range(len(rate))], y=rate,
                             fill="tozeroy", line=dict(color="#008300", width=1)))
    f.update_layout(template="plotly_white", height=280,
                    xaxis_title="time (s)", yaxis_title="network GB/s")
    figs.append(("Checkpoint network traffic over time", f))

    # ---- donation matrices ----
    m, mean_size = donation_matrix(events)
    ylab = [f"{c} ({mean_size.get(c, 0):.1f} GB/ckpt)" for c in classes]
    for name, idx, unit in (("shard placements (count)", 0, ""),
                            ("traffic hosted (GB)", 1, " GB")):
        z = [[m[(o, h)][idx] for h in classes] for o in classes]
        f = go.Figure(go.Heatmap(z=z, x=classes, y=ylab, colorscale="Greens",
                                 text=[[f"{v:.0f}{unit}" for v in r] for r in z],
                                 texttemplate="%{text}"))
        f.update_layout(template="plotly_white", height=340,
                        xaxis_title="HOSTING class (lends its disks)",
                        yaxis_title="OWNER class (whose checkpoint)")
        figs.append((f"Who checkpoints onto whom — {name}", f))

    # ---- class flows over time ----
    end_max = max((e["end"] for e in events), default=0)
    win = max(60.0, end_max / 40)
    nwin = int(end_max / win) + 1
    flows = defaultdict(lambda: np.zeros(nwin))
    for e, det in flush_events(events):
        if det.get("donors"):
            w = min(int(e["end"] / win), nwin - 1)
            per = (e.get("data_gb") or 0) / len(det["donors"])
            for d in det["donors"]:
                flows[f"{job_class(e.get('job_id') or '?')} → {job_class(job_of(d))}"][w] += per
    keys = sorted(flows, key=lambda k: -flows[k].sum())
    if keys:
        f = go.Figure(go.Heatmap(z=[flows[k] for k in keys],
                                 y=[f"{k} ({flows[k].sum():.0f} GB)" for k in keys],
                                 x=[i * win for i in range(nwin)], colorscale="Greens"))
        f.update_layout(template="plotly_white", height=90 + 28 * len(keys),
                        xaxis_title=f"time (s), {win:.0f}-s windows")
        figs.append(("Class-to-class flows over time (GB per window)", f))

    # ---- tables ----
    donors_rows = "".join(
        f"<tr><td>{n}</td><td>{c}</td><td>{gb_per_donor[n]:.1f}</td>"
        f"<td>{', '.join(f'{j}:{x}' for j, x in donor_sources[n].most_common(3))}</td></tr>"
        for n, c in per_donor.most_common(12))
    ledgers = []
    for j in jobs:
        rows = []
        for e, det in flush_events(events):
            if e.get("job_id") != j:
                continue
            it = (det.get("checkpoint_group") or "?").split("-iteration-")[-1]
            donors = det.get("donors") or []
            each = f" ({(e.get('data_gb') or 0)/len(donors):.2f} GB each)" if donors else ""
            where = (" + ".join(donors) + each) if donors else                 ("store" if det["path"] == "store" else "LOCAL (not durable)")
            rows.append(f"<tr><td>{e['start']:.0f}</td><td>{it}</td>"
                        f"<td>{det.get('checkpoint_owner_rank')}</td><td>{det['path']}</td>"
                        f"<td>{e.get('data_gb') or 0:.2f}</td><td>{e['end']-e['start']:.1f}</td>"
                        f"<td>{where}</td></tr>")
        recs = "".join(
            f"<tr><td colspan=6>t={e['start']:.0f}s rank{e.get('rank')} restored from "
            f"{(e.get('details') or {}).get('checkpoint_source_tier')} @ it "
            f"{(e.get('details') or {}).get('checkpoint_iteration')}</td></tr>"
            for e in restores if e.get("job_id") == j)
        ledgers.append(
            f"<details><summary><b>{j}</b> ({len(rows)} flushes)</summary>"
            f"<table border=0 cellpadding=4 style='font-size:12px;border-collapse:collapse'>"
            f"<tr style='color:#666'><th>t(s)</th><th>iter</th><th>rank</th><th>path</th>"
            f"<th>GB</th><th>dur(s)</th><th>destination</th></tr>"
            + "".join(rows) + recs + "</table></details>")

    parts = [f"<h1 style='font-family:sans-serif'>{title}</h1>",
             f"<div style='display:flex;gap:12px;flex-wrap:wrap;font-family:sans-serif'>{card_html}</div>",
             f"<p style='font-family:sans-serif;color:#444'>paths — {path_html}</p>"]
    for i, (heading, f) in enumerate(figs):
        parts.append(f"<h2 style='font-family:sans-serif'>{heading}</h2>")
        parts.append(f.to_html(full_html=False, include_plotlyjs=(i == 0)))
    parts.append("<h2 style='font-family:sans-serif'>Top donor nodes</h2>"
                 "<table border=0 cellpadding=4 style='font-family:sans-serif;font-size:13px'>"
                 "<tr style='color:#666'><th>node</th><th>shards hosted</th><th>GB</th>"
                 "<th>top senders</th></tr>" + donors_rows + "</table>")
    parts.append("<h2 style='font-family:sans-serif'>Per-job placement ledgers "
                 "(click to expand)</h2>" +
                 "<div style='font-family:sans-serif'>" + "".join(ledgers) + "</div>")
    out.write_text("<!doctype html><html><head><meta charset='utf-8'>"
                   f"<title>{title}</title></head><body style='max-width:1250px;"
                   "margin:20px auto;padding:0 16px'>" + "\n".join(parts) +
                   "</body></html>")
    print(f"report -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("results/crossjob_log.jsonl"))
    ap.add_argument("--job", help="placement timeline for this job id")
    ap.add_argument("--matrix", type=Path, help="render donation heatmap PNG here")
    ap.add_argument("--traffic", type=Path, help="render traffic-over-time PNG here")
    ap.add_argument("--flows", type=Path, help="render job-to-job flow heatmap PNG here")
    ap.add_argument("--sequence", type=Path, help="render interactive transfer-sequence HTML here")
    ap.add_argument("--report", type=Path, help="ONE self-contained HTML with everything")
    ap.add_argument("--window", type=float, default=60.0, help="window size for --flows")
    ap.add_argument("--donors", type=int, default=10, help="top-N donor nodes")
    args = ap.parse_args()
    events = load(args.log)
    if args.report:
        render_report(events, args.report, title=args.log.name.replace(".jsonl.gz", ""))
        return
    if args.sequence:
        render_sequence(events, args.sequence)
    if args.flows:
        render_flows(events, args.flows, job=args.job, window=args.window)
        if not args.job:
            return
    if args.job:
        print_job_timeline(events, args.job)
        return
    per_donor, gb, sources = print_summary(events)
    print_top_donors(per_donor, gb, sources, args.donors)
    if args.matrix:
        render_matrix(events, args.matrix)
    if args.traffic:
        render_traffic(events, args.traffic)


if __name__ == "__main__":
    main()
