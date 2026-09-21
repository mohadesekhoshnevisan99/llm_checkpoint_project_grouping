"""Build a standalone comparison page from scenario result JSON files.

The input files are the JSON documents written by ``run_scenario.py``.  More
than one file may be supplied so independently-run policy families can share a
single landing page.  Rows with the same ``(arm, seed)`` are de-duplicated;
the value from the last input file wins.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote

import plotly.graph_objects as go
import plotly.io as pio


@dataclass(frozen=True)
class PolicySummary:
    """Metrics for one policy, averaged across its simulation seeds."""

    arm: str
    policy: str
    is_ideal: bool
    seeds: tuple[object, ...]
    completion_s: float | None
    makespan_s: float | None
    censored_jobs_mean: float
    censored_runs: int
    censored_job_names: tuple[str, ...]
    overhead_pct: float | None
    goodput_pct: float | None
    failures: Mapping[str, float]
    restores: float
    recovery_sources: Mapping[str, float]
    initial_state_recoveries: float | None
    durable_pct: float | None
    flushes: float
    aborted_flushes: float
    fallback_flushes: float
    partial_grants: float
    lost_iters_max: float | None

    @property
    def run_count(self) -> int:
        return len(self.seeds)

    @property
    def failure_count(self) -> float:
        return sum(self.failures.values())


def _scenario_identity(value: object) -> str | None:
    if value is None:
        return None
    # Result files may have been produced on Windows or POSIX, or may mix an
    # absolute and relative spelling.  The scenario filename is the stable bit.
    return Path(str(value).replace("\\", "/")).name


def load_result_rows(paths: Sequence[Path]) -> tuple[str | None, list[dict]]:
    """Load and combine run_scenario rows, validating the scenario identity."""
    if not paths:
        raise ValueError("at least one result JSON file is required")

    scenario: str | None = None
    ordered_keys: list[tuple[object, ...]] = []
    rows_by_key: dict[tuple[object, ...], dict] = {}
    anonymous_index = 0

    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("rows"), list):
            raise ValueError(f"{path} is not a run_scenario result (missing rows list)")

        found = _scenario_identity(document.get("scenario"))
        if found is not None:
            if scenario is not None and found != scenario:
                raise ValueError(
                    f"result files describe different scenarios: {scenario!r} and "
                    f"{found!r}"
                )
            scenario = found

        for raw in document["rows"]:
            if not isinstance(raw, dict):
                raise ValueError(f"{path} contains a non-object result row")
            arm = raw.get("arm")
            if arm is None:
                raise ValueError(f"{path} contains a result row without an arm")
            if "seed" in raw:
                key = (str(arm), raw["seed"])
            else:
                # A missing seed is tolerated for hand-authored/legacy results,
                # but such rows cannot safely be considered duplicates.
                key = (str(arm), "__anonymous__", anonymous_index)
                anonymous_index += 1
            if key not in rows_by_key:
                ordered_keys.append(key)
            rows_by_key[key] = dict(raw)

    return scenario, [rows_by_key[key] for key in ordered_keys]


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _mean_counter(rows: Sequence[dict], field: str) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for row in rows:
        values = row.get(field) or {}
        if not isinstance(values, Mapping):
            continue
        for name, value in values.items():
            number = _number(value)
            if number is not None:
                totals[str(name)] += number
    divisor = len(rows) or 1
    return {
        name: total / divisor
        for name, total in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    }


def _mean_counter_entry(
    rows: Sequence[dict], field: str, entry: str
) -> float | None:
    """Mean one counter entry, preserving missing metadata as unknown.

    An explicitly present empty mapping is a measured zero. A missing or invalid
    mapping is not silently converted to zero because legacy result files may not
    have recorded the metric at all.
    """
    values: list[float] = []
    for row in rows:
        counter = row.get(field)
        if not isinstance(counter, Mapping):
            continue
        value = _number(counter.get(entry, 0))
        if value is not None:
            values.append(value)
    return _mean(values)


def _is_ideal(row: Mapping[str, object], ideal_policy: str) -> bool:
    candidates = {
        str(row.get("arm", "")).casefold(),
        str(row.get("policy_name", "")).casefold(),
    }
    return (
        ideal_policy.casefold() in candidates
        or str(row.get("policy_kind", "")).casefold() == "ideal"
    )


def summarise_policies(
    rows: Sequence[dict], *, ideal_policy: str = "ideal"
) -> list[PolicySummary]:
    """Aggregate rows by arm and compute seed-paired ideal comparisons."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        arm = str(row["arm"])
        if arm not in grouped:
            order.append(arm)
        grouped[arm].append(row)

    ideal_rows = [row for row in rows if _is_ideal(row, ideal_policy)]
    ideal_by_seed = {
        row.get("seed"): _number(row.get("train_end_s"))
        for row in ideal_rows
        if _number(row.get("train_end_s")) is not None
    }
    ideal_mean = _mean(_number(row.get("train_end_s")) for row in ideal_rows)

    summaries: list[PolicySummary] = []
    for arm in order:
        policy_rows = grouped[arm]
        comparisons: list[tuple[float, float]] = []
        for row in policy_rows:
            completion = _number(row.get("train_end_s"))
            ideal = ideal_by_seed.get(row.get("seed"), ideal_mean)
            if completion is not None and ideal is not None and completion > 0:
                comparisons.append(
                    ((completion / ideal - 1.0) * 100.0, ideal / completion * 100.0)
                )

        censored_names: Counter[str] = Counter()
        censored_runs = 0
        censored_total = 0
        for row in policy_rows:
            names = row.get("censored_jobs") or []
            if not isinstance(names, list):
                names = [str(names)]
            if names:
                censored_runs += 1
            censored_total += len(names)
            censored_names.update(str(name) for name in names)

        label = next(
            (
                str(row["policy_name"])
                for row in policy_rows
                if row.get("policy_name")
                and str(row["policy_name"]) != str(row.get("arm"))
            ),
            arm,
        )
        durable = _mean(_number(row.get("durable_frac")) for row in policy_rows)
        summaries.append(
            PolicySummary(
                arm=arm,
                policy=label,
                is_ideal=any(
                    _is_ideal(row, ideal_policy) for row in policy_rows
                ),
                seeds=tuple(row.get("seed", "unknown") for row in policy_rows),
                completion_s=_mean(
                    _number(row.get("train_end_s")) for row in policy_rows
                ),
                makespan_s=_mean(
                    _number(row.get("makespan_s")) for row in policy_rows
                ),
                censored_jobs_mean=censored_total / len(policy_rows),
                censored_runs=censored_runs,
                censored_job_names=tuple(censored_names),
                overhead_pct=_mean(value[0] for value in comparisons),
                goodput_pct=_mean(value[1] for value in comparisons),
                failures=_mean_counter(policy_rows, "failures"),
                restores=_mean(
                    _number(row.get("restores")) for row in policy_rows
                )
                or 0.0,
                recovery_sources=_mean_counter(
                    policy_rows, "recovery_source_tiers"
                ),
                initial_state_recoveries=_mean_counter_entry(
                    policy_rows, "recovery_source_tiers", "initial_state"
                ),
                durable_pct=durable * 100.0 if durable is not None else None,
                flushes=_mean(
                    _number(row.get("rank_flushes")) for row in policy_rows
                )
                or 0.0,
                aborted_flushes=_mean(
                    _number(row.get("aborted_flushes")) for row in policy_rows
                )
                or 0.0,
                fallback_flushes=_mean(
                    _number(row.get("fallback_flushes")) for row in policy_rows
                )
                or 0.0,
                partial_grants=_mean(
                    _number(row.get("partial_grants")) for row in policy_rows
                )
                or 0.0,
                lost_iters_max=_mean(
                    _number(row.get("lost_iters_max")) for row in policy_rows
                ),
            )
        )
    return summaries


def _format_number(value: float, *, decimals: int = 1) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.{decimals}f}"


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 3600:
        return f"{value:,.1f} s ({value / 3600:.2f} h)"
    return f"{value:,.1f} s"


def _format_pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:,.1f}%"


def _format_counter(values: Mapping[str, float]) -> str:
    if not values:
        return "0"
    total = sum(values.values())
    details = "; ".join(
        f"{html.escape(name)} {_format_number(value)}"
        for name, value in values.items()
    )
    return f"{_format_number(total)}<span class=\"submetric\">{details}</span>"


def _dashboard_for(
    summary: PolicySummary, dashboard_dir: Path | None
) -> Path | None:
    if dashboard_dir is None:
        return None
    exact_names = (
        f"{summary.arm}.html",
        f"{summary.arm}_aggregate.html",
    )
    for name in exact_names:
        candidate = dashboard_dir / name
        if candidate.is_file():
            return candidate
    patterns = (
        f"*_{summary.arm}_s*_aggregate.html",
        f"*_{summary.arm}_aggregate.html",
        f"*{summary.arm}*.html",
    )
    for pattern in patterns:
        matches = sorted(dashboard_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _relative_url(target: Path, output: Path) -> str:
    relative = os.path.relpath(target.resolve(), output.parent.resolve())
    return quote(relative.replace(os.sep, "/"), safe="/:")


def _comparison_figure(summaries: Sequence[PolicySummary]) -> go.Figure:
    labels = [summary.policy for summary in summaries]
    completions = [summary.completion_s for summary in summaries]
    makespans = [summary.makespan_s for summary in summaries]
    overhead_text = [
        "ideal unavailable"
        if summary.overhead_pct is None
        else f"{summary.overhead_pct:+.1f}% vs ideal"
        for summary in summaries
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=completions,
            y=labels,
            orientation="h",
            name="Training completion",
            marker_color="#3b82f6",
            customdata=overhead_text,
            hovertemplate=(
                "%{y}<br>completion %{x:,.1f} s<br>%{customdata}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=makespans,
            y=labels,
            mode="markers",
            name="Cluster makespan",
            marker={"color": "#f59e0b", "size": 10, "symbol": "diamond"},
            hovertemplate="%{y}<br>makespan %{x:,.1f} s<extra></extra>",
        )
    )
    ideal = next(
        (
            summary.completion_s
            for summary in summaries
            if summary.is_ideal
        ),
        None,
    )
    if ideal is not None:
        figure.add_vline(
            x=ideal,
            line_dash="dash",
            line_color="#10b981",
            annotation_text="ideal",
            annotation_position="top",
        )
    figure.update_layout(
        title="Completion and cluster makespan by policy",
        barmode="overlay",
        height=max(430, 58 * len(summaries) + 150),
        margin={"l": 130, "r": 30, "t": 90, "b": 70},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": "#172033"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={"title": "Seconds", "gridcolor": "#dbe3ee"},
        yaxis={"autorange": "reversed", "title": ""},
        hoverlabel={"bgcolor": "white"},
    )
    return figure


def _policy_rows(
    summaries: Sequence[PolicySummary],
    *,
    output: Path,
    dashboard_dir: Path | None,
) -> str:
    rows: list[str] = []
    for summary in summaries:
        dashboard = _dashboard_for(summary, dashboard_dir)
        if dashboard is None:
            policy = html.escape(summary.policy)
        else:
            url = html.escape(_relative_url(dashboard, output), quote=True)
            policy = (
                f'<a href="{url}" title="Open the per-policy trace dashboard">'
                f"{html.escape(summary.policy)}</a>"
            )
        if summary.censored_jobs_mean == 0:
            censored = "0"
        else:
            names = ", ".join(html.escape(name) for name in summary.censored_job_names)
            censored = (
                f"{_format_number(summary.censored_jobs_mean)} avg/run"
                f'<span class="submetric">{summary.censored_runs}/'
                f"{summary.run_count} runs · {names}</span>"
            )
        recovery = _format_counter(summary.recovery_sources)
        failures_restores = (
            f"{_format_number(summary.failure_count)} / "
            f"{_format_number(summary.restores)}"
            f'<span class="submetric">{_format_counter(summary.failures)}</span>'
        )
        pressure = (
            f"{_format_number(summary.aborted_flushes)} aborted"
            f'<span class="submetric">{_format_number(summary.fallback_flushes)} '
            f"local fallback; {_format_number(summary.partial_grants)} partial</span>"
        )
        rows.append(
            "<tr>"
            f'<th scope="row">{policy}<span class="submetric arm">'
            f"{html.escape(summary.arm)} · {summary.run_count} run"
            f'{"s" if summary.run_count != 1 else ""}</span></th>'
            f"<td>{_format_seconds(summary.completion_s)}</td>"
            f'<td class="number">{_format_pct(summary.overhead_pct, signed=True)}</td>'
            f'<td class="number">{_format_pct(summary.goodput_pct)}</td>'
            f"<td>{censored}</td>"
            f"<td>{failures_restores}</td>"
            f"<td>{recovery}</td>"
            f'<td class="number">{_format_number(summary.initial_state_recoveries) if summary.initial_state_recoveries is not None else "â€”"}</td>'
            f'<td class="number">{_format_pct(summary.durable_pct)}</td>'
            f'<td class="number">{_format_number(summary.flushes)}</td>'
            f"<td>{pressure}</td>"
            f'<td class="number">{_format_number(summary.lost_iters_max) if summary.lost_iters_max is not None else "â€”"}</td>'
            "</tr>"
        )
    return "".join(rows)


def generate_policy_matrix(
    result_paths: Sequence[Path],
    output: Path,
    *,
    dashboard_dir: Path | None = None,
    title: str | None = None,
    ideal_policy: str = "ideal",
) -> list[PolicySummary]:
    """Generate the HTML policy comparison and return its calculated rows."""
    scenario, rows = load_result_rows(result_paths)
    summaries = summarise_policies(rows, ideal_policy=ideal_policy)
    if not summaries:
        raise ValueError("result files contain no policy rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    if dashboard_dir is not None:
        dashboard_dir = dashboard_dir.resolve()
    heading = title or (
        f"Checkpoint policy matrix · {scenario}" if scenario else "Checkpoint policy matrix"
    )
    ideal_summary = next(
        (summary for summary in summaries if summary.is_ideal),
        None,
    )
    nonideal = [summary for summary in summaries if summary is not ideal_summary]
    best = max(
        (summary for summary in nonideal if summary.goodput_pct is not None),
        key=lambda summary: summary.goodput_pct or 0.0,
        default=None,
    )
    runs = sum(summary.run_count for summary in summaries)
    chart = pio.to_html(
        _comparison_figure(summaries),
        full_html=False,
        include_plotlyjs=True,
        config={"displaylogo": False, "responsive": True},
    )
    source_list = ", ".join(html.escape(path.name) for path in result_paths)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(heading)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#607087;
      --line:#dbe3ee; --panel:#fff; --wash:#f3f6fa; --blue:#2563eb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--wash); color:var(--ink);
      font:14px/1.45 Inter, "Segoe UI", Arial, sans-serif; }}
    main {{ width:min(1780px, calc(100% - 32px)); margin:28px auto 56px; }}
    h1 {{ margin:0 0 5px; font-size:clamp(24px,3vw,38px); letter-spacing:-.025em; }}
    .lede {{ color:var(--muted); margin:0 0 22px; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(150px,1fr));
      gap:12px; margin-bottom:16px; }}
    .card,.panel {{ background:var(--panel); border:1px solid var(--line);
      border-radius:12px; box-shadow:0 2px 10px rgba(23,32,51,.04); }}
    .card {{ padding:15px 17px; }}
    .card .label {{ color:var(--muted); font-size:12px; text-transform:uppercase;
      letter-spacing:.07em; }}
    .card .value {{ display:block; margin-top:5px; font-size:22px; font-weight:700; }}
    .panel {{ margin-top:16px; overflow:hidden; }}
    .chart {{ padding:8px 12px 0; }}
    .table-wrap {{ overflow:auto; }}
    table {{ width:100%; min-width:1420px; border-collapse:collapse; }}
    caption {{ padding:18px 18px 9px; text-align:left; font-size:18px; font-weight:700; }}
    thead th {{ position:sticky; top:0; z-index:1; background:#edf2f8;
      color:#445268; font-size:11px; letter-spacing:.045em; text-transform:uppercase;
      text-align:left; white-space:nowrap; }}
    th,td {{ padding:12px 13px; border-bottom:1px solid var(--line); vertical-align:top; }}
    tbody th {{ min-width:165px; text-align:left; }}
    tbody tr:hover {{ background:#f8fbff; }}
    tbody tr:last-child th,tbody tr:last-child td {{ border-bottom:0; }}
    a {{ color:var(--blue); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    .number {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
    .submetric {{ display:block; color:var(--muted); font-size:12px; margin-top:3px;
      white-space:normal; }}
    .submetric.arm {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
    .note {{ color:var(--muted); padding:4px 18px 18px; margin:0; }}
    footer {{ color:var(--muted); margin-top:14px; font-size:12px; }}
    @media(max-width:800px) {{ .cards {{ grid-template-columns:1fr 1fr; }}
      main {{ width:min(100% - 18px,1780px); margin-top:16px; }} }}
  </style>
</head>
<body><main>
  <h1>{html.escape(heading)}</h1>
  <p class="lede">Aggregate comparison across policy runs. Counts are means per run;
    overhead and goodput use the seed-matched ideal completion when available.</p>
  <section class="cards" aria-label="Experiment summary">
    <div class="card"><span class="label">Policies</span><span class="value">{len(summaries)}</span></div>
    <div class="card"><span class="label">Simulation runs</span><span class="value">{runs}</span></div>
    <div class="card"><span class="label">Ideal completion</span><span class="value">{_format_seconds(ideal_summary.completion_s if ideal_summary else None)}</span></div>
    <div class="card"><span class="label">Best non-ideal goodput</span><span class="value">{html.escape(best.policy) + " · " + _format_pct(best.goodput_pct) if best else "—"}</span></div>
  </section>
  <section class="panel chart">{chart}</section>
  <section class="panel table-wrap">
    <table>
      <caption>Aggregate policy metrics</caption>
      <thead><tr>
        <th>Strategy</th><th>Completion</th><th>Overhead vs ideal</th><th>Goodput</th>
        <th>Censored jobs</th><th>Failures / restores</th><th>Recovery sources</th>
        <th>Initial-state recoveries</th><th>Durable flushes</th>
        <th>Rank-weighted flushes</th><th>Flush pressure</th><th>Worst rollback (iterations)</th>
      </tr></thead>
      <tbody>{_policy_rows(summaries, output=output, dashboard_dir=dashboard_dir)}</tbody>
    </table>
    <p class="note">Completion is <code>train_end_s</code>. Durable is the mean durable
      fraction of completed rank flushes. Flush pressure reports aborted waves,
      zero-donor local fallbacks, and partial donor grants. A dash means the result did
      not report that metric; it is not silently treated as zero. Select a linked policy
      name to open its detailed aggregate trace dashboard.</p>
  </section>
  <footer>Sources: {source_list}</footer>
</main></body></html>"""
    output.write_text(page, encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standalone policy comparison from run_scenario JSON files."
    )
    parser.add_argument(
        "--result",
        type=Path,
        action="append",
        required=True,
        help="result JSON; repeat this option to merge independently-run policies",
    )
    parser.add_argument("--output", type=Path, required=True, help="HTML output path")
    parser.add_argument(
        "--dashboard-dir",
        type=Path,
        help=(
            "directory containing ARM.html, ARM_aggregate.html, or scenario-style "
            "*_ARM_sSEED_aggregate.html dashboards"
        ),
    )
    parser.add_argument("--title", help="page heading (defaults to the scenario name)")
    parser.add_argument(
        "--ideal-policy",
        default="ideal",
        help="arm/policy used as the ideal reference (default: ideal)",
    )
    args = parser.parse_args()
    summaries = generate_policy_matrix(
        args.result,
        args.output,
        dashboard_dir=args.dashboard_dir,
        title=args.title,
        ideal_policy=args.ideal_policy,
    )
    print(f"wrote {args.output} ({len(summaries)} policies)")


if __name__ == "__main__":
    main()
