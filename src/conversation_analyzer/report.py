"""HTML report generation with embedded Plotly charts."""

from pathlib import Path

import plotly.io as pio

from . import charts
from .stats import ConversationStats


def generate_report(stats: ConversationStats, output_path: str | Path) -> None:
    """Generate a self-contained HTML report with charts and stats."""
    # Each entry is (figure, optional data table HTML)
    chart_entries = [
        (charts.build_timeline(stats), None),
        (charts.build_talk_time_pie(stats), None),
        (charts.build_turn_duration_histogram(stats), _build_turn_duration_table(stats)),
        (charts.build_cumulative_talk_time(stats), None),
        (charts.build_response_time_histogram(stats), _build_response_time_table(stats)),
        (charts.build_yielding_latency_histogram(stats), _build_yielding_latency_table(stats)),
    ]

    # Build chart HTML divs
    chart_divs = []
    for i, (fig, data_table) in enumerate(chart_entries):
        div = pio.to_html(fig, full_html=False, include_plotlyjs=(i == 0))
        extra = data_table or ""
        chart_divs.append(f'<div class="chart">{div}{extra}</div>')

    charts_html = "\n".join(chart_divs)
    stats_html = _build_stats_table(stats)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conversation Analysis Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f8fafc; color: #1e293b; padding: 2rem; max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  .subtitle {{ color: #64748b; margin-bottom: 2rem; }}
  h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; color: #334155; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; font-weight: 600; font-size: 0.85rem; text-transform: uppercase;
       letter-spacing: 0.03em; color: #475569; }}
  td {{ font-size: 0.95rem; }}
  .metric-label {{ color: #64748b; }}
  .chart {{ margin-bottom: 1.5rem; background: white; border-radius: 8px; padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .speaker-a {{ color: #3b82f6; font-weight: 600; }}
  .speaker-b {{ color: #f97316; font-weight: 600; }}
  details {{ margin-top: 0.5rem; }}
  summary {{ cursor: pointer; color: #64748b; font-size: 0.85rem; padding: 0.4rem 0;
             user-select: none; }}
  summary:hover {{ color: #334155; }}
  .data-table {{ max-height: 400px; overflow-y: auto; margin-top: 0.5rem; }}
  .data-table table {{ font-size: 0.85rem; margin-bottom: 0; }}
  .data-table td, .data-table th {{ padding: 0.35rem 0.75rem; }}
</style>
</head>
<body>
<h1>Conversation Analysis Report</h1>
<p class="subtitle">Recording duration: {stats.duration_sec:.1f} seconds</p>

<h2>Summary Statistics</h2>
{stats_html}

<h2>Charts</h2>
{charts_html}
</body>
</html>"""

    Path(output_path).write_text(html)


def _build_stats_table(stats: ConversationStats) -> str:
    """Build HTML stats table with per-speaker metrics."""
    sa = stats.speaker_a
    sb = stats.speaker_b

    def fmt(v: float, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}"

    def row(label: str, val_a: str, val_b: str) -> str:
        return f"<tr><td class='metric-label'>{label}</td><td>{val_a}</td><td>{val_b}</td></tr>"

    rows = [
        row("Total talk time",
            f"{fmt(sa.total_talk_time)}s ({fmt(sa.talk_time_pct, 1)}%)",
            f"{fmt(sb.total_talk_time)}s ({fmt(sb.talk_time_pct, 1)}%)"),
        row("Number of turns", str(sa.num_turns), str(sb.num_turns)),
        row("Avg turn duration", f"{fmt(sa.avg_turn_duration)}s", f"{fmt(sb.avg_turn_duration)}s"),
        row("Median turn duration", f"{fmt(sa.median_turn_duration)}s", f"{fmt(sb.median_turn_duration)}s"),
        row("Min / Max turn",
            f"{fmt(sa.min_turn_duration)}s / {fmt(sa.max_turn_duration)}s",
            f"{fmt(sb.min_turn_duration)}s / {fmt(sb.max_turn_duration)}s"),
        row("Avg response time", f"{fmt(sa.avg_response_time)}s", f"{fmt(sb.avg_response_time)}s"),
        row("Median response time", f"{fmt(sa.median_response_time)}s", f"{fmt(sb.median_response_time)}s"),
        row("Std response time", f"{fmt(sa.std_response_time)}s", f"{fmt(sb.std_response_time)}s"),
        row("Min / Max response time",
            f"{fmt(sa.min_response_time)}s / {fmt(sa.max_response_time)}s",
            f"{fmt(sb.min_response_time)}s / {fmt(sb.max_response_time)}s"),
        row("Interruptions made", str(sa.interruptions_made), str(sb.interruptions_made)),
        row("Times interrupted", str(sa.times_interrupted), str(sb.times_interrupted)),
        row("Avg yielding latency", f"{fmt(sa.avg_yielding_latency)}s", f"{fmt(sb.avg_yielding_latency)}s"),
        row("Median yielding latency", f"{fmt(sa.median_yielding_latency)}s", f"{fmt(sb.median_yielding_latency)}s"),
    ]

    overlap_row = (
        f"<tr><td class='metric-label'>Total overlap</td>"
        f"<td colspan='2'>{fmt(stats.total_overlap_sec)}s ({fmt(stats.overlap_pct, 1)}%)</td></tr>"
    )
    silence_row = (
        f"<tr><td class='metric-label'>Total silence</td>"
        f"<td colspan='2'>{fmt(stats.total_silence_sec)}s ({fmt(stats.silence_pct, 1)}%)</td></tr>"
    )
    pause_row = (
        f"<tr><td class='metric-label'>Pauses</td>"
        f"<td colspan='2'>{stats.num_pauses} pauses, avg {fmt(stats.avg_pause_duration)}s, "
        f"longest {fmt(stats.longest_pause)}s</td></tr>"
    )

    return f"""<table>
<thead>
<tr><th>Metric</th><th class="speaker-a">{sa.label}</th><th class="speaker-b">{sb.label}</th></tr>
</thead>
<tbody>
{"".join(rows)}
{overlap_row}
{silence_row}
{pause_row}
</tbody>
</table>"""


def _build_turn_duration_table(stats: ConversationStats) -> str:
    """Expandable table of all turn durations."""
    rows = []
    for turn in stats.turns:
        sm, ss = divmod(turn.start, 60)
        em, es = divmod(turn.end, 60)
        rows.append(
            f"<tr><td>{turn.speaker}</td>"
            f"<td>{int(sm)}:{ss:05.2f}</td>"
            f"<td>{int(em)}:{es:05.2f}</td>"
            f"<td>{turn.duration:.2f}s</td></tr>"
        )
    if not rows:
        return ""
    return (
        f'<details><summary>View all turns ({len(rows)})</summary>'
        f'<div class="data-table"><table>'
        f"<thead><tr><th>Speaker</th><th>Start</th><th>End</th><th>Duration</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table></div></details>'
    )


def _build_response_time_table(stats: ConversationStats) -> str:
    """Expandable table of all response times with separate A→B and B→A columns."""
    sa = stats.speaker_a
    sb = stats.speaker_b
    # Rebuild per-transition data from turns
    entries: list[tuple[float, str, str, float]] = []
    for i in range(1, len(stats.turns)):
        prev = stats.turns[i - 1]
        curr = stats.turns[i]
        if prev.speaker == curr.speaker:
            continue
        gap = curr.start - prev.end
        if gap <= 0:
            continue
        entries.append((prev.end, prev.speaker, curr.speaker, gap))

    if not entries:
        return ""
    rows = []
    for time, prev_spk, resp_spk, gap in entries:
        a_to_b = f"{gap:.3f}s" if prev_spk == sa.label else ""
        b_to_a = f"{gap:.3f}s" if prev_spk == sb.label else ""
        mm, ss = divmod(time, 60)
        rows.append(
            f"<tr><td>{int(mm)}:{ss:05.2f}</td>"
            f"<td>{a_to_b}</td>"
            f"<td>{b_to_a}</td></tr>"
        )
    return (
        f'<details><summary>View all response times ({len(rows)})</summary>'
        f'<div class="data-table"><table>'
        f"<thead><tr><th>At</th>"
        f'<th class="speaker-a">{sa.label} \u2192 {sb.label}</th>'
        f'<th class="speaker-b">{sb.label} \u2192 {sa.label}</th>'
        f"</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table></div></details>'
    )


def _build_yielding_latency_table(stats: ConversationStats) -> str:
    """Expandable table of yielding latencies when speaker A interrupts speaker B.

    Only includes interruptions where B had been speaking for >= 4 seconds.
    """
    sa_label = stats.speaker_a.label
    sb_label = stats.speaker_b.label
    rows = []
    for intr in stats.interruptions:
        if (intr.interrupter != sa_label or intr.speech_before < 4.0
                or intr.interrupter_duration < 2.0 or not intr.yielded):
            continue
        mm, ss = divmod(intr.start_time, 60)
        rows.append(
            f"<tr><td>{int(mm)}:{ss:05.2f}</td>"
            f"<td>{intr.speech_before:.1f}s</td>"
            f"<td>{intr.yielding_latency:.3f}s</td></tr>"
        )
    if not rows:
        return ""
    return (
        f'<details><summary>View all {sb_label} yields ({len(rows)})</summary>'
        f'<div class="data-table"><table>'
        f"<thead><tr><th>At</th><th>Speaking Before</th><th>Yielding Latency</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table></div></details>'
    )
