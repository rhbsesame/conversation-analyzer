"""Plotly chart builders for conversation analysis."""

import plotly.graph_objects as go

from .stats import ConversationStats

COLOR_A = "#3b82f6"  # blue
COLOR_B = "#f97316"  # orange
COLOR_SILENCE = "#d1d5db"  # gray


def build_timeline(stats: ConversationStats) -> go.Figure:
    """Horizontal bars showing speech activity for both speakers across the recording."""
    fig = go.Figure()

    label_a = stats.speaker_a.label
    label_b = stats.speaker_b.label

    for turn in stats.turns:
        color = COLOR_A if turn.speaker == label_a else COLOR_B
        y_val = label_a if turn.speaker == label_a else label_b
        fig.add_trace(go.Bar(
            x=[turn.duration],
            y=[y_val],
            base=[turn.start],
            orientation="h",
            marker_color=color,
            name=turn.speaker,
            showlegend=False,
            hovertemplate=f"{turn.speaker}<br>"
                          f"Start: %{{base:.2f}}s<br>"
                          f"Duration: %{{x:.2f}}s<extra></extra>",
        ))

    fig.update_layout(
        title="Speech Timeline",
        xaxis_title="Time (seconds)",
        barmode="stack",
        height=250,
        margin=dict(l=100, r=20, t=50, b=40),
        yaxis=dict(categoryorder="array", categoryarray=[label_b, label_a]),
    )
    return fig


def build_talk_time_pie(stats: ConversationStats) -> go.Figure:
    """Pie chart: Speaker A vs Speaker B vs Silence."""
    labels = [stats.speaker_a.label, stats.speaker_b.label, "Silence"]
    values = [
        stats.speaker_a.total_talk_time,
        stats.speaker_b.total_talk_time,
        stats.total_silence_sec,
    ]
    colors = [COLOR_A, COLOR_B, COLOR_SILENCE]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.1f}s (%{percent})<extra></extra>",
    )])
    fig.update_layout(title="Talk Time Distribution", height=400)
    return fig


def build_turn_duration_histogram(stats: ConversationStats) -> go.Figure:
    """Overlaid histograms of turn lengths, one per speaker."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=stats.speaker_a.turn_durations,
        name=stats.speaker_a.label,
        marker_color=COLOR_A,
        opacity=0.7,
        xbins=dict(size=0.25),
    ))
    fig.add_trace(go.Histogram(
        x=stats.speaker_b.turn_durations,
        name=stats.speaker_b.label,
        marker_color=COLOR_B,
        opacity=0.7,
        xbins=dict(size=0.25),
    ))
    fig.update_layout(
        title="Turn Duration Distribution",
        xaxis_title="Duration (seconds)",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
    )
    return fig


def build_cumulative_talk_time(stats: ConversationStats) -> go.Figure:
    """Line chart of accumulated speaking time over the recording, per speaker."""
    label_a = stats.speaker_a.label
    label_b = stats.speaker_b.label

    # Build time series for each speaker
    times_a, cum_a = _cumulative_series(
        [(t.start, t.duration) for t in stats.turns if t.speaker == label_a]
    )
    times_b, cum_b = _cumulative_series(
        [(t.start, t.duration) for t in stats.turns if t.speaker == label_b]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times_a, y=cum_a, mode="lines", name=label_a,
        line=dict(color=COLOR_A, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=times_b, y=cum_b, mode="lines", name=label_b,
        line=dict(color=COLOR_B, width=2),
    ))
    fig.update_layout(
        title="Cumulative Talk Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Cumulative Talk Time (seconds)",
        height=400,
    )
    return fig


def build_response_time_histogram(stats: ConversationStats) -> go.Figure:
    """Distribution of turn-taking latencies, by direction.

    Rebuilds data from turns (same logic as the response time table)
    to guarantee the histogram and table always show identical data.
    """
    sa_label = stats.speaker_a.label
    sb_label = stats.speaker_b.label
    a_to_b: list[float] = []
    b_to_a: list[float] = []
    for i in range(1, len(stats.turns)):
        prev = stats.turns[i - 1]
        curr = stats.turns[i]
        if prev.speaker == curr.speaker:
            continue
        gap = curr.start - prev.end
        if gap <= 0:
            continue
        if prev.speaker == sa_label:
            a_to_b.append(gap)
        else:
            b_to_a.append(gap)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=a_to_b,
        name=f"{sa_label} \u2192 {sb_label}",
        marker_color=COLOR_A,
        opacity=0.7,
        xbins=dict(size=0.25),
    ))
    fig.add_trace(go.Histogram(
        x=b_to_a,
        name=f"{sb_label} \u2192 {sa_label}",
        marker_color=COLOR_B,
        opacity=0.7,
        xbins=dict(size=0.25),
    ))
    fig.update_layout(
        title="Turn-Taking Latency (Response Time)",
        xaxis_title="Latency (seconds)",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
    )
    return fig


def build_yielding_latency_histogram(stats: ConversationStats) -> go.Figure:
    """Distribution of yielding times when speaker B is interrupted by speaker A.

    Only includes interruptions where B had been speaking for >= 4 seconds.
    """
    sa_label = stats.speaker_a.label
    sb_label = stats.speaker_b.label
    latencies = [
        intr.yielding_latency
        for intr in stats.interruptions
        if intr.interrupter == sa_label and intr.speech_before >= 4.0
        and intr.interrupter_duration >= 3.0
    ]

    fig = go.Figure()
    if latencies:
        fig.add_trace(go.Histogram(
            x=latencies,
            name=f"{sb_label} yielding",
            marker_color=COLOR_B,
            opacity=0.7,
            xbins=dict(size=0.25),
        ))
    fig.update_layout(
        title=f"{sb_label} Yielding Latency (when {sa_label} interrupts)",
        xaxis_title="Yielding Latency (seconds)",
        yaxis_title="Count",
        height=400,
    )
    return fig


def _cumulative_series(
    turn_data: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Build cumulative time series from list of (start_time, duration)."""
    if not turn_data:
        return [0.0], [0.0]

    times = [0.0]
    cumulative = [0.0]
    total = 0.0

    for start, duration in sorted(turn_data):
        times.append(start)
        cumulative.append(total)
        total += duration
        times.append(start + duration)
        cumulative.append(total)

    return times, cumulative
