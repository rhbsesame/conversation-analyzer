"""Turn-taking, silence, latency, and interruption statistics."""

from dataclasses import dataclass, field

import numpy as np

from .vad import SpeechSegment


@dataclass
class Turn:
    """A speaking turn by one speaker."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Interruption:
    """An interruption event where one speaker starts while the other is still speaking."""

    interrupter: str
    interrupted: str
    start_time: float
    yielding_latency: float  # time until interrupted speaker stops


@dataclass
class SpeakerStats:
    """Statistics for a single speaker."""

    label: str
    total_talk_time: float = 0.0
    talk_time_pct: float = 0.0
    num_turns: int = 0
    turn_durations: list[float] = field(default_factory=list)
    response_times: list[float] = field(default_factory=list)
    interruptions_made: int = 0
    times_interrupted: int = 0
    yielding_latencies: list[float] = field(default_factory=list)

    @property
    def avg_turn_duration(self) -> float:
        return float(np.mean(self.turn_durations)) if self.turn_durations else 0.0

    @property
    def median_turn_duration(self) -> float:
        return float(np.median(self.turn_durations)) if self.turn_durations else 0.0

    @property
    def min_turn_duration(self) -> float:
        return float(np.min(self.turn_durations)) if self.turn_durations else 0.0

    @property
    def max_turn_duration(self) -> float:
        return float(np.max(self.turn_durations)) if self.turn_durations else 0.0

    @property
    def avg_response_time(self) -> float:
        return float(np.mean(self.response_times)) if self.response_times else 0.0

    @property
    def median_response_time(self) -> float:
        return float(np.median(self.response_times)) if self.response_times else 0.0

    @property
    def std_response_time(self) -> float:
        return float(np.std(self.response_times)) if self.response_times else 0.0

    @property
    def min_response_time(self) -> float:
        return float(np.min(self.response_times)) if self.response_times else 0.0

    @property
    def max_response_time(self) -> float:
        return float(np.max(self.response_times)) if self.response_times else 0.0

    @property
    def avg_yielding_latency(self) -> float:
        return float(np.mean(self.yielding_latencies)) if self.yielding_latencies else 0.0

    @property
    def median_yielding_latency(self) -> float:
        return float(np.median(self.yielding_latencies)) if self.yielding_latencies else 0.0

    @property
    def std_yielding_latency(self) -> float:
        return float(np.std(self.yielding_latencies)) if self.yielding_latencies else 0.0

    @property
    def min_yielding_latency(self) -> float:
        return float(np.min(self.yielding_latencies)) if self.yielding_latencies else 0.0

    @property
    def max_yielding_latency(self) -> float:
        return float(np.max(self.yielding_latencies)) if self.yielding_latencies else 0.0


@dataclass
class ConversationStats:
    """All computed conversation metrics."""

    duration_sec: float
    speaker_a: SpeakerStats
    speaker_b: SpeakerStats
    turns: list[Turn]
    interruptions: list[Interruption]
    total_overlap_sec: float = 0.0
    overlap_pct: float = 0.0
    total_silence_sec: float = 0.0
    silence_pct: float = 0.0
    num_pauses: int = 0
    avg_pause_duration: float = 0.0
    longest_pause: float = 0.0


def compute_stats(
    segments_a: list[SpeechSegment],
    segments_b: list[SpeechSegment],
    duration_sec: float,
    label_a: str = "Speaker A",
    label_b: str = "Speaker B",
) -> ConversationStats:
    """Compute all conversation statistics from detected speech segments."""
    speaker_a = SpeakerStats(label=label_a)
    speaker_b = SpeakerStats(label=label_b)

    # Talk time
    speaker_a.total_talk_time = sum(s.duration_sec for s in segments_a)
    speaker_b.total_talk_time = sum(s.duration_sec for s in segments_b)
    if duration_sec > 0:
        speaker_a.talk_time_pct = speaker_a.total_talk_time / duration_sec * 100
        speaker_b.talk_time_pct = speaker_b.total_talk_time / duration_sec * 100

    # Build turns from merged timeline
    turns = _build_turns(segments_a, segments_b, label_a, label_b)
    speaker_a.num_turns = sum(1 for t in turns if t.speaker == label_a)
    speaker_b.num_turns = sum(1 for t in turns if t.speaker == label_b)
    speaker_a.turn_durations = [t.duration for t in turns if t.speaker == label_a]
    speaker_b.turn_durations = [t.duration for t in turns if t.speaker == label_b]

    # Turn-taking latency (response time)
    _compute_response_times(turns, speaker_a, speaker_b, label_a, label_b)

    # Interruption analysis
    interruptions = _detect_interruptions(segments_a, segments_b, label_a, label_b)
    for intr in interruptions:
        if intr.interrupter == label_a:
            speaker_a.interruptions_made += 1
            speaker_b.times_interrupted += 1
            speaker_b.yielding_latencies.append(intr.yielding_latency)
        else:
            speaker_b.interruptions_made += 1
            speaker_a.times_interrupted += 1
            speaker_a.yielding_latencies.append(intr.yielding_latency)

    # Overlap
    total_overlap = _compute_overlap(segments_a, segments_b)
    overlap_pct = total_overlap / duration_sec * 100 if duration_sec > 0 else 0.0

    # Silence/pause analysis
    silence_info = _compute_silence(segments_a, segments_b, duration_sec)

    return ConversationStats(
        duration_sec=duration_sec,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        turns=turns,
        interruptions=interruptions,
        total_overlap_sec=total_overlap,
        overlap_pct=overlap_pct,
        total_silence_sec=silence_info["total"],
        silence_pct=silence_info["total"] / duration_sec * 100 if duration_sec > 0 else 0.0,
        num_pauses=silence_info["count"],
        avg_pause_duration=silence_info["avg"],
        longest_pause=silence_info["longest"],
    )


def _build_turns(
    segments_a: list[SpeechSegment],
    segments_b: list[SpeechSegment],
    label_a: str,
    label_b: str,
) -> list[Turn]:
    """Build a timeline of speaking turns from both speakers' segments."""
    events: list[tuple[float, float, str]] = []
    for seg in segments_a:
        events.append((seg.start_sec, seg.end_sec, label_a))
    for seg in segments_b:
        events.append((seg.start_sec, seg.end_sec, label_b))

    events.sort(key=lambda e: e[0])

    turns: list[Turn] = []
    for start, end, speaker in events:
        if turns and turns[-1].speaker == speaker:
            # Extend current turn if same speaker
            turns[-1] = Turn(speaker=speaker, start=turns[-1].start, end=max(turns[-1].end, end))
        else:
            turns.append(Turn(speaker=speaker, start=start, end=end))

    return turns


def _compute_response_times(
    turns: list[Turn],
    speaker_a: SpeakerStats,
    speaker_b: SpeakerStats,
    label_a: str,
    label_b: str,
) -> None:
    """Compute turn-taking latency for clean (non-interruption) speaker transitions.

    Only counts transitions where the new speaker starts after the previous
    speaker finishes (positive gap). Overlapping transitions are interruptions
    and tracked separately.
    """
    for i in range(1, len(turns)):
        prev = turns[i - 1]
        curr = turns[i]
        if prev.speaker == curr.speaker:
            continue
        gap = curr.start - prev.end
        if gap <= 0:
            continue  # overlap/zero-gap = interruption or boundary artifact
        if curr.speaker == label_a:
            speaker_a.response_times.append(gap)
        else:
            speaker_b.response_times.append(gap)


def _detect_interruptions(
    segments_a: list[SpeechSegment],
    segments_b: list[SpeechSegment],
    label_a: str,
    label_b: str,
) -> list[Interruption]:
    """Detect interruptions: one speaker starts while the other is still speaking."""
    interruptions: list[Interruption] = []

    # Check B interrupting A
    for seg_b in segments_b:
        for seg_a in segments_a:
            if seg_b.start_sec > seg_a.start_sec and seg_b.start_sec < seg_a.end_sec:
                yielding_latency = seg_a.end_sec - seg_b.start_sec
                interruptions.append(Interruption(
                    interrupter=label_b,
                    interrupted=label_a,
                    start_time=seg_b.start_sec,
                    yielding_latency=yielding_latency,
                ))
                break  # only count one interruption per B segment

    # Check A interrupting B
    for seg_a in segments_a:
        for seg_b in segments_b:
            if seg_a.start_sec > seg_b.start_sec and seg_a.start_sec < seg_b.end_sec:
                yielding_latency = seg_b.end_sec - seg_a.start_sec
                interruptions.append(Interruption(
                    interrupter=label_a,
                    interrupted=label_b,
                    start_time=seg_a.start_sec,
                    yielding_latency=yielding_latency,
                ))
                break

    interruptions.sort(key=lambda x: x.start_time)
    return interruptions


def _compute_overlap(
    segments_a: list[SpeechSegment],
    segments_b: list[SpeechSegment],
) -> float:
    """Compute total overlap time where both speakers are speaking simultaneously."""
    total = 0.0
    for sa in segments_a:
        for sb in segments_b:
            overlap_start = max(sa.start_sec, sb.start_sec)
            overlap_end = min(sa.end_sec, sb.end_sec)
            if overlap_end > overlap_start:
                total += overlap_end - overlap_start
    return total


def _compute_silence(
    segments_a: list[SpeechSegment],
    segments_b: list[SpeechSegment],
    duration_sec: float,
) -> dict:
    """Compute silence statistics (periods where neither speaker is active)."""
    # Merge all speech into a single sorted list of intervals
    all_intervals = [(s.start_sec, s.end_sec) for s in segments_a] + [
        (s.start_sec, s.end_sec) for s in segments_b
    ]
    all_intervals.sort()

    if not all_intervals:
        return {"total": duration_sec, "count": 1 if duration_sec > 0 else 0,
                "avg": duration_sec, "longest": duration_sec}

    # Merge overlapping intervals
    merged = [all_intervals[0]]
    for start, end in all_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Compute gaps
    pauses: list[float] = []
    if merged[0][0] > 0:
        pauses.append(merged[0][0])
    for i in range(1, len(merged)):
        gap = merged[i][0] - merged[i - 1][1]
        if gap > 0:
            pauses.append(gap)
    if merged[-1][1] < duration_sec:
        pauses.append(duration_sec - merged[-1][1])

    total_silence = sum(pauses)
    return {
        "total": total_silence,
        "count": len(pauses),
        "avg": total_silence / len(pauses) if pauses else 0.0,
        "longest": max(pauses) if pauses else 0.0,
    }
