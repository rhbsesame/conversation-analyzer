"""Tests for stats.py — turn-taking, latency, interruption detection."""

from conversation_analyzer.stats import (
    ConversationStats,
    Interruption,
    Turn,
    compute_stats,
    _build_turns,
    _compute_overlap,
    _detect_interruptions,
)
from conversation_analyzer.vad import SpeechSegment


class TestBuildTurns:
    def test_simple_alternating(self):
        segs_a = [SpeechSegment(0.0, 2.0)]
        segs_b = [SpeechSegment(3.0, 5.0)]
        turns = _build_turns(segs_a, segs_b, "A", "B")
        assert len(turns) == 2
        assert turns[0].speaker == "A"
        assert turns[1].speaker == "B"

    def test_consecutive_same_speaker_merged(self):
        segs_a = [SpeechSegment(0.0, 2.0), SpeechSegment(2.5, 4.0)]
        segs_b = []
        turns = _build_turns(segs_a, segs_b, "A", "B")
        # Both A segments should merge into one turn
        assert len(turns) == 1
        assert turns[0].speaker == "A"
        assert turns[0].end == 4.0

    def test_overlapping_creates_separate_turns(self):
        segs_a = [SpeechSegment(0.0, 3.0)]
        segs_b = [SpeechSegment(2.0, 5.0)]
        turns = _build_turns(segs_a, segs_b, "A", "B")
        assert len(turns) == 2
        assert turns[0].speaker == "A"
        assert turns[1].speaker == "B"

    def test_empty(self):
        turns = _build_turns([], [], "A", "B")
        assert turns == []


class TestDetectInterruptions:
    def test_b_interrupts_a(self):
        segs_a = [SpeechSegment(0.0, 5.0)]
        segs_b = [SpeechSegment(3.0, 7.0)]
        interruptions = _detect_interruptions(segs_a, segs_b, "A", "B")
        assert len(interruptions) == 1
        assert interruptions[0].interrupter == "B"
        assert interruptions[0].interrupted == "A"
        assert interruptions[0].start_time == 3.0
        assert interruptions[0].yielding_latency == 2.0  # A stops at 5.0, B started at 3.0

    def test_a_interrupts_b(self):
        segs_a = [SpeechSegment(2.0, 6.0)]
        segs_b = [SpeechSegment(0.0, 4.0)]
        interruptions = _detect_interruptions(segs_a, segs_b, "A", "B")
        assert len(interruptions) == 1
        assert interruptions[0].interrupter == "A"
        assert interruptions[0].interrupted == "B"
        assert interruptions[0].yielding_latency == 2.0  # B stops at 4.0, A started at 2.0

    def test_no_interruptions(self):
        segs_a = [SpeechSegment(0.0, 2.0)]
        segs_b = [SpeechSegment(3.0, 5.0)]
        interruptions = _detect_interruptions(segs_a, segs_b, "A", "B")
        assert len(interruptions) == 0

    def test_mutual_interruption(self):
        # Both speakers overlap: A 0-5, B 3-8, then A 6-10
        segs_a = [SpeechSegment(0.0, 5.0), SpeechSegment(6.0, 10.0)]
        segs_b = [SpeechSegment(3.0, 8.0)]
        interruptions = _detect_interruptions(segs_a, segs_b, "A", "B")
        # B interrupts A (B starts at 3.0 during A's 0-5)
        # A interrupts B (A starts at 6.0 during B's 3-8)
        assert len(interruptions) == 2
        interrupters = {i.interrupter for i in interruptions}
        assert interrupters == {"A", "B"}


class TestComputeOverlap:
    def test_no_overlap(self):
        segs_a = [SpeechSegment(0.0, 2.0)]
        segs_b = [SpeechSegment(3.0, 5.0)]
        assert _compute_overlap(segs_a, segs_b) == 0.0

    def test_partial_overlap(self):
        segs_a = [SpeechSegment(0.0, 3.0)]
        segs_b = [SpeechSegment(2.0, 5.0)]
        assert abs(_compute_overlap(segs_a, segs_b) - 1.0) < 1e-9

    def test_full_containment(self):
        segs_a = [SpeechSegment(0.0, 10.0)]
        segs_b = [SpeechSegment(2.0, 5.0)]
        assert abs(_compute_overlap(segs_a, segs_b) - 3.0) < 1e-9


class TestComputeStats:
    def test_basic_conversation(self):
        # A speaks 0-2, B speaks 3-5, A speaks 6-8
        segs_a = [SpeechSegment(0.0, 2.0), SpeechSegment(6.0, 8.0)]
        segs_b = [SpeechSegment(3.0, 5.0)]
        duration = 10.0

        stats = compute_stats(segs_a, segs_b, duration, "A", "B")

        assert stats.duration_sec == 10.0
        assert abs(stats.speaker_a.total_talk_time - 4.0) < 1e-9
        assert abs(stats.speaker_b.total_talk_time - 2.0) < 1e-9
        assert stats.speaker_a.num_turns == 2
        assert stats.speaker_b.num_turns == 1
        assert stats.total_overlap_sec == 0.0
        assert len(stats.interruptions) == 0

    def test_with_overlap(self):
        segs_a = [SpeechSegment(0.0, 4.0)]
        segs_b = [SpeechSegment(3.0, 6.0)]
        duration = 6.0

        stats = compute_stats(segs_a, segs_b, duration, "A", "B")
        assert abs(stats.total_overlap_sec - 1.0) < 1e-9

    def test_response_times_clean_transition(self):
        # A speaks 0-2, then B speaks 4-6 (2s gap = B's response time)
        segs_a = [SpeechSegment(0.0, 2.0)]
        segs_b = [SpeechSegment(4.0, 6.0)]
        duration = 6.0

        stats = compute_stats(segs_a, segs_b, duration, "A", "B")
        assert len(stats.speaker_b.response_times) == 1
        assert abs(stats.speaker_b.response_times[0] - 2.0) < 1e-9

    def test_response_times_excludes_interruptions(self):
        # A speaks 0-4, B interrupts at 3 (overlap) — no response time recorded
        segs_a = [SpeechSegment(0.0, 4.0)]
        segs_b = [SpeechSegment(3.0, 6.0)]
        duration = 6.0

        stats = compute_stats(segs_a, segs_b, duration, "A", "B")
        assert len(stats.speaker_b.response_times) == 0
        assert len(stats.speaker_a.response_times) == 0

    def test_turn_properties(self):
        t = Turn(speaker="A", start=1.0, end=3.5)
        assert t.duration == 2.5

    def test_empty_segments(self):
        stats = compute_stats([], [], 10.0, "A", "B")
        assert stats.speaker_a.total_talk_time == 0.0
        assert stats.speaker_b.total_talk_time == 0.0
        assert stats.total_silence_sec == 10.0
