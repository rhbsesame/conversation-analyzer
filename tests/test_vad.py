"""Tests for vad.py — voice activity detection with synthetic signals."""

import numpy as np

from conversation_analyzer.vad import SpeechSegment, detect_speech, _frames_to_segments


class TestFramesToSegments:
    def test_empty(self):
        assert _frames_to_segments(np.array([], dtype=bool), 10, 1) == []

    def test_all_silence(self):
        is_speech = np.zeros(100, dtype=bool)
        assert _frames_to_segments(is_speech, 10, 1) == []

    def test_all_speech(self):
        is_speech = np.ones(100, dtype=bool)
        result = _frames_to_segments(is_speech, 10, 1)
        assert len(result) == 1
        assert result[0] == (0, 100)

    def test_merge_small_gap(self):
        # Two speech regions separated by a gap of 3 frames (< min_silence=5)
        is_speech = np.zeros(30, dtype=bool)
        is_speech[0:10] = True
        is_speech[13:25] = True
        result = _frames_to_segments(is_speech, min_silence_frames=5, min_speech_frames=1)
        assert len(result) == 1
        assert result[0] == (0, 25)

    def test_no_merge_large_gap(self):
        is_speech = np.zeros(30, dtype=bool)
        is_speech[0:10] = True
        is_speech[20:30] = True
        result = _frames_to_segments(is_speech, min_silence_frames=5, min_speech_frames=1)
        assert len(result) == 2

    def test_drop_short_segments(self):
        is_speech = np.zeros(30, dtype=bool)
        is_speech[0:2] = True  # Too short (< 5 frames)
        is_speech[20:30] = True  # Long enough
        result = _frames_to_segments(is_speech, min_silence_frames=3, min_speech_frames=5)
        assert len(result) == 1
        assert result[0] == (20, 30)


class TestDetectSpeech:
    def test_pure_silence(self):
        signal = np.zeros(16000)  # 1 second of silence
        segments = detect_speech(signal, 16000, frame_ms=30, threshold=0.01)
        assert segments == []

    def test_pure_tone(self):
        sr = 16000
        t = np.arange(sr * 2) / sr  # 2 seconds
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        segments = detect_speech(signal, sr, frame_ms=30, threshold=0.01, min_speech_ms=100)
        assert len(segments) >= 1
        total_speech = sum(s.duration_sec for s in segments)
        # Should detect most of the signal as speech
        assert total_speech > 1.5

    def test_alternating_tone_and_silence(self):
        sr = 16000
        # 1 sec tone, 1 sec silence, 1 sec tone
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        silence = np.zeros(sr)
        signal = np.concatenate([tone, silence, tone])

        segments = detect_speech(signal, sr, frame_ms=30, threshold=0.01,
                                 min_speech_ms=200, min_silence_ms=300)
        assert len(segments) == 2
        # First segment should be around 0-1s
        assert segments[0].start_sec < 0.1
        assert abs(segments[0].end_sec - 1.0) < 0.15
        # Second segment should be around 2-3s
        assert abs(segments[1].start_sec - 2.0) < 0.15

    def test_speech_segment_properties(self):
        seg = SpeechSegment(start_sec=1.0, end_sec=3.5)
        assert seg.duration_sec == 2.5

    def test_short_signal(self):
        # Signal shorter than one frame
        signal = np.zeros(10)
        segments = detect_speech(signal, 16000, frame_ms=30)
        assert segments == []
