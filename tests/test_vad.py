"""Tests for vad.py — voice activity detection."""

import numpy as np

from conversation_analyzer.vad import SpeechSegment, detect_speech, _merge_and_filter


class TestMergeAndFilter:
    def test_empty(self):
        assert _merge_and_filter([], 200, 300) == []

    def test_no_merge_large_gap(self):
        segments = [
            SpeechSegment(0.0, 1.0),
            SpeechSegment(2.0, 3.0),
        ]
        result = _merge_and_filter(segments, min_speech_ms=200, min_silence_ms=300)
        assert len(result) == 2

    def test_merge_small_gap(self):
        segments = [
            SpeechSegment(0.0, 1.0),
            SpeechSegment(1.1, 2.0),  # gap = 0.1s < 0.3s
        ]
        result = _merge_and_filter(segments, min_speech_ms=200, min_silence_ms=300)
        assert len(result) == 1
        assert result[0].start_sec == 0.0
        assert result[0].end_sec == 2.0

    def test_drop_short_segments(self):
        segments = [
            SpeechSegment(0.0, 0.05),  # 50ms — too short
            SpeechSegment(1.0, 2.0),   # 1s — long enough
        ]
        result = _merge_and_filter(segments, min_speech_ms=200, min_silence_ms=300)
        assert len(result) == 1
        assert result[0].start_sec == 1.0

    def test_merge_then_filter(self):
        # Two short segments close together — merge makes them long enough
        segments = [
            SpeechSegment(0.0, 0.15),   # 150ms
            SpeechSegment(0.2, 0.45),   # 250ms, gap = 50ms
        ]
        result = _merge_and_filter(segments, min_speech_ms=200, min_silence_ms=300)
        assert len(result) == 1
        assert result[0].start_sec == 0.0
        assert result[0].end_sec == 0.45


class TestDetectSpeech:
    def test_pure_silence(self):
        signal = np.zeros(16000)  # 1 second of silence
        segments = detect_speech(signal, 16000)
        assert segments == []

    def test_pure_tone_not_detected_as_speech(self):
        """Silero correctly rejects a pure sine wave as non-speech."""
        sr = 16000
        t = np.arange(sr * 2) / sr  # 2 seconds
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        segments = detect_speech(signal, sr, min_speech_ms=100)
        assert segments == []

    def test_silence_between_tones(self):
        """Pure tones with silence — Silero sees no speech, returns empty."""
        sr = 16000
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        silence = np.zeros(sr)
        signal = np.concatenate([tone, silence, tone])

        segments = detect_speech(signal, sr, min_speech_ms=200, min_silence_ms=300)
        assert segments == []

    def test_speech_segment_properties(self):
        seg = SpeechSegment(start_sec=1.0, end_sec=3.5)
        assert seg.duration_sec == 2.5

    def test_short_signal(self):
        # Signal too short for Silero to process
        signal = np.zeros(10)
        segments = detect_speech(signal, 16000)
        assert segments == []

    def test_empty_signal(self):
        signal = np.array([], dtype=np.float64)
        segments = detect_speech(signal, 16000)
        assert segments == []

    def test_resampling_runs_without_error(self):
        """Signals at non-16kHz rates are resampled internally without error."""
        sr = 44100
        t = np.arange(sr * 2) / sr  # 2 seconds
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        # Should not raise — resampling from 44.1kHz to 16kHz works
        segments = detect_speech(signal, sr, min_speech_ms=100)
        assert isinstance(segments, list)
