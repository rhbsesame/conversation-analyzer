"""Voice activity detection using Silero VAD."""

from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from .audio import resample

_model = None


def _get_model():
    """Return the cached Silero VAD model (loaded once)."""
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


@dataclass
class SpeechSegment:
    """A contiguous segment of detected speech."""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def detect_speech(
    signal: np.ndarray,
    sample_rate: int,
    min_speech_ms: int = 200,
    min_silence_ms: int = 300,
) -> list[SpeechSegment]:
    """Detect speech segments in a single-channel signal using Silero VAD.

    Args:
        signal: 1-D float64 audio signal normalized to [-1, 1].
        sample_rate: Sample rate in Hz.
        min_speech_ms: Minimum speech segment duration to keep (ms).
        min_silence_ms: Minimum silence gap to split segments (ms).

    Returns:
        List of SpeechSegment with start/end times in seconds.
    """
    if len(signal) == 0:
        return []

    # Silero requires 16kHz (or 8kHz); resample if needed
    target_sr = 16000
    resampled = resample(signal, sample_rate, target_sr)

    if len(resampled) < 512:
        return []

    # Convert to torch tensor (float32)
    wav_tensor = torch.from_numpy(resampled.astype(np.float32))

    model = _get_model()
    timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        sampling_rate=target_sr,
        return_seconds=True,
    )

    segments = [
        SpeechSegment(start_sec=ts["start"], end_sec=ts["end"])
        for ts in timestamps
    ]

    return _merge_and_filter(segments, min_speech_ms, min_silence_ms)


def _merge_and_filter(
    segments: list[SpeechSegment],
    min_speech_ms: int,
    min_silence_ms: int,
) -> list[SpeechSegment]:
    """Merge segments separated by short gaps and drop short segments.

    Args:
        segments: Speech segments from Silero (already in seconds).
        min_speech_ms: Drop segments shorter than this (ms).
        min_silence_ms: Merge segments separated by gaps shorter than this (ms).
    """
    if not segments:
        return []

    min_silence_sec = min_silence_ms / 1000.0
    min_speech_sec = min_speech_ms / 1000.0

    # Merge segments separated by gaps < min_silence_sec
    merged: list[SpeechSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.start_sec - prev.end_sec < min_silence_sec:
            merged[-1] = SpeechSegment(prev.start_sec, seg.end_sec)
        else:
            merged.append(seg)

    # Drop segments shorter than min_speech_sec
    return [s for s in merged if s.duration_sec >= min_speech_sec]
