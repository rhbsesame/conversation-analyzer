"""Energy-based voice activity detection."""

from dataclasses import dataclass

import numpy as np


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
    frame_ms: int = 30,
    threshold: float | None = None,
    min_speech_ms: int = 200,
    min_silence_ms: int = 300,
) -> list[SpeechSegment]:
    """Detect speech segments in a single-channel signal.

    Args:
        signal: 1-D float64 audio signal normalized to [-1, 1].
        sample_rate: Sample rate in Hz.
        frame_ms: Frame size in milliseconds for RMS computation.
        threshold: RMS energy threshold. If None, auto-detect.
        min_speech_ms: Minimum speech segment duration to keep (ms).
        min_silence_ms: Minimum silence gap to split segments (ms).

    Returns:
        List of SpeechSegment with start/end times in seconds.
    """
    frame_samples = int(sample_rate * frame_ms / 1000)
    if frame_samples == 0:
        return []

    # Compute RMS energy per frame
    n_frames = len(signal) // frame_samples
    if n_frames == 0:
        return []

    frames = signal[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    rms = np.sqrt(np.mean(frames**2, axis=1))

    # Auto-detect threshold
    if threshold is None:
        threshold = _auto_threshold(rms)

    # Mark frames as speech/silence
    is_speech = rms > threshold

    # Convert to segments, merging gaps < min_silence_ms
    min_silence_frames = int(min_silence_ms / frame_ms)
    min_speech_frames = int(min_speech_ms / frame_ms)

    segments = _frames_to_segments(is_speech, min_silence_frames, min_speech_frames)

    # Convert frame indices to seconds
    return [
        SpeechSegment(
            start_sec=start * frame_ms / 1000.0,
            end_sec=end * frame_ms / 1000.0,
        )
        for start, end in segments
    ]


def _auto_threshold(rms: np.ndarray) -> float:
    """Compute an automatic RMS threshold using percentile-based approach."""
    if len(rms) == 0:
        return 0.0

    # Use the 30th percentile of all frames as a baseline noise floor,
    # then set threshold slightly above it
    noise_floor = np.percentile(rms, 30)
    # Threshold at noise_floor + 40% of the range above it
    peak = np.percentile(rms, 95)
    if peak <= noise_floor:
        return noise_floor
    return noise_floor + 0.4 * (peak - noise_floor)


def _frames_to_segments(
    is_speech: np.ndarray,
    min_silence_frames: int,
    min_speech_frames: int,
) -> list[tuple[int, int]]:
    """Convert boolean frame array to merged (start, end) frame index pairs.

    Merges speech frames separated by gaps shorter than min_silence_frames.
    Drops segments shorter than min_speech_frames.
    """
    if len(is_speech) == 0:
        return []

    segments: list[tuple[int, int]] = []
    in_segment = False
    start = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_segment:
            start = i
            in_segment = True
        elif not speech and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(is_speech)))

    if not segments:
        return []

    # Merge segments separated by gaps < min_silence_frames
    merged: list[tuple[int, int]] = [segments[0]]
    for seg_start, seg_end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if seg_start - prev_end < min_silence_frames:
            merged[-1] = (prev_start, seg_end)
        else:
            merged.append((seg_start, seg_end))

    # Drop segments shorter than min_speech_frames
    return [(s, e) for s, e in merged if e - s >= min_speech_frames]
