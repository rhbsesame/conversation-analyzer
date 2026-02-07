"""Tests for audio.py — WAV loading, stereo validation, normalization."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from conversation_analyzer.audio import load_wav


def _write_wav(path: Path, sample_rate: int, data: np.ndarray) -> None:
    wavfile.write(str(path), sample_rate, data)


class TestLoadWav:
    def test_stereo_int16(self, tmp_path):
        sr = 16000
        samples = sr * 2  # 2 seconds
        left = np.sin(2 * np.pi * 440 * np.arange(samples) / sr)
        right = np.sin(2 * np.pi * 880 * np.arange(samples) / sr)
        data = np.column_stack([left, right])
        data_int16 = (data * 32767).astype(np.int16)

        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path, sr, data_int16)

        rate, l, r = load_wav(wav_path)
        assert rate == sr
        assert l.dtype == np.float64
        assert r.dtype == np.float64
        assert len(l) == samples
        assert len(r) == samples
        # Check normalization range
        assert np.max(np.abs(l)) <= 1.0
        assert np.max(np.abs(r)) <= 1.0

    def test_stereo_float32(self, tmp_path):
        sr = 44100
        samples = sr
        data = np.random.uniform(-1, 1, (samples, 2)).astype(np.float32)

        wav_path = tmp_path / "float32.wav"
        _write_wav(wav_path, sr, data)

        rate, l, r = load_wav(wav_path)
        assert rate == sr
        assert l.dtype == np.float64
        assert r.dtype == np.float64
        np.testing.assert_allclose(l, data[:, 0].astype(np.float64), atol=1e-6)

    def test_mono_rejected(self, tmp_path):
        sr = 16000
        data = np.zeros(sr, dtype=np.int16)  # mono

        wav_path = tmp_path / "mono.wav"
        _write_wav(wav_path, sr, data)

        with pytest.raises(ValueError, match="stereo"):
            load_wav(wav_path)

    def test_normalization_range(self, tmp_path):
        sr = 16000
        # Max int16 values
        data = np.array([[32767, -32768]] * sr, dtype=np.int16)

        wav_path = tmp_path / "max.wav"
        _write_wav(wav_path, sr, data)

        _, l, r = load_wav(wav_path)
        assert np.isclose(np.max(l), 32767 / 32768)
        assert np.isclose(np.min(r), -1.0)
