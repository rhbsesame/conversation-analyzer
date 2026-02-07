"""WAV loading, channel splitting, normalization, and resampling."""

from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_wav(path: str | Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Load a stereo WAV file and return (sample_rate, left, right).

    Channels are normalized to float64 in the range [-1, 1].

    Raises:
        ValueError: If the file is not stereo.
    """
    sample_rate, data = wavfile.read(path)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Expected stereo WAV (2 channels), got {data.ndim}D array"
            f" with shape {data.shape}"
        )

    # Normalize to float64 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    elif data.dtype == np.float64:
        pass
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

    left = data[:, 0]
    right = data[:, 1]
    return sample_rate, left, right


def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D signal from orig_sr to target_sr.

    Returns the signal unchanged if sample rates already match.
    """
    if orig_sr == target_sr:
        return signal
    g = gcd(orig_sr, target_sr)
    return resample_poly(signal, target_sr // g, orig_sr // g).astype(signal.dtype)
