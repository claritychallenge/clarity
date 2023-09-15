"""Signal processing utilities."""
from __future__ import annotations

# pylint: disable=import-error
import numpy as np
import scipy
import soxr
from numpy import ndarray


def clip_signal(signal: np.ndarray, soft_clip: bool = False) -> tuple[np.ndarray, int]:
    """Clip the signal.

    Args:
        signal (np.ndarray): Signal to be clipped and saved.
        soft_clip (bool): Whether to use soft clipping.

    Returns:
        signal (np.ndarray): Clipped signal.
        n_clipped (int): Number of samples clipped.
    """

    if soft_clip:
        signal = np.tanh(signal)
    n_clipped = np.sum(np.abs(signal) > 1.0)
    signal = np.clip(signal, -1.0, 1.0)
    return signal, int(n_clipped)


def compute_rms(signal: ndarray) -> float:
    """Compute RMS of signal

    Args:
        signal: Signal to compute RMS of.
    Returns:
        float: RMS of the signal.
    """
    if len(signal) == 0:
        return 0
    return np.sqrt(np.mean(np.square(signal)))


def denormalize_signals(sources: ndarray, ref: ndarray) -> ndarray:
    """Scale signals back to the original scale.

    Args:
        sources (ndarray): Source to be scaled.
        ref (ndarray): Original sources to be used for reverting scaling.

    Returns:
        ndarray: Signal rescaled back to its original."""
    return sources * ref.std() + ref.mean()


def normalize_signal(signal: ndarray) -> tuple[ndarray, ndarray]:
    """Standardize the signal to have zero mean and unit variance.

    Args:
        signal: The signal to be standardized.
    Returns:
        The standardized signal and the reference signal.
    """
    ref = signal.mean(0)
    return (signal - ref.mean()) / ref.std(), ref


def resample(
    signal: ndarray,
    sample_rate: float,
    new_sample_rate: float,
    method: str = "soxr",
) -> ndarray:
    """Resample a signal to a new sample rate.

    This is a simple wrapper around  soxr and scipy.signal.resample with the resampling
    expressed in terms of input and output sampling rates.

    It also ensures that for multichannel signals, resampling is in the time
    domain, i.e. down the columns.

    Args:
        signal: The signal to be resampled.
        sample_rate: The original sample rate.
        new_sample_rate: The new sample rate.
        method: determine the approach use.
    Returns:
        The resampled signal.
    """
    if sample_rate == new_sample_rate:
        return signal

    if method == "soxr":
        return soxr.resample(signal, sample_rate, new_sample_rate, quality="HQ")

    if method == "polyphase":
        sample_rate = int(sample_rate)
        new_sample_rate = int(new_sample_rate)
        gcd = np.gcd(sample_rate, new_sample_rate)
        uprate = new_sample_rate // gcd
        downrate = sample_rate // gcd
        return scipy.signal.resample_poly(signal, up=uprate, down=downrate)

    if method == "fft":
        return scipy.signal.resample(
            signal, int(new_sample_rate * signal.shape[0] / sample_rate)
        )

    raise ValueError(f"Unknown resampling method: {method}")


def to_16bit(signal: np.ndarray) -> np.ndarray:
    """Convert the signal to 16 bit.

    Args:
        signal (np.ndarray): Signal to be converted.

    Returns:
        signal (np.ndarray): Converted signal.
    """
    signal = signal * 32768.0
    signal = np.clip(signal, -32768.0, 32767.0)
    return signal.astype(np.dtype("int16"))
