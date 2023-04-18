"""Signal processing utilities."""
from __future__ import annotations

import numpy as np
import scipy
from numpy import ndarray


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


def resample(signal: ndarray, sample_rate: float, new_sample_rate: float) -> ndarray:
    """Resample a signal to a new sample rate.

    This is a sipmle wrapper around scipy.signal.resample. with the resampling
    expressed in terms of sampling rates rather than desired number of samples.
    It also ensures that for multichannel signals, resampling is in the time
    domain, i.e. down the columns.

    Args:
        signal: The signal to be resampled.
        sample_rate: The original sample rate.
        new_sample_rate: The new sample rate.
    Returns:
        The resampled signal.
    """
    if sample_rate == new_sample_rate:
        return signal
    return scipy.signal.resample(
        signal, int(new_sample_rate * signal.shape[0] / sample_rate)
    )
