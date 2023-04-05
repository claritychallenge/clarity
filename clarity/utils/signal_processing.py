"""Signal processing utilities."""
from __future__ import annotations

import numpy as np


def compute_rms(signal: np.ndarray) -> float:
    """Compute RMS of signal

    Args:
        signal: Signal to compute RMS of.
    Returns:
        float: RMS of the signal.
    """
    return np.sqrt(np.mean(np.square(signal)))


def denormalize_signals(sources: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale signals back to the original scale.

    Args:
        sources (np.ndarray): Source to be scaled.
        ref (np.ndarray): Original sources to be used for reverting scaling.

    Returns:
        np.ndarray: Signal rescaled back to its original."""
    return sources * ref.std() + ref.mean()


def normalize_signal(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize the signal to have zero mean and unit variance.

    Args:
        signal: The signal to be standardized.
    Returns:
        The standardized signal and the reference signal.
    """
    ref = signal.mean(0)
    return (signal - ref.mean()) / ref.std(), ref
