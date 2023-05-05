from __future__ import annotations

import numpy as np


def compute_rms(signal: np.ndarray) -> float:
    """Compute RMS of signal
    Args:
        signal: Signal to compute RMS of.
    Returns:
        float: RMS of the signal.
    """
    if len(signal) == 0:
        return 0
    return np.sqrt(np.mean(np.square(signal)))


def normalize_signal(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize the signal to have zero mean and unit variance.

    Args:
        signal: The signal to be standardized.
    Returns:
        The standardized signal and the reference signal.
    """
    ref = signal.mean(0)
    return (signal - ref.mean()) / ref.std(), ref


def denormalize_signals(sources: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale signals back to the original scale.

    Args:
        sources (np.ndarray): Source to be scaled.
        ref (np.ndarray): Original sources to be used for reverting scaling.

    Returns:
        np.ndarray: Signal rescaled back to its original."""
    return sources * ref.std() + ref.mean()


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
