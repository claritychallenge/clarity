from typing import Tuple

import numpy as np


def normalize_signal(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
