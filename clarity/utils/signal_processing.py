"""Signal processing utilities."""
# pylint: disable=import-error
from __future__ import annotations

import numpy as np
import scipy
import soxr
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


def correlate(
    x: np.ndarray,
    y: np.ndarray,
    mode="full",
    method="auto",
    lags: int | float | None = None,
) -> np.ndarray:
    """
    Wrap of ``scipy.signal.correlate`` that includes a mode
    for maxlag.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    Args:
        x (np.ndarray): First signal
        y (np.ndarray): Second signal
        mode (str): Mode to pass to ``scipy.signal.correlate``
        method (str):
            'maxlag': Implement cross correlation with a maximum number of lags.
                      x and y must have the same length.
                based on https://stackoverflow.com/questions/30677241/
                        how-to-limit-cross-correlation-window-width-in-numpy
            "auto": Run scipy.signal.correlate with method='auto'
            'direct': Run scipy.signal.correlate with method='direct'
            'fft': Run scipy.signal.correlate with method='fft'
        lags (int): Maximum number of lags for `method` "maxlag".
    Returns:
        np.ndarray: cross correlation of x and y
    """
    if method == "maxlag":
        if lags is None:
            raise ValueError("maxlag must be specified for method='maxlag'")
        lags = int(lags)

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same length")

        py = np.pad(y.conj(), 2 * lags, mode="constant")
        # pylint: disable=unsubscriptable-object
        T = np.lib.stride_tricks.as_strided(
            py[2 * lags :],
            shape=(2 * lags + 1, len(y) + 2 * lags),
            strides=(-py.strides[0], py.strides[0]),
        )
        px = np.pad(x, lags, mode="constant")
        return T.dot(px)

    if method in ["auto", "direct", "fft"]:
        # Run scipy signal correlate with the specified method and mode
        return scipy.signal.correlate(x, y, mode=mode, method=method)

    raise ValueError(f"Unknown method: {method}")
