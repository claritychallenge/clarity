"""Utilities for data generation."""
from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import numpy as np
import scipy
import scipy.io

SPEECH_FILTER: Final = np.array(
    scipy.io.loadmat(
        Path(__file__).parent / "params/speech_weight.mat",
        squeeze_me=True,
    )["filt"]
)


def better_ear_speechweighted_snr(target: np.ndarray, noise: np.ndarray) -> float:
    """Calculate effective better ear SNR.

    Args:
        target (np.ndarray):
        noise (np.ndarray):

    Returns:
        (float)
    Maximum Signal Noise Ratio between left and right channel.
    """
    if np.ndim(target) == 1:
        # analysis left ear and right ear for single channel target
        left_snr = speechweighted_snr(target, noise[:, 0])
        right_snr = speechweighted_snr(target, noise[:, 1])
    else:
        # analysis left ear and right ear for two channel target
        left_snr = speechweighted_snr(target[:, 0], noise[:, 0])
        right_snr = speechweighted_snr(target[:, 1], noise[:, 1])
    # snr is the max of left and right
    be_snr = max(left_snr, right_snr)
    return be_snr


def speechweighted_snr(target: np.ndarray, noise: np.ndarray) -> float:
    """Apply speech weighting filter to signals and get SNR.

    Args:
        target (np.ndarray):
        noise (np.ndarray):

    Returns:
        (float):
    Signal Noise Ratio
    """

    target_filt = scipy.signal.convolve(
        target, SPEECH_FILTER, mode="full", method="fft"
    )
    noise_filt = scipy.signal.convolve(noise, SPEECH_FILTER, mode="full", method="fft")

    # rms of the target after speech weighted filter
    targ_rms = np.sqrt(np.mean(target_filt**2))

    # rms of the noise after speech weighted filter
    noise_rms = np.sqrt(np.mean(noise_filt**2))
    sw_snr = np.divide(targ_rms, noise_rms)
    return sw_snr


def sum_signals(signals: list) -> np.ndarray | Literal[0]:
    """Return sum of a list of signals.

    Signals are stored as a list of ndarrays whose size can vary in the first
    dimension, i.e., so can sum mono or stereo signals etc.
    Shorter signals are zero padded to the length of the longest.

    Args:
        signals (list): List of signals stored as ndarrays

    Returns:
        np.ndarray: The sum of the signals

    """
    max_length = max(x.shape[0] for x in signals)
    return sum(pad(x, max_length) for x in signals)


def pad(signal: np.ndarray, length: int) -> np.ndarray:
    """Zero pad signal to required length.

    Assumes required length is not less than input length.

    Args:
        signal (np.array):
        length (int):

    Returns:
        np.array:
    """

    if length < signal.shape[0]:
        raise ValueError("Length must be greater than signal length")

    return np.pad(
        signal, ([(0, length - signal.shape[0])] + ([(0, 0)] * (len(signal.shape) - 1)))
    )
