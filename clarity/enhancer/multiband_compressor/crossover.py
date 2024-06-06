"""Class compute crossover filter for one crossover frequency."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, tf2zpk, zpk2tf

if TYPE_CHECKING:
    from numpy import ndarray


def compute_coefficients(
    xover_freqs: ndarray, sample_rate: float = 44100, order: int = 4
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Compute the filter coefficients.
    These are independent of the signal.

    Args:
        xover_freqs (ndarray): The crossover frequencies.
        sample_rate (float): The sample rate of the signal.
        order (int): The order of the filter.

    Returns:
        bsotre (ndarray): The numerator of the filter.
        astore (ndarray): The denominator of the filter.
        bstore_phi (ndarray): The phase correction for the numerator of the
            all pass filter.
        astore_phi (ndarray): The phase correction for the denominator of the
            all pass filter.
    """
    bstore = np.zeros((order + 1, xover_freqs.shape[0], 2))
    astore = np.zeros((order + 1, xover_freqs.shape[0], 2))
    bstore_phi = np.zeros((2 * order + 1, xover_freqs.shape[0]))
    astore_phi = np.zeros((2 * order + 1, xover_freqs.shape[0]))

    for i, freq in enumerate(xover_freqs):
        for ifilt in range(2):
            if ifilt == 0:
                btype = "low"
            else:
                btype = "high"
            b, a = linkwitz_riley(freq / (sample_rate / 2), btype, order)
            bstore[:, i, ifilt] = b
            astore[:, i, ifilt] = a

        bstore_phi[:, i], astore_phi[:, i] = make_all_pass(
            bstore[:, i, 0],
            astore[:, i, 0],
            bstore[:, i, 1],
            astore[:, i, 1],
        )

    return bstore, astore, bstore_phi, astore_phi


def make_all_pass(
    b1: ndarray, a1: ndarray, b2: ndarray, a2: ndarray
) -> tuple[ndarray, ndarray]:
    """
    Take two filters [b1,a1] and [b2,a2]
    and calculate the coefficient of a filter that is equivalent to
    passing the input through each filter in parallel and summing the result

    Args:
        b1: Numerator of the first filter
        a1: Denominator of the first filter
        b2: Numerator of the second filter
        a2: Denominator of the second filter

    Returns:
        np.ndarray: The numerator and denominator of the all pass filter.
    """
    # to poles and zeros
    r1, p1, k1 = tf2zpk(b1, a1)
    r2, p2, k2 = tf2zpk(b2, a2)
    roots1 = np.concatenate((r1, p2))
    roots2 = np.concatenate((r2, p1))
    poly1 = np.poly(roots1)
    poly2 = np.poly(roots2)
    bt = poly1 * k1 + poly2 * k2
    at = np.poly(np.concatenate((p1, p2)))
    return bt, at


def linkwitz_riley(
    xover_freqs: float, btype: str, order: int = 4
) -> tuple[ndarray, ndarray]:
    """
    Compute the Linkwitz-Riley filter.
    Makes a filter that is equivalent to passing through butterworth
    twice to get linkwitz-riley

    Args:
        xover_freqs: The crossover frequency.
        btype: The type of filter.
        order: The order of the filter.

    Returns:
        np.ndarray: The numerator and denominator of the filter.
    """
    # filter coefficients for Butterworth
    b, a = butter(order // 2, xover_freqs, btype=btype)
    # get poles, zeros and gain
    r, p, k = tf2zpk(b, a)
    # duplicate zeros
    r = np.concatenate((r, r))
    # duplicate poles
    p = np.concatenate((p, p))
    # square the gain
    k = k**2
    # convert poles, zeros and gain back to filter coefficients
    b, a = zpk2tf(r, p, k)
    return b, a


class Crossover:
    """
    A class to compute crossover filter for two or more crossover frequencies.
    This is based on [1]

    References:
    [1] D'Appolito, J. A. (1984, October). Active realization of multiway all-pass
    crossover systems. In Audio Engineering Society Convention 76.
    Audio Engineering Society.

    Example:
    >>> xover_freqs = np.array(
    ...    [250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
    ...) * np.sqrt(2)
    >>> xover = Crossover(xover_freqs)
    >>> signal = np.random.randn(1000)
    >>> filtered_signal = xover(signal)
    >>> xover.plot_filter()
    """

    FILTER_ORDER = 4  # Implementation only for filters order 4

    def __init__(
        self,
        freq_crossover: list | ndarray | int,
        sample_rate: float = 44100,
    ) -> None:
        """Initialize the crossover filter.

        Args:
            freq_crossover (list | ndarray | int): The crossover frequency (Hz).
            sample_rate (float): The sample rate of the signal (Hz).
        """
        if isinstance(freq_crossover, int):
            freq_crossover = [freq_crossover]

        self.xover_freqs = np.array(freq_crossover)
        self.sample_rate = sample_rate

        # Initialize the filter coefficients

        self.bstore, self.astore, self.bstore_phi, self.astore_phi = (
            compute_coefficients(self.xover_freqs, self.sample_rate, self.FILTER_ORDER)
        )

    def __call__(self, signal: np.ndarray, axis: int = -1) -> np.ndarray:
        """Apply the crossover filter to the signal.

        Args:
            signal (np.ndarray): The input signal.
            axis (int): The axis along which to apply the filter.
              Default is -1. More information in ```scipy.signal.lfilter```
              documentation.

        Returns:
            np.ndarray: The filtered signal.
        """
        output_signal = []
        for filter_idx in range(self.xover_freqs.shape[0] + 1):
            output_signal.append(
                self.xover_component(
                    signal, filter_idx, len(self.xover_freqs), axis=axis
                )
            )
        return np.array(output_signal)

    def xover_component(
        self, signal: ndarray, filter_idx: int, len_xover_freqs: int, axis: int = -1
    ) -> ndarray:
        """Apply the crossover filter to the signal.

        Args:
            signal (np.ndarray): The input signal.
            filter_idx (int): The index of the filter to apply.
            len_xover_freqs (int): The number of crossover frequencies.
            axis (int): The axis along which to apply the filter.
              Default is -1. ```More information in scipy.signal.lfilter```
              documentation.

        Returns:
            np.ndarray: The filtered signal.
        """
        # The low pass filter component
        if filter_idx < len_xover_freqs:
            signal = lfilter(
                self.bstore[:, filter_idx, 0],
                self.astore[:, filter_idx, 0],
                signal,
                axis=axis,
            )
        # The high pass filter component
        if filter_idx > 0:
            if filter_idx == 1:
                signal = lfilter(
                    self.bstore[:, 0, 1], self.astore[:, 0, 1], signal, axis=axis
                )
            else:
                for m in range(filter_idx):
                    signal = lfilter(
                        self.bstore[:, m, 1], self.astore[:, m, 1], signal, axis=axis
                    )

        # The phi filter component
        if len_xover_freqs + 1 > 2:
            if filter_idx == len_xover_freqs - 2:
                signal = lfilter(
                    self.bstore_phi[:, filter_idx + 1],
                    self.astore_phi[:, filter_idx + 1],
                    signal,
                    axis=axis,
                )
            elif filter_idx < len_xover_freqs - 2:
                for m in range(filter_idx + 1, len_xover_freqs):
                    signal = lfilter(
                        self.bstore_phi[:, m], self.astore_phi[:, m], signal, axis=axis
                    )

        return signal

    def __str__(self):
        """Method to print the crossover filter."""
        return f"Crossover frequencies: {[round(x, 4) for x in self.xover_freqs]} Hz"

    def plot_filter(self):
        """Method to plot the frequency response of the filter.
        This can help to validate the Class is generating the expected filters
        """

        x = np.concatenate(([1], np.zeros(65500)))
        y = self(x)

        fplt = np.linspace(0, self.sample_rate, len(x))
        plt.figure()
        for i in range(len(self.xover_freqs) + 1):
            Y = np.fft.fft(y[i, :])
            plt.plot(fplt, 20 * np.log10(np.abs(Y)))

        ytotal = np.sum(y, axis=0)
        Ytotal = np.fft.fft(ytotal)
        plt.plot(
            fplt,
            20 * np.log10(np.abs(Ytotal)),
        )
        print(np.sum(20 * np.log10(np.abs(Ytotal))))
        plt.xlim([-10, int(self.sample_rate / 2)])
        plt.ylim([-250, 10])
        plt.show()
