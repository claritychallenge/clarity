"""Class compute crossover filter for one crossover frequency."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, tf2zpk, zpk2tf


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

    def __init__(
        self,
        freq_crossover: list | ndarray | int,
        N: int = 4,
        sample_rate: float = 44100,
    ) -> None:
        """Initialize the crossover filter.

        Args:
            freq_crossover (list | ndarray | int): The crossover frequency (Hz).
            N (int): The order of the filter.
            sample_rate (float): The sample rate of the signal (Hz).
        """
        if isinstance(freq_crossover, int):
            freq_crossover = [freq_crossover]

        if N != 4:
            raise ValueError(f"The order of the filter must be 4. {N} was provided.")

        self.xover_freqs = np.array(freq_crossover)
        self.order = N

        self.sample_rate = sample_rate

        # Initialize the filter coefficients
        self.bstore = np.zeros((self.order + 1, self.xover_freqs.shape[0], 2))
        self.astore = np.zeros((self.order + 1, self.xover_freqs.shape[0], 2))
        self.bstore_phi = np.zeros((2 * self.order + 1, self.xover_freqs.shape[0]))
        self.astore_phi = np.zeros((2 * self.order + 1, self.xover_freqs.shape[0]))
        self.compute_coefficients()

    def compute_coefficients(self) -> None:
        """Compute the filter coefficients.
        These are independednt of the signal and can be computed once.
        """
        for idx, xover_freq in enumerate(self.xover_freqs):
            for ifilt in range(2):
                if ifilt == 0:
                    btype = "low"
                else:
                    btype = "high"
                b, a = self.linkwitz_riley(xover_freq / (self.sample_rate / 2), btype)
                self.bstore[:, idx, ifilt] = b
                self.astore[:, idx, ifilt] = a

            self.bstore_phi[:, idx], self.astore_phi[:, idx] = self.make_all_pass(
                self.bstore[:, idx, 0],
                self.astore[:, idx, 0],
                self.bstore[:, idx, 1],
                self.astore[:, idx, 1],
            )

    @staticmethod
    def make_all_pass(b1, a1, b2, a2):
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

    def linkwitz_riley(self, xover_freqs, btype):
        """
        Compute the Linkwitz-Riley filter.
        Makes a filter that is equivalent to passing through butterworth
        twice to get linkwitz-riley

        Args:
            xover_freqs: The crossover frequency.
            btype: The type of filter.

        Returns:
            np.ndarray: The numerator and denominator of the filter.
        """
        # filter coefficients for Butterworth
        b, a = butter(self.order // 2, xover_freqs, btype=btype)
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

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply the crossover filter to the signal.

        Args:
            signal (np.ndarray): The input signal.

        Returns:
            np.ndarray: The filtered signal.
        """
        # Running only with 5 crossover filters
        filtered_signal = np.zeros((len(self.xover_freqs) + 1, signal.shape[0]))
        for filter_idx in range(self.xover_freqs.shape[0] + 1):
            filtered_signal[filter_idx] = self.xover_component(
                signal, filter_idx, len(self.xover_freqs)
            )
        return filtered_signal

    def xover_component(self, signal, filter_idx, len_xover_freqs):
        # The low pass filter component
        if filter_idx < len_xover_freqs:
            signal = lfilter(
                self.bstore[:, filter_idx, 0], self.astore[:, filter_idx, 0], signal
            )
        # The high pass filter component
        if filter_idx > 0:
            if filter_idx == 1:
                signal = lfilter(self.bstore[:, 0, 1], self.astore[:, 0, 1], signal)
            else:
                for m in range(filter_idx):
                    signal = lfilter(self.bstore[:, m, 1], self.astore[:, m, 1], signal)

        # The phi filter component
        if len_xover_freqs + 1 > 2:
            if filter_idx == len_xover_freqs - 2:
                signal = lfilter(
                    self.bstore_phi[:, filter_idx + 1],
                    self.astore_phi[:, filter_idx + 1],
                    signal,
                )
            elif filter_idx < len_xover_freqs - 2:
                for m in range(filter_idx + 1, len_xover_freqs):
                    signal = lfilter(
                        self.bstore_phi[:, m], self.astore_phi[:, m], signal
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
        for idx in range(len(self.xover_freqs) + 1):
            Y = np.fft.fft(y[idx, :])
            plt.plot(fplt, 20 * np.log10(np.abs(Y)))

        ytotal = np.sum(y, axis=0)
        Ytotal = np.fft.fft(ytotal)
        plt.plot(fplt, 20 * np.log10(np.abs(Ytotal)))
        plt.xlim([-10, int(self.sample_rate / 2)])
        plt.ylim([-250, 10])
        plt.show()
