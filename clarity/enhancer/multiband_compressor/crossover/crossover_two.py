"""Class compute crossover filter for one crossover frequency."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, tf2zpk, zpk2tf

from clarity.enhancer.multiband_compressor.crossover.crossover_base import CrossoverBase


class CrossoverTwoOrMore(CrossoverBase):
    """
    A class to compute crossover filter for two or more crossover frequencies.
    This is based on [1]

    References:
    [1] D'Appolito, J. A. (1984, October). Active realization of multiway all-pass
    crossover systems. In Audio Engineering Society Convention 76.
    Audio Engineering Society.
    """

    def __init__(
        self,
        freq_crossover: list | int,
        N: int = 4,
        sample_rate: float = 44100,
    ) -> None:
        """Initialize the crossover filter.

        Args:
            freq_crossover (list | int): The crossover frequency (Hz).
            N (int): The order of the filter.
            sample_rate (float): The sample rate of the signal (Hz).
        """
        super().__init__(freq_crossover, N, sample_rate)
        # Initialize the filter coefficients
        self.bstore = None
        self.astore = None
        self.bstore_phi = None
        self.astore_phi = None
        self.compute_coefficients()

    def compute_coefficients(self) -> None:
        """Compute the filter coefficients.
        These are independednt of the signal and can be computed once.
        """
        self.bstore = np.zeros((self.order + 1, self.xover_freqs.shape[0], 2))
        self.astore = np.zeros((self.order + 1, self.xover_freqs.shape[0], 2))
        self.bstore_phi = np.zeros((2 * self.order + 1, self.xover_freqs.shape[0]))
        self.astore_phi = np.zeros((2 * self.order + 1, self.xover_freqs.shape[0]))

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
        if self.xover_freqs.shape[0] != 5:
            warnings.warn(
                "This class is implemented for 5 crossover filters."
                " We are working on generalise it for any number of crossover"
                " frequencies greater than 2."
                " For now, the signal will be returned without filtering."
                " Returning the input signal without filtering."
            )
            return signal

        # Apply the crossover filters
        filtered_signal = np.zeros((len(self.xover_freqs) + 1, signal.shape[0]))
        for idx, xover_freq in enumerate(self.xover_freqs):
            if idx == 0:
                filtered_signal[idx] = self.xover_1(signal)
            elif idx == 1:
                filtered_signal[idx] = self.xover_2(signal)
            elif idx == 2:
                filtered_signal[idx] = self.xover_3(signal)
            elif idx == 3:
                filtered_signal[idx] = self.xover_4(signal)
            elif idx == 4:
                filtered_signal[idx] = self.xover_5(signal)
            else:
                raise ValueError("This class is implemented for 5 crossover filters.")
            filtered_signal[5] = self.xover_6(signal)

        return filtered_signal

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

        plt.show()

    def xover_6(self, signal):
        for m in range(5):
            signal = lfilter(self.bstore[:, m, 1], self.astore[:, m, 1], signal)
        return signal

    def xover_5(self, signal):
        for m in range(4):
            signal = lfilter(self.bstore[:, m, 1], self.astore[:, m, 1], signal)
        signal = lfilter(self.bstore[:, 4, 0], self.astore[:, 4, 0], signal)
        return signal

    def xover_4(self, signal):
        for m in range(3):
            signal = lfilter(self.bstore[:, m, 1], self.astore[:, m, 1], signal)
        signal = lfilter(self.bstore[:, 3, 0], self.astore[:, 3, 0], signal)
        signal = lfilter(self.bstore_phi[:, 4], self.astore_phi[:, 4], signal)
        return signal

    def xover_3(self, signal):
        for m in range(2):
            signal = lfilter(self.bstore[:, m, 1], self.astore[:, m, 1], signal)
        signal = lfilter(self.bstore[:, 2, 0], self.astore[:, 2, 0], signal)
        for m in range(3, 5):
            signal = lfilter(self.bstore_phi[:, m], self.astore_phi[:, m], signal)
        return signal

    def xover_2(self, signal):
        signal = lfilter(self.bstore[:, 0, 1], self.astore[:, 0, 1], signal)
        signal = lfilter(self.bstore[:, 1, 0], self.astore[:, 1, 0], signal)
        for m in range(2, 5):
            signal = lfilter(self.bstore_phi[:, m], self.astore_phi[:, m], signal)
        return signal

    def xover_1(self, signal):
        signal = lfilter(self.bstore[:, 0, 0], self.astore[:, 0, 0], signal)
        for m in range(1, 5):
            signal = lfilter(self.bstore_phi[:, m], self.astore_phi[:, m], signal)
        return signal


if __name__ == "__main__":
    # Test the class
    xover_freqs = np.array([250, 500, 1000, 2000, 4000]) / np.sqrt(2)
    xover = CrossoverTwoOrMore(xover_freqs)
    signal = np.random.randn(1000)
    filtered_signal = xover(signal)
    print(filtered_signal.shape)
    xover.plot_filter()
