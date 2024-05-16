"""Class compute crossover filter for one crossover frequency."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from clarity.enhancer.multiband_compressor.crossover.crossover_base import (
    CrossoverBase,
)
from scipy.signal import butter, tf2zpk, zpk2tf

import warnings


class CrossoverTwoOrMore(CrossoverBase):
    """
    A class to compute crossover filter for two or more crossover frequencies.
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
