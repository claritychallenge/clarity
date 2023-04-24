"""4th-order gammatone auditory filter."""
from __future__ import annotations

# pylint: disable=import-error
import logging
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from scipy.signal import lfilter

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class GammatoneFilter:
    def __init__(
        self, freq_sample: float, ear_q: float = 9.26449, min_bandwidth: float = 24.7
    ):
        """
        This implementation is based on the c program
        published on-line by Ning Ma, U. Sheffield, UK [1]. that gives an
        implementation of the Martin Cooke filters [2]:
        an impulse-invariant transformation of the gammatone filter.
        The signal is demodulated down to baseband using a complex exponential,
        and then passed through a cascade of four one-pole low-pass filters.

        Arguments:
            freq_sample (float): sampling rate in Hz
            ear_q (float): ???
            min_bandwidth (float): ???

        References:
            [1] Ma N, Green P, Barker J, Coy A (2007) Exploiting correlogram
                   structure for robust speech recognition with multiple speech
                   sources. Speech Communication, 49 (12): 874-891. Available at
                   <https://doi.org/10.1016/j.specom.2007.05.003>
                   <https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/gammatone/>
            [2] Cooke, M. (1993) Modelling auditory processing and organisation.
                   Cambridge University Press
        """
        self.freq_sample = freq_sample
        self.ear_q = ear_q
        self.min_bandwidth = min_bandwidth
        self.tpt = 2 * np.pi / self.freq_sample

    def compute(
        self, signal: ndarray, bandwidth: float, center_freq: float
    ) -> tuple[ndarray, ndarray]:
        erb = self.min_bandwidth + (center_freq / self.ear_q)

        # Filter the first signal
        # Initialize the filter coefficients
        tpt_bw = bandwidth * self.tpt * erb * 1.019
        a = np.exp(-tpt_bw)
        a_1 = 4.0 * a
        a_2 = -6.0 * a * a
        a_3 = 4.0 * a * a * a
        a_4 = -a * a * a * a
        a_5 = 4.0 * a * a
        gain = 2.0 * (1 - a_1 - a_2 - a_3 - a_4) / (1 + a_1 + a_5)

        # Initialize the complex demodulation
        npts = len(signal)
        sincf, coscf = self.gammatone_bandwidth_demodulation(
            npts, self.tpt, center_freq
        )

        # Filter the real and imaginary parts of the signal
        ureal = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], signal * coscf)
        uimag = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], signal * sincf)

        # lfilter can return different types
        assert isinstance(ureal, np.ndarray)
        assert isinstance(uimag, np.ndarray)

        # Extract the BM velocity and the envelope
        basilar_membrane = gain * (ureal * coscf + uimag * sincf)
        envelope = gain * np.sqrt(ureal * ureal + uimag * uimag)

        return envelope, basilar_membrane

    @staticmethod
    @njit
    def gammatone_bandwidth_demodulation(npts: int, tpt: float, center_freq: float):
        """Gamma tone bandwidth demodulation

        Arguments:
            npts (): ???
            tpt (): ???
            center_freq (): ???

        Returns:
            sincf (): ???
            coscf (): ???
        """
        center_freq_sin = np.zeros(npts)
        center_freq_cos = np.zeros(npts)

        cos_n = np.cos(tpt * center_freq)
        sin_n = np.sin(tpt * center_freq)
        cold = 1.0
        sold = 0.0
        center_freq_cos[0] = cold
        center_freq_sin[0] = sold
        for n in range(1, npts):
            arg = cold * cos_n + sold * sin_n
            sold = sold * cos_n - cold * sin_n
            cold = arg
            center_freq_cos[n] = cold
            center_freq_sin[n] = sold

        return center_freq_sin, center_freq_cos
