from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
import scipy.signal

from clarity.evaluator.msbg.msbg_utils import firwin2

if TYPE_CHECKING:
    from numpy import ndarray


class NALR:
    def __init__(self, nfir: int, sample_rate: int) -> None:
        """
        Args:
            nfir: Order of the NAL-R EQ filter and the matching delay
            fs: Sampling rate in Hz
        """
        self.nfir = nfir
        # Processing parameters
        self.fmax = 0.5 * sample_rate

        # Audiometric frequencies
        self.aud = np.array([250, 500, 1000, 2000, 4000, 6000])

        # Design a flat filter having the same delay as the NAL-R filter
        self.delay = np.zeros(nfir + 1)
        self.delay[nfir // 2] = 1.0

    def hl_interp(self, hl: np.ndarray, cfs: np.ndarray) -> ndarray:
        try:
            hl_interpf = scipy.interpolate.interp1d(cfs, hl)
        except ValueError as exception:
            raise ValueError(
                "Hearing losses (hl) and center frequencies (cfs) don't match!"
            ) from exception
        return hl_interpf(self.aud)

    def build(
        self,
        hl: np.ndarray,
        cfs: np.ndarray | None = None,
    ) -> tuple[ndarray, ndarray]:
        """
        Args:
            hl: hearing thresholds at [250, 500, 1000, 2000, 4000, 6000] Hz
            cfs: center frequencies of the hearing thresholds. If None, the default
                values are used.
        Returns:
            NAL-R FIR filter
            delay
        """

        # Apply interpolation only if cfs is not None
        if cfs is not None:
            hl = self.hl_interp(np.array(hl), np.array(cfs))

        mloss = np.max(hl)

        if mloss > 0:
            # Compute the NAL-R frequency response at the audiometric frequencies
            bias = np.array([-17, -8, 1, -1, -2, -2])
            t3 = hl[1] + hl[2] + hl[3]
            if t3 <= 180:
                xave = 0.05 * t3
            else:
                xave = 9.0 + 0.116 * (t3 - 180)
            gdB = xave + 0.31 * hl + bias
            gdB = np.clip(gdB, a_min=0, a_max=None)

            # Design the linear-phase FIR filter
            fv = np.append(
                np.append(0, self.aud), self.fmax
            )  # Frequency vector for the interpolation
            cfreq = (
                np.linspace(0, self.nfir, self.nfir + 1) / self.nfir
            )  # Uniform frequency spacing from 0 to 1
            gdBv = np.append(
                np.append(gdB[0], gdB), gdB[-1]
            )  # gdB vector for the interpolation
            interpf = scipy.interpolate.interp1d(fv, gdBv)
            gain = interpf(self.fmax * cfreq)
            glin = np.power(10, gain / 20.0)
            nalr = firwin2(self.nfir + 1, cfreq, glin)
        else:
            nalr = self.delay.copy()
        return nalr, self.delay

    def apply(self, nalr: np.ndarray, wav: np.ndarray) -> ndarray:
        """
        Args:
            nalr: built NAL-R FIR filter
            wav: one dimensional wav signal

        Returns:
            amplified signal
        """
        return np.convolve(wav, nalr)
