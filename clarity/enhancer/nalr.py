from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
import scipy.signal

from clarity.evaluator.msbg.msbg_utils import firwin2

if TYPE_CHECKING:
    from numpy import ndarray


class NALR:
    def __init__(self, nfir: int, sample_rate: float) -> None:
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
        """Interpolate the hearing loss at the audiometric frequencies."""
        try:
            hl_interp_fn = scipy.interpolate.interp1d(cfs, hl)
        except ValueError as exception:
            raise ValueError(
                "Hearing losses (hl) and center frequencies (cfs) don't match!"
            ) from exception
        return hl_interp_fn(self.aud)

    def build(
        self,
        hl: ndarray,
        cfs: ndarray | None = None,
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
            hl = self.hl_interp(hl, cfs)

        max_loss = np.max(hl)

        if max_loss > 0:
            # Compute the NAL-R frequency response at the audiometric frequencies
            bias = np.array([-17, -8, 1, -1, -2, -2])

            critical_loss = hl[1] + hl[2] + hl[3]  # <-- loss at 500 Hz to 2 kHz
            if critical_loss <= 180:
                x_ave = 0.05 * critical_loss
            else:
                x_ave = 9.0 + 0.116 * (critical_loss - 180)
            gain_db = x_ave + 0.31 * hl + bias
            gain_db = np.clip(gain_db, a_min=0, a_max=None)

            # Design the linear-phase FIR filter

            # Build the gain interpolation function
            freq_ext: ndarray = np.concatenate(
                (np.array([0.0]), self.aud, np.array([self.fmax]))
            )
            gain_db_ext = np.concatenate(([gain_db[0]], gain_db, [gain_db[-1]]))
            interp_fn = scipy.interpolate.interp1d(freq_ext, gain_db_ext)

            # Interpolate gains at uniform frequency spacing from 0 to 1
            center_freqs = np.linspace(0, self.nfir, self.nfir + 1) / self.nfir
            interpolated_gain_db = interp_fn(self.fmax * center_freqs)
            interpolated_gain_linear = np.power(10, interpolated_gain_db / 20.0)
            nalr = firwin2(self.nfir + 1, center_freqs, interpolated_gain_linear)
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
