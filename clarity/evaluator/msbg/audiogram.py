"""Supplied dataclass to represent a monaural audiogram"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from numpy import ndarray

DEFAULT_CLARITY_AUDIOGRAM_FREQUENCIES: Final = (
    250,
    500,
    1000,
    2000,
    3000,
    4000,
    6000,
    8000,
)

FULL_STANDARD_AUDIOGRAM_FREQUENCIES: Final = np.array(
    [
        125,
        250,
        500,
        750,
        1000,
        1500,
        2000,
        3000,
        4000,
        6000,
        8000,
        10000,
        12000,
        14000,
        16000,
    ]
)


def logx_interp1d(x_in: ndarray, y_in: ndarray, x_out: ndarray) -> ndarray:
    """Linear interpolation on logarithmic x-axis.

    Interpolates linearly on a logarithmic x-axis, e.g. suitable for
    interpolating audiogram levels at arbitrary frequencies.

    Note, for x below the minimum of x_in, the first value of y_in is used,
    and for x above the maximum of x_in, the last value of y_in is used.

    Args:
        x_in (ndarray): x-axis values, e.g. audiogram frequencies (Hz)
        y_in (ndarray): y-axis values, e.g., audiogram levels
        x_out (ndarray): x-axis value at which y-value are required

    Returns:
        ndarray: interpolated y-values

    """
    return interp1d(
        np.log(x_in), y_in, bounds_error=False, fill_value=(y_in[0], y_in[-1])
    )(np.log(x_out))


@dataclass
class Audiogram:
    """Dataclass to represent an audiogram.

    Attributes:
        levels (ndarray): The levels for the left and right ear
        frequencies (ndarray): The frequencies at which the levels are measured
    """

    levels: np.ndarray
    frequencies: np.ndarray = np.array(DEFAULT_CLARITY_AUDIOGRAM_FREQUENCIES)

    def __post_init__(self) -> None:
        """Check that dimensions of levels and frequencies match."""
        if len(self.levels) != len(self.frequencies):
            raise ValueError(
                f"Levels ({len(self.levels)}) and frequencies ({len(self.frequencies)})"
                " must have the same length"
            )

    @property
    def severity(self) -> str:
        """Categorise HL severity level for the audiogram.

        Note that this categorisation is different from that of the British
        Society of Audiology, which recommends descriptors mild, moderate,
        severe and profound for average hearing threshold levels at 250, 500,
        1000, 2000 and 4000 Hz of 21-40 dB HL, 41-70 dB HL, 71-95 dB HL
        and > 95 dB HL, respectively (BSA Pure-tone air-conduction and
        bone-conduction threshold audiometry with and without masking
        2018).

        Returns:
            str -- severity level, one of SEVERE, MODERATE, MILD, NOTHING

        """
        # calculate mean hearing loss between critical frequencies of 2 & 8 kHz
        critical_freqs = np.logical_and(
            2000 <= self.frequencies, self.frequencies <= 8000
        )
        critical_levels = self.levels[critical_freqs]
        # Remove any None values
        critical_levels = [x for x in critical_levels if x is not None]
        # Ignore any None values
        impairment_degree = np.mean(critical_levels) if len(critical_levels) > 0 else 0

        if impairment_degree > 56:
            return "SEVERE"
        if impairment_degree > 35:
            return "MODERATE"
        if impairment_degree > 15:
            return "MILD"

        return "NOTHING"

    def resample(self, new_frequencies: ndarray) -> Audiogram:
        """Resample the audiogram to a new set of frequencies.

        Args:
            new_frequencies (ndarray): The new frequencies to resample to

        Returns:
            Audiogram: New audiogram with resampled frequencies

        """
        return Audiogram(
            levels=logx_interp1d(self.frequencies, self.levels, new_frequencies),
            frequencies=new_frequencies,
        )


# Reference processing: use to check levels between original and processed,
AUDIOGRAM_REF = Audiogram(
    frequencies=FULL_STANDARD_AUDIOGRAM_FREQUENCIES,
    levels=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
)

# mild age-related hearing loss, slightly reduced from CF, first-time aid wearers group
# (used in Stafa talk by MAS)
AUDIOGRAM_MILD = Audiogram(
    frequencies=FULL_STANDARD_AUDIOGRAM_FREQUENCIES,
    levels=np.array([5, 10, 15, 18, 19, 22, 25, 28, 31, 35, 38, 40, 40, 45, 50]),
)

# mod hearing loss based on mild N2 flat/mod sloping from Bisgaard et al. 2020
AUDIOGRAM_MODERATE = Audiogram(
    frequencies=FULL_STANDARD_AUDIOGRAM_FREQUENCIES,
    levels=np.array([15, 20, 20, 22.5, 25, 30, 35, 40, 45, 50, 55, 55, 60, 65, 65]),
)

# (mod-)severe age-related hearing loss, average of MAS/KA summer proj 2011, elderly HI
AUDIOGRAM_MODERATE_SEVERE = Audiogram(
    frequencies=FULL_STANDARD_AUDIOGRAM_FREQUENCIES,
    levels=np.array([19, 19, 28, 35, 40, 47, 52, 56, 58, 58, 63, 70, 75, 80, 80]),
)
