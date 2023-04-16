"""Dataclass to represent a monaural audiogram"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

DEFAULT_CLARITY_AUDIOGRAM_FREQUENCIES: Final = np.array(
    [
        250,
        500,
        1000,
        2000,
        3000,
        4000,
        6000,
        8000,
    ]
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


@dataclass
class Audiogram:
    """Dataclass to represent an audiogram.

    Attributes:
        levels (ndarray): The audiometric levels in dB HL
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

        if impairment_degree > 56.0:
            return "SEVERE"
        if impairment_degree > 35.0:
            return "MODERATE"
        if impairment_degree > 15.0:
            return "MILD"

        return "NOTHING"

    def resample(self, new_frequencies: ndarray) -> Audiogram:
        """Resample the audiogram to a new set of frequencies.

        Interpolates linearly on a log frequency axis.

        Args:
            new_frequencies (ndarray): The new frequencies to resample to

        Returns:
            Audiogram: New audiogram with resampled frequencies

        """

        return Audiogram(
            levels=np.interp(
                np.log(new_frequencies),
                np.log(self.frequencies),
                self.levels,
                left=self.levels[0],
                right=self.levels[-1],
            ),
            frequencies=new_frequencies,
        )


# Reference audiograms originally defined in the Cambridge group
# MSBG model MATLAB code.

# No loss
AUDIOGRAM_REF: Final = Audiogram(
    frequencies=FULL_STANDARD_AUDIOGRAM_FREQUENCIES,
    levels=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
)

# Mild age-related hearing loss, slightly reduced from CF, first-time aid
# wearers group (used in Stafa talk by MAS).
AUDIOGRAM_MILD: Final = Audiogram(
    frequencies=FULL_STANDARD_AUDIOGRAM_FREQUENCIES,
    levels=np.array([5, 10, 15, 18, 19, 22, 25, 28, 31, 35, 38, 40, 40, 45, 50]),
)

# Moderate hearing loss based on mild N2 flat/mod sloping
# from Bisgaard et al. 2020
AUDIOGRAM_MODERATE: Final = Audiogram(
    frequencies=FULL_STANDARD_AUDIOGRAM_FREQUENCIES,
    levels=np.array([15, 20, 20, 22.5, 25, 30, 35, 40, 45, 50, 55, 55, 60, 65, 65]),
)

# Moderate-severe age-related hearing loss, average of MAS/KA summer proj 2011,
# elderly HI
AUDIOGRAM_MODERATE_SEVERE: Final = Audiogram(
    frequencies=FULL_STANDARD_AUDIOGRAM_FREQUENCIES,
    levels=np.array([19, 19, 28, 35, 40, 47, 52, 56, 58, 58, 63, 70, 75, 80, 80]),
)
