"""Dataclass to represent a monaural audiogram"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Final

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
    frequencies: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_CLARITY_AUDIOGRAM_FREQUENCIES)
    )

    def __post_init__(self) -> None:
        """Check that dimensions of levels and frequencies match."""

        # Ensure that levels and frequencies are numpy arrays
        self.levels = np.array(self.levels)
        self.frequencies = np.array(self.frequencies)

        if len(self.levels) != len(self.frequencies):
            raise ValueError(
                f"Levels ({len(self.levels)}) and frequencies ({len(self.frequencies)})"
                " must have the same length"
            )

        if len(self.frequencies) != len(np.unique(self.frequencies)):
            raise ValueError("Frequencies must be unique")

        if not np.all(np.diff(self.frequencies) > 0):
            raise ValueError("Frequencies must be in ascending order")

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

    def has_frequencies(self, frequencies: ndarray) -> bool:
        """Check if the audiogram has the given frequencies.

        Args:
            frequencies (ndarray): The frequencies to check

        Returns:
            bool: True if the audiogram has the given frequencies

        """
        return np.all(np.isin(frequencies, self.frequencies, assume_unique=True))

    def resample(
        self, new_frequencies: ndarray, linear_frequency: bool = False
    ) -> Audiogram:
        """Resample the audiogram to a new set of frequencies.

        Interpolates linearly on a (by default) log frequency axis. If
        linear_frequencies is set True then interpolation is done on a linear
        frequency axis.

        Args:
            new_frequencies (ndarray): The new frequencies to resample to

        Returns:
            Audiogram: New audiogram with resampled frequencies

        """

        # Either log frequency scaling or linear frequency scaling
        axis_fn: Callable = (lambda x: x) if linear_frequency else np.log

        return Audiogram(
            levels=np.interp(
                axis_fn(new_frequencies),
                axis_fn(self.frequencies),
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

# No loss
AUDIOGRAM_REF_CLARITY: Final = Audiogram(
    frequencies=DEFAULT_CLARITY_AUDIOGRAM_FREQUENCIES,
    levels=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
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


@dataclass
class Listener:
    """Dataclass to represent a Listener.

    The listener is currently defined by their left and right ear
    audiogram. In later versions, this may be extended to include
    further audiometric data.

    The class provides methods for reading metadata files which
    will also include some basic validation.

    Attributes:
        id (str): The ID of the listener
        audiogram_left (Audiogram): The audiogram for the left ear
        audiogram_right (Audiogram): The audiogram for the right ear
    """

    audiogram_left: Audiogram
    audiogram_right: Audiogram
    id: str = ""

    @staticmethod
    def from_dict(listener_dict: dict) -> Listener:
        """Create a Listener from a dict.

        The dict structure and fields are based on those used
        in the Clarity metadata files.

        Args:
            listener_dict (dict): The listener dict

        Returns:
            Listener: The listener

        """
        return Listener(
            id=listener_dict["name"],
            audiogram_left=Audiogram(
                levels=listener_dict["audiogram_levels_l"],
                frequencies=listener_dict["audiogram_cfs"],
            ),
            audiogram_right=Audiogram(
                levels=listener_dict["audiogram_levels_r"],
                frequencies=listener_dict["audiogram_cfs"],
            ),
        )

    @staticmethod
    def load_listener_dict(filename: Path) -> dict[str, Listener]:
        """Read a Clarity Listener dict file.

        The standard Clarity metadata files presents listeners as a
        dictionary of listeners, keyed by listener ID.

        Args:
            filename (Path): The path to the listener dict file

        Returns:
            dict[str, Listener]: A dict of listeners keyed by id

        """
        with open(filename, encoding="utf-8") as fp:
            listeners_raw = json.load(fp)

        listeners = {}
        for listener_id, listener_dict in listeners_raw.items():
            listeners[listener_id] = Listener.from_dict(listener_dict)
        return listeners
