from dataclasses import dataclass

import numpy as np


@dataclass
class Audiogram:
    """Dataclass to represent an audiogram."""

    levels_l: np.ndarray
    levels_r: np.ndarray
    cfs: np.ndarray

    @property
    def severity(self):
        """Categorise HL severity level for the audiogram.

        Note that this categorisation is different from that of the British Society of
        Audiology, which recommends descriptors mild, moderate, severe
        and profound for average hearing threshold levels at 250, 500,
        1000, 2000 and 4000 Hz of 21-40 dB HL, 41-70 dB HL, 71-95 dB HL
        and > 95 dB HL, respectively (BSA Pure-tone air-conduction and
        bone-conduction threshold audiometry with and without masking
        2018).

        Returns:
            str: severity level, one of SEVERE, MODERATE, MILD, NOTHING
                for each ear [left, right]

        """
        # calculate mean hearing loss between 2 & 8 kHz
        impairment_freqs = np.logical_and(2000 <= self.cfs, self.cfs <= 8000)
        severity_levels = [None, None]
        for i, levels in enumerate([self.levels_l, self.levels_r]):
            impairment_degree = np.mean(levels[impairment_freqs])
            if impairment_degree > 56:
                severity_level = "SEVERE"
            elif impairment_degree > 35:
                severity_level = "MODERATE"
            elif impairment_degree > 15:
                severity_level = "MILD"
            else:
                severity_level = "NOTHING"
            severity_levels[i] = severity_level
        return severity_levels

    def select_subset_of_cfs(self, selected_cfs):
        """Make a new audiogram using a given subset of centre freqs.

        Note, any selected_cfs that do not exist in the original audiogram
        are simply ignored

        Args:
            selected_cfs (list): List of centre frequencies to include

        Returns:
            Audiogram: New audiogram with reduced set of frequencies

        """
        indices = [i for i, cf in enumerate(self.cfs) if cf in selected_cfs]
        return Audiogram(
            levels_l=self.levels_l[indices],
            levels_r=self.levels_r[indices],
            cfs=self.cfs[indices],
        )
