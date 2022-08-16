"""Supplied dataclass to represent a monaural audiogram"""

from dataclasses import dataclass

import numpy as np

DEFAULT_CLARITY_CFS = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]

FULL_STANDARD_CFS = np.array(
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
        levels (ndarray): The levels for the left and right ear
        cfs (ndarray): The centre-frequencies at which the levels are measured
    """

    levels: np.ndarray
    cfs: np.ndarray = np.array(DEFAULT_CLARITY_CFS)

    @property
    def severity(self):
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
        # calculate mean hearing loss between 2 & 8 kHz
        impairment_freqs = np.logical_and(2000 <= self.cfs, self.cfs <= 8000)
        tmp = self.levels[impairment_freqs]
        # Ignore any None values
        impairment_degree = np.mean(tmp[tmp is not None])

        if impairment_degree > 56:
            severity_level = "SEVERE"
        elif impairment_degree > 35:
            severity_level = "MODERATE"
        elif impairment_degree > 15:
            severity_level = "MILD"
        else:
            severity_level = "NOTHING"
        return severity_level


# Reference processing: use to check levels between original and processed,
AUDIOGRAM_REF = Audiogram(
    cfs=FULL_STANDARD_CFS,
    levels=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
)

# mild age-related hearing loss, slightly reduced from CF, first-time aid wearers group
# (used in Stafa talk by MAS)
AUDIOGRAM_MILD = Audiogram(
    cfs=FULL_STANDARD_CFS,
    levels=np.array([5, 10, 15, 18, 19, 22, 25, 28, 31, 35, 38, 40, 40, 45, 50]),
)

# mod hearing loss based on mild N2 flat/mod sloping from Bisgaard et al. 2020
AUDIOGRAM_MODERATE = Audiogram(
    cfs=FULL_STANDARD_CFS,
    levels=np.array([15, 20, 20, 22.5, 25, 30, 35, 40, 45, 50, 55, 55, 60, 65, 65]),
)

# (mod-)severe age-related hearing loss, average of MAS/KA summer proj 2011, elderly HI
AUDIOGRAM_MODERATE_SEVERE = Audiogram(
    cfs=FULL_STANDARD_CFS,
    levels=np.array([19, 19, 28, 35, 40, 47, 52, 56, 58, 58, 63, 70, 75, 80, 80]),
)

if __name__ == "__main__":
    # test code
    audiogram = Audiogram(
        cfs=np.array([1000, 3000, 10000]), levels=np.array([10, 50, 80])
    )
    print(audiogram, audiogram.severity)
    print(Audiogram(cfs=np.array([3000]), levels=np.array([10])).severity)
    print(Audiogram(cfs=np.array([3000]), levels=np.array([20])).severity)
    print(Audiogram(cfs=np.array([3000]), levels=np.array([40])).severity)
    print(Audiogram(cfs=np.array([3000]), levels=np.array([90])).severity)
