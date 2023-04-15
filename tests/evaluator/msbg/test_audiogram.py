"""Tests for audiogram module"""

import numpy as np
import pytest

from clarity.evaluator.msbg.audiogram import (
    AUDIOGRAM_MILD,
    AUDIOGRAM_MODERATE,
    AUDIOGRAM_MODERATE_SEVERE,
    AUDIOGRAM_REF,
    Audiogram,
)


def test_audiogram():
    """Test Audiogram class"""
    levels = np.array([60, 70, 80])
    cfs = np.array([250, 500, 1000])
    audiogram = Audiogram(levels=levels, frequencies=cfs)
    assert np.all(audiogram.levels == levels)
    assert np.all(audiogram.frequencies == cfs)


@pytest.mark.parametrize(
    "cfs, levels, severity",
    [
        (np.array([500, 1000, 2000, 3000]), np.array([60, 70, 10, 10]), "NOTHING"),
        (np.array([500, 1000, 2000, 3000]), np.array([60, 70, 20, 20]), "MILD"),
        (np.array([500, 1000, 2000, 3000]), np.array([60, 70, 40, 40]), "MODERATE"),
        (np.array([500, 1000, 2000, 3000]), np.array([60, 70, 80, 80]), "SEVERE"),
        (np.array([500, 800, 900, 1000]), np.array([60, 70, 80, 90]), "NOTHING"),
        (np.array([9000, 10000]), np.array([120, 120]), "NOTHING"),
    ],
)
def test_audiogram_severity(cfs, levels, severity):
    """Test Audiogram severity calculation"""
    assert Audiogram(levels=levels, frequencies=cfs).severity == severity


def test_standard_audiograms_are_correct_severity():
    """Test standard audiogram"""
    assert AUDIOGRAM_REF.severity == "NOTHING"
    assert AUDIOGRAM_MILD.severity == "MILD"
    assert AUDIOGRAM_MODERATE.severity == "MODERATE"
    assert AUDIOGRAM_MODERATE_SEVERE.severity == "SEVERE"
