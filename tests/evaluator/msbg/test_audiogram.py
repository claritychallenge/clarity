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


def test_audiogram_init():
    """test construction of audiogram"""
    levels = np.array([45, 45, 35, 45, 60, 65])
    frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])
    audiogram = Audiogram(levels=levels, frequencies=frequencies)
    assert np.all(audiogram.levels == levels)
    assert np.all(audiogram.frequencies == frequencies)
    assert isinstance(audiogram.levels, np.ndarray)


def test_audiogram_init_error():
    """test that cannot be constructed with invalid number of arguments"""
    # Need to disable pylint so that we can test the invalid constructor calls
    with pytest.raises(TypeError):
        Audiogram(None)  # pylint: disable=no-value-for-parameter
    with pytest.raises(ValueError):
        Audiogram([1])  # pylint: disable=no-value-for-parameter
    with pytest.raises(TypeError):
        Audiogram(3, 4, 5, 6)  # pylint: disable=too-many-function-args


@pytest.mark.parametrize(
    "levels, frequencies, expected",
    [
        ([45], [250], "NOTHING"),
        ([15], [2000], "NOTHING"),
        ([16], [2000], "MILD"),
        ([35], [2000], "MILD"),
        ([36], [2000], "MODERATE"),
        ([56], [2000], "MODERATE"),
        ([57], [2000], "SEVERE"),
        ([90], [8001], "NOTHING"),
        ([16, 16], [8000, 8001], "MILD"),
    ],
)
def test_severity(levels, frequencies, expected):
    """test severity calculation"""
    audiogram = Audiogram(levels=np.array(levels), frequencies=np.array(frequencies))
    assert audiogram.severity == expected


@pytest.mark.parametrize(
    "requested_frequencies, expected_levels",
    [
        (np.array([250, 1000.0, 6000.0]), np.array([1.0, 3.0, 6.0])),
        (np.array([260.0, 1000.0, 8000.0]), np.array([1.05658353, 3.0, 6.0])),
    ],
)
def test_resample(requested_frequencies, expected_levels):
    """test that a subset of cfs can be chosen"""
    levels = np.array([1, 2, 3, 4, 5, 6])
    frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])
    audiogram = Audiogram(levels=levels, frequencies=frequencies)

    # Include cfs that don't exist in the requested subset
    subset_audiogram = audiogram.resample(requested_frequencies)
    assert np.allclose(subset_audiogram.frequencies, requested_frequencies)
    assert np.allclose(subset_audiogram.levels, expected_levels)
