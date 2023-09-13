"""Tests for audiogram module"""

from pathlib import Path

import numpy as np
import pytest

from clarity.utils.audiogram import (
    AUDIOGRAM_MILD,
    AUDIOGRAM_MODERATE,
    AUDIOGRAM_MODERATE_SEVERE,
    AUDIOGRAM_REF,
    Audiogram,
    Listener,
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
    "test_freqs, result",
    [
        (np.array([]), True),
        (np.array([250, 1000]), True),
        (np.array([250, 500, 1000]), True),
        (np.array([250, 500, 1000, 2000]), False),
        (np.array([2000]), False),
    ],
)
def test_has_frquencies(test_freqs, result):
    """test that frequencies are correctly identified"""
    audiogram = Audiogram(
        levels=np.array([45, 45, 35]), frequencies=np.array([250, 500, 1000])
    )
    assert audiogram.has_frequencies(test_freqs) == result


@pytest.mark.parametrize(
    "frequencies, levels",
    [
        (np.array([250.0, 250.0, 500.0]), np.array([45, 45, 35])),  # duplicate freqs
        (np.array([250.0, 500.0]), np.array([45, 45, 35])),  # different lengths
        (np.array([250.0, 500.0, 10.0]), np.array([45, 45, 35])),  # freqs not ordered
    ],
)
def test_audiogram_invalid(frequencies, levels):
    """test that invalid audiograms raise ValueError"""
    with pytest.raises(ValueError):
        Audiogram(levels=levels, frequencies=frequencies)


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
    assert subset_audiogram.frequencies == pytest.approx(
        requested_frequencies, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert subset_audiogram.levels == pytest.approx(
        expected_levels, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_listener_read_listener_dict():
    """Test reading listener dictionary"""
    listener_dict_path = Path("tests/test_data/metadata/listeners.json")
    listeners_dict = Listener.load_listener_dict(listener_dict_path)

    # Check the dataclass and underlying audiogram dataclass have
    # been constructed correctly and that a sample of values
    # match the file contents
    assert len(listeners_dict) == 83
    listener = listeners_dict["L0100"]
    assert list(listener.__dict__.keys()) == ["audiogram_left", "audiogram_right", "id"]
    assert listener.id == "L0100"
    assert np.all(
        listener.audiogram_left.frequencies
        == np.array(
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
    )
    assert np.all(
        listener.audiogram_right.levels == np.array([20, 40, 55, 40, 20, 15, 5, -5])
    )
