"""Tests for enhancer.gha.audiogram module"""

import numpy as np
import pytest

from clarity.enhancer.gha.audiogram import Audiogram


def test_audiogram_init():
    """test construction of audiogram"""
    levels_l = np.array([45, 45, 35, 45, 60, 65])
    levels_r = np.array([45, 45, 35, 45, 60, 65])
    cfs = np.array([250, 500, 1000, 2000, 4000, 6000])
    audiogram = Audiogram(levels_l=levels_l, levels_r=levels_r, cfs=cfs)
    assert np.all(audiogram.levels_l == levels_l)
    assert np.all(audiogram.levels_r == levels_r)
    assert np.all(audiogram.cfs == cfs)
    assert isinstance(audiogram.levels_l, np.ndarray)


def test_audiogram_init_error():
    """test that cannot be constructed with invalid number of arguments"""
    # Need to disable pylint so that we can test the invalid constructor calls
    with pytest.raises(TypeError):
        Audiogram()  # pylint: disable=no-value-for-parameter
    with pytest.raises(TypeError):
        Audiogram([1])  # pylint: disable=no-value-for-parameter
    with pytest.raises(TypeError):
        Audiogram(3, 4, 5, 6)  # pylint: disable=too-many-function-args


@pytest.mark.parametrize(
    "left, right, cfs, expected",
    [
        ([45], [45], [250], ["NOTHING", "NOTHING"]),
        ([15], [16], [2000], ["NOTHING", "MILD"]),
        ([35], [36], [2000], ["MILD", "MODERATE"]),
        ([56], [57], [2000], ["MODERATE", "SEVERE"]),
        ([90], [90], [8001], ["NOTHING", "NOTHING"]),
        ([90, 90], [16, 16], [8000, 8001], ["SEVERE", "MILD"]),
    ],
)
def test_severity(left, right, cfs, expected):
    """test severity calculation"""
    audiogram = Audiogram(
        levels_l=np.array(left), levels_r=np.array(right), cfs=np.array(cfs)
    )
    assert audiogram.severity == expected


def test_select_subset_of_cfs():
    """test that a subset of cfs can be chosen"""
    levels_l = np.array([1, 2, 3, 4, 5, 6])
    levels_r = np.array([11, 12, 13, 14, 15, 16])
    cfs = np.array([250, 500, 1000, 2000, 4000, 6000])
    audiogram = Audiogram(levels_l=levels_l, levels_r=levels_r, cfs=cfs)

    # Select a subset of cfs
    subset_audiogram = audiogram.select_subset_of_cfs([250, 1000, 6000])
    assert np.all(subset_audiogram.levels_l == np.array([1, 3, 6]))
    assert np.all(subset_audiogram.levels_r == np.array([11, 13, 16]))
    assert np.all(subset_audiogram.cfs == np.array([250, 1000, 6000]))

    # Include cfs that don't exist in the requested subset
    subset_audiogram = audiogram.select_subset_of_cfs([251, 1000, 6001])
    assert np.all(subset_audiogram.levels_l == np.array([3]))
    assert np.all(subset_audiogram.levels_r == np.array([13]))
    assert np.all(subset_audiogram.cfs == np.array([1000]))

    # Include only a cf that doesn't exist in the requested subset
    subset_audiogram = audiogram.select_subset_of_cfs([999])
    assert np.all(subset_audiogram.levels_l == np.array([]))
    assert np.all(subset_audiogram.levels_r == np.array([]))
    assert np.all(subset_audiogram.cfs == np.array([]))
    # Severity of an 'empty' audiogram should be NOTHING
    assert subset_audiogram.severity == ["NOTHING", "NOTHING"]
