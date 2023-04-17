"""Tests for enhancer.gha.gha_utils module"""
import hashlib

import numpy as np
import pytest

from clarity.enhancer.gha.gha_utils import (
    format_gaintable,
    get_gaintable,
    multifit_apply_noisegate,
)
from clarity.utils.audiogram import Audiogram


@pytest.fixture
def gaintable():
    """fixture for gaintable"""
    np.set_printoptions(threshold=1000)
    np.random.seed(0)
    levels = np.array([45, 45, 35, 45, 60, 65])
    frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])
    audiogram = Audiogram(levels=levels, frequencies=frequencies)
    noisegate_levels = np.array([38, 38, 36, 37, 32, 26, 23, 22, 8])
    noisegate_slope = 0.0
    cr_level = 0.0
    max_output_level = 100.0
    this_gaintable = get_gaintable(
        audiogram_left=audiogram,
        audiogram_right=audiogram,
        noisegate_levels=noisegate_levels,
        noisegate_slope=noisegate_slope,
        cr_level=cr_level,
        max_output_level=max_output_level,
    )
    return this_gaintable


def test_get_gaintable(gaintable):
    """test that the gaintable is generated correctly"""

    # check that the gaintable has the correct keys
    assert gaintable.keys() == {
        "sGt_uncorr",
        "sGt",
        "noisegatelevel",
        "noisegateslope",
        "frequencies",
        "levels",
        "channels",
    }
    # check that the gaintable has the expected values
    assert np.sum(np.array(gaintable["sGt_uncorr"])) == pytest.approx(
        52476.611733926344, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.array(gaintable["sGt"])) == pytest.approx(
        45866.4547337617, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_format_gaintable(gaintable):
    """test that the gaintable is formatted correctly"""

    # check that the formatted gaintable string has the expected value
    formatted = format_gaintable(gaintable, noisegate_corr=True)
    assert (
        hashlib.md5(formatted.encode("utf-8")).hexdigest()
        == "d6e6997fd9f1bcaf690d7e6a36813bde"
    )

    formatted = format_gaintable(gaintable, noisegate_corr=False)
    assert (
        hashlib.md5(formatted.encode("utf-8")).hexdigest()
        == "af02f7ca070ff61a51a1f8572e381818"
    )


def test_multifit_apply_noisegate(gaintable):
    """test that the noise gate is applied correctly"""
    INITIAL_SGT_SUM = 52476.611733926344
    FINAL_SGT_SUM = 45866.4547337617
    sGt = gaintable["sGt_uncorr"]
    assert np.sum(sGt) == pytest.approx(
        INITIAL_SGT_SUM, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    sFit_model_frequencies = gaintable["frequencies"]
    sFit_model_levels = gaintable["levels"]
    noisegate_levels = gaintable["noisegatelevel"]
    noisegate_slope = gaintable["noisegateslope"]
    corrected_sGt = multifit_apply_noisegate(
        sGt,
        sFit_model_frequencies,
        sFit_model_levels,
        noisegate_levels,
        noisegate_slope,
    )

    assert np.sum(sGt) == pytest.approx(
        INITIAL_SGT_SUM, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(corrected_sGt) == pytest.approx(
        FINAL_SGT_SUM, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
