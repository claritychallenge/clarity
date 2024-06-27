"""Tests for the CamfitGainTable class."""
import numpy as np
import pytest

from clarity.utils.audiogram import Audiogram
from recipes.cad2.common.gain_table import CamfitGainTable


@pytest.fixture
def audiogram_left():
    """Return a left audiogram"""
    return Audiogram(
        levels=np.array([30, 40, 50, 60, 70, 80, 90]),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 6000, 8000]),
    )


@pytest.fixture
def audiogram_right():
    """Return a right audiogram"""
    return Audiogram(
        levels=np.array([25, 35, 45, 55, 65, 75, 85]),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 6000, 8000]),
    )


@pytest.fixture
def camfit_gain_table():
    """Return a CamfitGainTable instance"""
    return CamfitGainTable(
        noisegate_levels=45,
        noisegate_slope=1.0,
        cr_level=0.0,
        max_output_level=100.0,
    )


def test_camfit_gain_table_initialization(camfit_gain_table):
    """Test the initialization of the CamfitGainTable class."""
    assert camfit_gain_table.noisegate_levels == 45
    assert camfit_gain_table.noisegate_slope == 1.0
    assert camfit_gain_table.cr_level == 0.0
    assert camfit_gain_table.max_output_level == 100.0
    np.testing.assert_array_equal(
        camfit_gain_table.interpolation_freqs,
        np.array([250, 500, 1000, 2000, 4000, 6000, 8000]),
    )


def test_process_with_interpolation(camfit_gain_table, audiogram_left, audiogram_right):
    """Test the process method with interpolation."""
    gain_table_left, gain_table_right = camfit_gain_table.process(
        audiogram_left, audiogram_right
    )
    # 121 levels and 7 interpolation frequencies
    assert gain_table_left.shape == (121, 7)
    assert gain_table_right.shape == (121, 7)
    assert np.sum(gain_table_left) == pytest.approx(
        6861.845020899, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(gain_table_right) == pytest.approx(
        5710.311368, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_process_without_interpolation(
    camfit_gain_table, audiogram_left, audiogram_right
):
    """Test the process method without interpolation."""
    gain_table_left, gain_table_right = camfit_gain_table.process(
        audiogram_left, audiogram_right, interpolate=False
    )
    assert gain_table_left.shape == (121, 9)  # 121 levels and 9 CAMFIT frequencies
    assert gain_table_right.shape == (121, 9)
    assert np.sum(gain_table_left) == pytest.approx(
        7338.2454156615, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(gain_table_right) == pytest.approx(
        5779.251299002, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_interpolate(camfit_gain_table):
    """Test the interpolate method."""
    # simulate a gain table with 121 levels and 9 CAMFIT frequencies
    np.random.seed(42)
    gain_table = np.random.rand(121, 9)
    interpolated_gain_table = camfit_gain_table.interpolate(gain_table)
    assert interpolated_gain_table.shape == (121, 7)
    assert np.sum(interpolated_gain_table) == pytest.approx(
        417.02357412757, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
