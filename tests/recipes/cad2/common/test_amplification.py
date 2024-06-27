"""Tests for the hearing aid amplification module."""
import numpy as np
import pytest

from clarity.enhancer.multiband_compressor import MultibandCompressor
from clarity.utils.audiogram import Listener
from recipes.cad2.common.amplification import HearingAid
from recipes.cad2.common.gain_table import CamfitGainTable


@pytest.fixture
def compressor_params():
    """Parameters for the multiband compressor."""
    return {
        "crossover_frequencies": np.array([250, 500, 2000, 3000, 4000]) / np.sqrt(2),
        "sample_rate": 16000,
    }


@pytest.fixture
def gain_table_params():
    """Parameters for the gain table."""
    return {
        "noisegate_levels": 40,
        "noisegate_slope": 0,
        "cr_level": 0,
        "max_output_level": 100,
    }


@pytest.fixture
def hearing_aid(compressor_params, gain_table_params):
    """Hearing aid instance."""
    return HearingAid(compressor_params, gain_table_params)


@pytest.fixture
def listener():
    """Listener audiogram."""
    return Listener.from_dict(
        {
            "name": "listener_14",
            "audiogram_cfs": [250, 500, 2000, 3000, 4000, 8000],
            "audiogram_levels_l": [30, 20, 40, 50, 90, 75],
            "audiogram_levels_r": [25, 25, 30, 30, 50, 60],
        }
    )


def test_hearing_aid_initialization(hearing_aid):
    """Test hearing aid initialization."""
    assert isinstance(hearing_aid.compressor, MultibandCompressor)
    assert isinstance(hearing_aid.gain_table, CamfitGainTable)


def test_set_compressors(hearing_aid, listener):
    """Test setting compressors."""
    hearing_aid.set_compressors(listener)
    assert len(hearing_aid.mbc) == 2  # Expecting compressors for both ears


def test_compute_bill_params():
    """Test computing parameters."""
    gain_table = {
        65: np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 10.0, 10.0]),
        75: np.array([20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 10.0, 10.0, 10.0]),
        95: np.array([30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 10.0, 10.0, 10.0]),
    }
    gain, cr = HearingAid.compute_bill_params(gain_table)
    assert gain.shape == (6,)
    assert cr.shape == (6,)
    assert np.sum(gain) == pytest.approx(
        124.93704061, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(cr) == pytest.approx(
        5.2, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_call(hearing_aid, listener):
    """Test calling the hearing aid."""
    np.random.seed(42)
    hearing_aid.set_compressors(listener)
    signal = np.random.randn(500)
    enhanced_signal = hearing_aid(signal[np.newaxis, :])
    assert enhanced_signal.shape == signal[np.newaxis, :].shape
    assert np.sum(enhanced_signal) == pytest.approx(
        19.016856471, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
