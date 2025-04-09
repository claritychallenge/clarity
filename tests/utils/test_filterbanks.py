"""Test for filterbanks module"""

import numpy as np
import pytest

from clarity.utils.filterbanks import (
    Filterbank,
    Gammatone,
    gammatone_bandwidth_demodulation,
)


@pytest.fixture
def dummy_signal():
    np.random.seed(0)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)  # A4 sine wave
    return signal, sr


def test_gammatone_init():
    """Test init"""
    g = Gammatone(center_freq=1000, sample_rate=16000)
    assert g.center_freq == 1000
    assert g.sample_rate == 16000
    assert g.erb == pytest.approx(
        132.644732, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert g.center_freq_sin is None
    assert g.center_freq_cos is None


def test_gammatone_call_repeatable(dummy_signal):
    """Test gammatone filter"""
    signal, sr = dummy_signal
    g = Gammatone(center_freq=1000, sample_rate=sr)
    bm1, env1 = g(signal, bandwidth=1.0)
    bm2, env2 = g(signal, bandwidth=1.0)
    assert np.sum(bm1) == pytest.approx(
        np.sum(bm2), rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(env1) == pytest.approx(
        np.sum(env2), rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    assert bm1.shape == signal.shape
    assert env1.shape == signal.shape


def test_filterbank_init_valid():
    """test filterbank init"""
    center_freqs = [200, 400, 800]
    sr = 16000
    fb = Filterbank(center_freqs, sr)
    assert fb.n_filter == 3
    assert list(fb.filters.keys()) == sorted(center_freqs)


def test_filterbank_init_invalid_filter():
    """Test not implemented filter"""
    with pytest.raises(TypeError):
        Filterbank(500, 16000, filter_type="other_filter")


def test_filterbank_call_shape_float_bw(dummy_signal):
    """Test filterbank"""
    signal, sr = dummy_signal
    fb = Filterbank([300, 600, 1200], sr)
    bm, env = fb(signal, bandwidth=1.0)
    assert bm.shape == (3, len(signal))
    assert env.shape == (3, len(signal))
    assert np.sum(bm) == pytest.approx(
        0.210428, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(env) == pytest.approx(
        1354.7720728, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_gammatone_bandwidth_demodulation():
    """Test gammatone bandwidth demodulation"""
    centre_freq_sin, centre_freq_cos = gammatone_bandwidth_demodulation(
        npts=100,
        tpt=0.001,
        center_freq=1000,
        # center_freq_cos=np.zeros(100),
        # center_freq_sin=np.zeros(100),
    )
    assert centre_freq_sin.shape == (100,)
    assert centre_freq_cos.shape == (100,)
    assert np.sum(centre_freq_sin) == pytest.approx(
        -0.3791946274493412, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(centre_freq_cos) == pytest.approx(
        -0.39460748051808026, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
