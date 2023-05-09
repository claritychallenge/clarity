"""Tests for gammatone_filters module"""
# pylint: disable=import-error
import numpy as np
import pytest

from clarity.evaluator.ha_metric.ear_model import GammatoneFilter


def test_gammatone_basilar_membrane():
    """Test gammatone basilar membrane"""
    np.random.seed(0)
    sig_len = 600
    ref = 100 * np.random.random(size=sig_len)
    proc = ref + 10 * np.random.random(size=sig_len)

    gamma_filter = GammatoneFilter(freq_sample=24000)
    (
        reference_envelope,
        reference_basilar_membrane,
    ) = gamma_filter.compute(
        signal=ref,
        bandwidth=1.4,
        center_freq=1000,
    )
    (
        processed_envelope,
        processed_basilar_membrane,
    ) = gamma_filter.compute(
        signal=proc,
        bandwidth=2.0,
        center_freq=1000,
    )

    # check shapes
    assert reference_envelope.shape == (600,)
    assert reference_basilar_membrane.shape == (600,)
    assert processed_envelope.shape == (600,)
    assert processed_basilar_membrane.shape == (600,)
    # check values
    assert np.sum(np.abs(reference_envelope)) == pytest.approx(
        3605.427313705984, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(reference_basilar_membrane)) == pytest.approx(
        2288.3557465, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(processed_envelope)) == pytest.approx(
        4426.111706599469, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(processed_basilar_membrane)) == pytest.approx(
        2804.93743475, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_gammatone_bandwidth_demodulation():
    """Test gammatone bandwidth demodulation"""
    gamma_filter = GammatoneFilter(24000.0)
    (
        centre_freq_sin,
        centre_freq_cos,
    ) = gamma_filter.gammatone_bandwidth_demodulation(
        npts=100,
        tpt=0.001,
        center_freq=1000,
    )
    assert centre_freq_sin.shape == (100,)
    assert centre_freq_cos.shape == (100,)
    assert np.sum(centre_freq_sin) == pytest.approx(
        -0.3791946274493412, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(centre_freq_cos) == pytest.approx(
        -0.39460748051808026, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
