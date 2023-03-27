"""Tests for smearing module"""
import numpy as np
import pytest

from clarity.evaluator.msbg.smearing import Smearer, make_smear_mat3

# audfilt

# make_smear_mat3

# smear3

# Smearer


def test_change_filter():
    """Test changing filter broadening factors and sampling frequency"""
    smearer = Smearer(0.5, 1.5, 44100)
    f_smear_old = smearer.f_smear
    smearer.rl = 0.8
    smearer.ru = 1.2
    smearer.fs = 22050
    smearer.f_smear = make_smear_mat3(smearer.rl, smearer.ru, smearer.fs)
    assert isinstance(smearer.f_smear, np.ndarray)
    assert smearer.f_smear.shape == (256, 256)
    assert not np.array_equal(smearer.f_smear, f_smear_old)


def test_smear_valid_input():
    """Test smear with valid input"""
    np.random.seed(0)
    r_lower = 0.5
    r_upper = 1.5
    sample_freq = 44100
    input_signal = np.random.rand(10000)
    output_signal = Smearer(r_lower, r_upper, sample_freq).smear(input_signal)
    assert output_signal.shape == (10240,)
    assert np.sum(np.abs(output_signal)) == pytest.approx(
        5066.986397433977, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_make_smear_mat3_valid_input():
    """Tests that make_smear_mat3 returns matrix with the correct dimensions"""
    r_lower = 0.5
    r_upper = 1.5
    sample_freq = 44100
    f_smear = make_smear_mat3(rl=r_lower, ru=r_upper, fs=sample_freq)
    assert f_smear.shape == (256, 256)
