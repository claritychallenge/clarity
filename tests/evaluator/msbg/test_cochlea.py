"""Tests for cochlea module"""

import numpy as np
import pytest

from clarity.evaluator.msbg.cochlea import (
    Cochlea,
    FilterBank,
    compute_envelope,
    compute_recruitment_parameters,
    gammatone_filterbank,
    recruitment,
)
from clarity.utils.audiogram import Audiogram


def test_filterbank():
    """Test Filterbank class"""
    nums = np.array([4, 5, 6])
    denoms = np.array([1, 2, 3])
    filter_bank = FilterBank(nums=nums, denoms=denoms)
    assert np.all(filter_bank.nums == nums)
    assert np.all(filter_bank.denoms == denoms)


@pytest.mark.parametrize(
    "center_freqs, cf_expansion_expected",
    [
        (
            np.array([250, 500, 1000]),
            np.array([-0.0256410256410, -0.02189781021897, -0.019108280254]),
        ),
        (
            np.array([150, 500, 1000]),  # extrapolate lower
            np.array([-0.0256410256410, -0.02189781021897, -0.019108280254]),
        ),
        (
            np.array([250, 500, 2000]),  # extrapolate higher
            np.array([-0.0256410256410, -0.02189781021897, -0.01910828025477707]),
        ),
    ],
)
def test_compute_recruitment_parameters(center_freqs, cf_expansion_expected):
    """Test compute_recruitment_parameters"""
    audiogram_cfs = np.array([250, 500, 1000])
    levels = np.array([60, 70, 80])
    n_channels = len(levels)
    catch_up = 1.5
    audiogram = Audiogram(levels=levels, frequencies=audiogram_cfs)
    cf_expansion, eq_loud_db_catch_up = compute_recruitment_parameters(
        gtn_cf=center_freqs,
        audiogram=audiogram,
        catch_up=catch_up,
    )
    assert cf_expansion.shape == (n_channels,)
    assert eq_loud_db_catch_up.shape == (n_channels,)
    assert cf_expansion == pytest.approx(
        cf_expansion_expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert eq_loud_db_catch_up == pytest.approx(
        np.array([catch_up] * n_channels),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )


def test_gammatone_filterbank():
    """Test gammatone_filterbank"""

    np.random.seed(0)
    signal = np.random.random(1000)
    levels = np.array([30, 40, 40])
    audiogram_cfs = np.array([250, 500, 2000])
    audiogram = Audiogram(levels=levels, frequencies=audiogram_cfs)

    cochlea = Cochlea(audiogram=audiogram)

    coch_sig_out = gammatone_filterbank(
        x=signal,
        ngamma=cochlea.gtfbank_params["NGAMMA"],
        gtn_filters=FilterBank(
            cochlea.gtfbank_params["GTn_nums"], cochlea.gtfbank_params["GTn_denoms"]
        ),
        gtn_delays=cochlea.gtfbank_params["GTnDelays"],
        start2poleHP=cochlea.gtfbank_params["Start2PoleHP"],
        hp_filters=FilterBank(
            cochlea.gtfbank_params["HP_nums"], cochlea.gtfbank_params["HP_denoms"]
        ),
    )
    assert coch_sig_out.shape == (28, len(signal))
    assert np.sum(np.abs(coch_sig_out)) == pytest.approx(
        1327.2811434076052, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_compute_envelope():
    """Test compute_envelope"""
    np.random.seed(0)
    sample_rate = 44100
    # ERB spaced filter centre frequencies in Hz
    erbn_cf = np.array([250, 500, 1000])
    n_channels = len(erbn_cf)
    signal = np.random.random((n_channels, 1000))
    result = compute_envelope(coch_sig=signal, erbn_cf=erbn_cf, fs=sample_rate)
    assert result.shape == (n_channels, 1000)
    assert np.sum(np.abs(result)) == pytest.approx(
        1444.2528021105204, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_recruitment():
    """Test recruitment"""
    np.random.seed(0)
    sample_rate = 44100

    erbn_cf = np.array([250, 500, 1000])
    n_channels = len(erbn_cf)
    signal = np.random.random((n_channels, 1000))
    expansion_ratios = np.array([1.0, 1.0, 1.0])
    eq_loud_db = np.array([60.0, 70.0, 80.0])

    envelope = compute_envelope(coch_sig=signal, erbn_cf=erbn_cf, fs=sample_rate)
    result = recruitment(
        coch_sig=signal,
        envelope=envelope,
        SPL_equiv_0dB=100.0,
        expansion_ratios=expansion_ratios,
        eq_loud_db=eq_loud_db,
    )
    assert result.shape == (n_channels, 1000)
    assert np.sum(np.abs(result)) == pytest.approx(
        1512.7196632453372, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "levels, severity, has_smearer",
    [
        (np.array([30, 40, 10]), "NOTHING", False),
        (np.array([30, 40, 20]), "MILD", True),
        (np.array([30, 40, 40]), "MODERATE", True),
        (np.array([30, 40, 80]), "SEVERE", True),
    ],
)
def test_cochlea(levels, severity, has_smearer):
    """Test Cochlea class"""
    audiogram_cfs = np.array([250, 500, 2000])
    audiogram = Audiogram(levels=levels, frequencies=audiogram_cfs)
    assert audiogram.severity == severity

    cochlea = Cochlea(audiogram=audiogram)
    assert (cochlea.smearer is not None) == has_smearer


@pytest.mark.parametrize(
    "levels, expected, n_samples_out",
    [
        (np.array([30, 40, 10]), 140.40357187610562, 1000),
        (np.array([30, 40, 40]), 42.38705275894786, 1216),  # smearer adds samples
    ],
)
def test_cochlea_simulate(levels, expected, n_samples_out):
    """Test Cochlea class"""
    np.random.seed(0)
    signal = np.random.random(1000)
    audiogram_cfs = np.array([250, 500, 2000])
    audiogram = Audiogram(levels=levels, frequencies=audiogram_cfs)

    cochlea = Cochlea(audiogram=audiogram)
    result = cochlea.simulate(coch_sig=signal, equiv_0dB_file_SPL=100.0)
    assert result.shape == (n_samples_out,)
    assert np.sum(np.abs(result)) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
