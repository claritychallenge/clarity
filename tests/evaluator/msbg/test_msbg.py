"""Tests for msbg module"""

import numpy as np
import pytest

from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import DF_ED, FF_ED
from clarity.utils.audiogram import AUDIOGRAM_MODERATE_SEVERE, Audiogram


def test_ear():
    """Test Ear constructor"""
    ear = Ear()
    assert ear.cochlea is None
    assert ear.ahr == 20


def test_set_audiogram():
    """Test Ear.set_audiogram"""
    ear = Ear()
    ear.set_audiogram(audiogram=AUDIOGRAM_MODERATE_SEVERE)
    assert ear.cochlea is not None

    # Bad audiogram - will log a warning
    audiogram_bad = Audiogram(levels=np.array([150]), frequencies=np.array([4000]))
    ear.set_audiogram(audiogram=audiogram_bad)
    assert ear.cochlea is not None


def test_get_src_correction():
    """Test Ear.get_src_correction"""
    correction = Ear.get_src_correction("ff")
    assert np.all(correction == FF_ED)
    correction = Ear.get_src_correction("df")
    assert np.all(correction == DF_ED)
    correction = Ear.get_src_correction("ITU")
    assert np.sum(correction) == pytest.approx(
        215.69444085326438, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_get_src_correction_error():
    """Test Ear.get_src_correction with invalid src_pos"""
    with pytest.raises(ValueError):
        Ear.get_src_correction("bad")


def test_src_to_cochlea_filter():
    """Test Ear.src_to_cochlea_filter"""
    np.random.seed(0)
    ip_sig = np.random.rand(1000)
    signal = Ear.src_to_cochlea_filt(
        input_signal=ip_sig,
        src_correction=Ear.get_src_correction("ff"),
        sample_rate=16000,
        backward=False,
    )
    assert signal.shape == (1000,)
    assert np.sum(np.abs(signal)) == pytest.approx(
        285.70213562623803, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize("n_channels", [1, 2])
def test_make_calibration_signal(n_channels):
    """Test Ear.make_calibration_signal"""
    np.random.seed(0)
    ear = Ear()
    signal, silence = ear.make_calibration_signal(ref_rms_db=60, n_channels=n_channels)
    assert signal.shape == (
        n_channels,
        44100 * 2.65,
    )  # 2.65 s signal
    assert silence.shape == (
        n_channels,
        44100 * 0.05,
    )  # 50 ms silence
    assert np.sum(np.abs(silence)) == pytest.approx(
        0.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(signal)) == pytest.approx(
        90853246.9096005 * n_channels,
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )


def test_ear_process():
    """Test ear.process"""
    np.random.seed(0)

    signal = np.random.rand(500)

    ear = Ear()
    # Can process with audiogram
    processed_signals = ear.process(signal)
    assert len(processed_signals) == 1
    assert processed_signals[0].shape == (500,)  # same length as input signal
    assert np.sum(np.abs(processed_signals[0])) == pytest.approx(
        0.3250822358794586, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    ear.set_audiogram(audiogram=AUDIOGRAM_MODERATE_SEVERE)

    processed_signals = ear.process(signal)
    assert len(processed_signals) == 1
    assert processed_signals[0].shape == (704,)  # longer than input due to smearing
    assert np.sum(np.abs(processed_signals[0])) == pytest.approx(
        0.5448164192836198, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    signal2 = np.random.rand(500, 2)
    processed_signals = ear.process(signal2)
    assert len(processed_signals) == 2
    assert processed_signals[0].shape == (704,)
    assert processed_signals[1].shape == (704,)
    assert np.sum(np.abs(processed_signals[0])) == pytest.approx(
        0.4983800098691009, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(processed_signals[1])) == pytest.approx(
        0.514370533984153, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    signal1 = np.random.rand(500, 1)
    processed_signals = ear.process(signal1)
    assert len(processed_signals) == 1
    assert processed_signals[0].shape == (704,)
    assert np.sum(np.abs(processed_signals[0])) == pytest.approx(
        0.5840767608116147, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    signal = np.random.rand(500)
    processed_signals = ear.process(signal, add_calibration=True)
    assert len(processed_signals) == 1
    assert processed_signals[0].shape == (119808,)
    assert np.sum(np.abs(processed_signals[0])) == pytest.approx(
        48480.61370282146, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_ear_process_error():
    """Test ear.process with invalid sample frequency"""
    signal = np.random.rand(500)

    ear = Ear()

    # Try processing before audiogram is set
    ear.process(signal)

    # Try processing with invalid sample frequency
    ear.set_audiogram(audiogram=AUDIOGRAM_MODERATE_SEVERE)
    ear.sample_rate = 16000
    with pytest.raises(ValueError):
        ear.process(signal)
