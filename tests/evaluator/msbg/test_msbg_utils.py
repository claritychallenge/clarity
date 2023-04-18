"""Tests for msbg_utils module"""
import numpy as np
import pytest

from clarity.evaluator.msbg.msbg_utils import (
    firwin2,
    gen_eh2008_speech_noise,
    gen_tone,
    measure_rms,
    pad,
    read_gtf_file,
)
from clarity.utils.file_io import read_signal, write_signal

# import scipy


def test_read_gtf_file():
    """Test reading of the gammatone filter file"""
    gtf = read_gtf_file("msbg_hparams/GT4FBank_Brd1.5E_Spaced1.1E_44100Fs.json")
    # Check the values are of expected types (NB all lists converted to ndarrays)
    for record in gtf:
        assert type(record) in {int, float, str, np.ndarray}
    # Check the gtf dict has the expected keys
    assert set(gtf.keys()) == {
        "__version__",
        "__globals__",
        "Fs",
        "BROADEN",
        "SPACING",
        "NGAMMA",
        "GTnDelays",
        "GTn_denoms",
        "GTn_nums",
        "GTn_CentFrq",
        "ERBn_CentFrq",
        "HP_denoms",
        "HP_nums",
        "HP_FCorner",
        "HP_Delays",
        "NChans",
        "Start2PoleHP",
        "Recombination_dB",
        "DateCreated",
    }


def test_firwin2():
    """Test firwin2"""
    n_taps = 128
    frequencies = np.array([0.0, 0.1, 0.9, 1.0])
    filter_gains = np.array([0, 1.0, 1.0, 0.0])
    params = firwin2(
        n_taps=n_taps,
        frequencies=frequencies,
        filter_gains=filter_gains,
        window=None,  # ("kaiser", 4),
    )
    # _params_scipy = scipy.signal.firwin2(
    #    numtaps=n_taps,
    #    freq=frequencies,
    #    gain=filter_gains,
    #    window=None,  # ("kaiser", 4)
    # )
    assert params.shape == (128,)
    assert np.sum(np.abs(params)) == pytest.approx(
        2.5662415502127844, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    # TODO: check why the test below fails
    # I thought this was meant to give something like the scipy version, but it doesn't
    # assert params == pytest.approx(params_scipy)


def test_gen_tone():
    """Test gen_tone"""
    signal = gen_tone(500, 0.1, 44100.0, 20.0)
    assert signal.shape == (4410,)
    assert np.sum(np.abs(signal)) == pytest.approx(
        39703.40087188665, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_gen_eh2008_speech_noise():
    """Test gen_eh2008_speech_noise"""
    np.random.seed(0)
    signal = gen_eh2008_speech_noise(0.1, 44100.0, 0.0)
    assert signal.shape == (4410,)
    assert np.sum(np.abs(signal)) == pytest.approx(
        3544.690935132768, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_measure_rms():
    """Test measure_rms"""
    np.random.seed(0)
    signal = gen_tone(500, 0.05, 44100.0, 20.0)
    noise = gen_eh2008_speech_noise(0.05, 44100.0, 0.0)
    rms, idx, rel_dB_thresh, active = measure_rms(signal + noise, 44100, 0.0, 10.0)
    assert rms == pytest.approx(
        10.723997044548266, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert idx.shape == (441,)
    assert np.sum(idx) == pytest.approx(
        97020, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert rel_dB_thresh == pytest.approx(
        -0.0008828901826447577, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert active == pytest.approx(
        20.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "initial, desired_length",
    [
        (np.array([1.0, 1.0, 1.0, 1.0]), 4),  # check can do no change
        (np.array([]), 100),  # check can pad empty array
    ],
)
def test_pad(initial, desired_length):
    """Test pad function"""
    result = pad(initial, desired_length)
    assert result.shape == (desired_length,)
    assert np.sum(np.abs(result)) == pytest.approx(
        np.sum(np.abs(initial)), rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_pad_error():
    """Test pad function fails when padded length is smaller than input"""
    with pytest.raises(ValueError):
        pad(np.array([1.0, 1.0, 1.0, 1.0]), 3)


@pytest.mark.parametrize(
    "signal, floating_point, strict",
    [
        (np.array([1.1, -2.0, 0.0, 44.0, -54.0]), True, True),  # float32
        (np.array([0.1, -0.2, 0.1, -0.5, 0.99]), False, True),  # int16
        (np.array([0.1, -0.2, 0.1, -1.00, 0.99]), False, True),  # int16
        (np.array([0.1, -0.2, 0.1, -1.00, 1.99]), False, False),  # int16
        (np.array([0.1, -0.2, 0.1, -1.00, 0.99]), False, True),  # int16
    ],
)
def test_write_read_loop(tmp_path, signal, floating_point, strict):
    """Test write_signal and read_signal"""
    tmp_filename = tmp_path / "test.wav"
    # signal = np.array([1.1, -2.0, 0.0, 44.0, -54.0])
    write_signal(
        tmp_filename, signal, 44100, floating_point=floating_point, strict=strict
    )
    result = read_signal(tmp_filename)

    # Some precision is lost as convert to int16 and back again
    assert result.shape == signal.shape
    # The test where strict is False has overflow which is not caught and hence
    # reading back the signal has changed
    if strict:
        assert result == pytest.approx(signal, abs=1.0 / 16384)
    else:
        # Deliberate fail: shows why strict is True is needed
        assert result != pytest.approx(signal, abs=1.0 / 16384)


def test_write_error(tmp_path):
    """Test write_signal and read_signal"""
    tmp_filename = tmp_path / "test.wav"
    signal = np.array([0.1, -0.1, 1.00])  # sample out of range
    with pytest.raises(ValueError):
        write_signal(tmp_filename, signal, 44100, floating_point=False, strict=True)
