"""Test for the haaqi-rms module"""
# pylint: disable=import-error
import numpy as np
import pytest

from recipes.cad1.task1.baseline.haaqi_rms import align_signals, find_silence_segments


def test_align_signals():
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    processed_signal = reference_signal.copy()
    processed_signal[50:] = processed_signal[:-50]
    processed_signal[0:50] = 0

    proc = align_signals(reference_signal, processed_signal)

    assert proc.shape == (600,)
    assert np.sum(np.abs(reference_signal)) == pytest.approx(
        29892.167176853407, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(proc)) == pytest.approx(
        27199.009291096496, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_find_silence_segments():
    np.random.seed(0)
    duration = 0.5
    sample_rate = 8000
    sig_len = int(duration * sample_rate)
    reference_signal = np.concatenate(
        (
            100 * np.random.random(size=sig_len),
            np.zeros(sig_len),
            100 * np.random.random(size=sig_len),
            np.zeros(sig_len),
        )
    )

    silence_length = 0.1
    silence, non_silence = find_silence_segments(
        reference_signal, sample_rate, silence_length
    )
    print(silence, non_silence)
    assert silence == [[4000, 7999], [12000, 15999]]
    assert non_silence == [[0, 3999], [8000, 11999]]
