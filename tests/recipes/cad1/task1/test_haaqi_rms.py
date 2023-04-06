"""Test for the haaqi-rms module"""
import numpy as np
import pytest

from recipes.cad1.task1.baseline.haaqi_rms import (
    align_signals,
    compute_haaqi_rms,
    find_silence_segments,
)


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
    assert silence == [[4000, 7999], [12000, 15999]]
    assert non_silence == [[0, 3999], [8000, 11999]]


def test_compute_haaqi_rms():
    np.random.seed(0)
    duration = 0.5
    sample_rate = 1000
    sig_len = int(duration * sample_rate)
    reference_signal = np.concatenate(
        (
            100 * np.random.random(size=sig_len),
            np.zeros(sig_len),
            100 * np.random.random(size=sig_len),
            np.zeros(sig_len),
        )
    )
    processed_signal = reference_signal.copy()
    processed_signal[50:] = processed_signal[:-50]
    processed_signal[0:50] = 0

    processed_signal = align_signals(reference_signal, processed_signal)

    score = compute_haaqi_rms(
        processed_signal,
        reference_signal,
        audiogram=np.array([45, 45, 35, 45, 60, 65]),
        audiogram_frequencies=np.array([250, 500, 1000, 2000, 4000, 6000]),
        sample_rate=sample_rate,
        silence_length=0.1,
    )
    assert score == pytest.approx(
        0.131773752, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
