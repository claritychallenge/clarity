"""Test CrossoverOne class."""
import numpy as np
import pytest
from matplotlib import pyplot as plt

from clarity.enhancer.multiband_compressor.crossover import CrossoverOne


def test_initialization():
    """Test initialization of the CrossoverOne class."""
    freq_crossover = 1000
    sample_rate = 16000
    N = 4

    crossover = CrossoverOne(
        freq_crossover=freq_crossover, N=N, sample_rate=sample_rate
    )

    assert crossover.xover_freqs[0] == freq_crossover
    assert crossover.order == N
    assert crossover.sample_rate == sample_rate
    assert crossover.sos is not None


def test_filtering():
    """Test filtering of the CrossoverOne class."""
    freq_crossover = 1000
    sample_rate = 16000
    N = 4

    crossover = CrossoverOne(
        freq_crossover=freq_crossover, N=N, sample_rate=sample_rate
    )

    np.random.seed(0)
    signal = np.random.randn(sample_rate)
    filtered_signal = crossover(signal)

    assert filtered_signal.shape == (2, sample_rate)
    assert np.any(filtered_signal[0])  # Lowpass should not be all zeros
    assert np.any(filtered_signal[1])  # Highpass should not be all zeros


def test_compute_coefficients():
    freq_crossover = 1000
    sample_rate = 44100
    N = 4

    crossover = CrossoverOne(
        freq_crossover=freq_crossover, N=N, sample_rate=sample_rate
    )
    sos = crossover.compute_coefficients()

    # Verify the shape of the SOS matrix
    expected_shape = (2, 2, 6)
    assert (
        sos.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {sos.shape}"

    # Verify the filter coefficients
    assert np.sum(sos[0, 0, :]) == pytest.approx(
        0.0368319878, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(sos[1, 0, :]) == pytest.approx(
        0.01841599390, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_plot_filter(monkeypatch):
    """Test plot_filter method of the CrossoverOne class."""
    monkeypatch.setattr(plt, "show", lambda: None)
    freq_crossover = 1000
    sample_rate = 16000
    N = 4

    crossover = CrossoverOne(
        freq_crossover=freq_crossover, N=N, sample_rate=sample_rate
    )

    # Check if plot_filter runs without error
    crossover.plot_filter()
