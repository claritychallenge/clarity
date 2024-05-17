""" Tests for the CrossoverTwoOrMore class """

import numpy as np
import pytest
from matplotlib import pyplot as plt

# Adjust the import based on your actual module structure
from clarity.enhancer.multiband_compressor.crossover.crossover_two import (
    CrossoverTwoOrMore,
)


def test_initialization():
    """Test the initialization of the CrossoverTwoOrMore class"""
    xover_freqs = [500, 1000, 2000, 4000, 8000]
    sample_rate = 44100
    N = 4

    crossover = CrossoverTwoOrMore(
        freq_crossover=xover_freqs, N=N, sample_rate=sample_rate
    )

    assert np.array_equal(crossover.xover_freqs, xover_freqs)
    assert crossover.order == N
    assert crossover.sample_rate == sample_rate


def test_compute_coefficients():
    """Test the compute_coefficients method of the CrossoverTwoOrMore class"""
    xover_freqs = [500, 1000, 2000, 4000, 8000]
    sample_rate = 44100
    N = 4

    crossover = CrossoverTwoOrMore(
        freq_crossover=xover_freqs, N=N, sample_rate=sample_rate
    )
    crossover.compute_coefficients()

    # Check if bstore and astore have been initialized correctly
    assert crossover.bstore is not None
    assert crossover.astore is not None
    assert crossover.bstore.shape == (N + 1, len(xover_freqs), 2)
    assert crossover.astore.shape == (N + 1, len(xover_freqs), 2)
    assert crossover.bstore_phi.shape == (2 * N + 1, len(xover_freqs))
    assert crossover.astore_phi.shape == (2 * N + 1, len(xover_freqs))


def test_filtering():
    """Test the filtering of the CrossoverTwoOrMore class"""
    xover_freqs = [500, 1000, 2000, 4000, 8000]
    sample_rate = 44100
    N = 4

    crossover = CrossoverTwoOrMore(
        freq_crossover=xover_freqs, N=N, sample_rate=sample_rate
    )

    np.random.seed(0)
    signal = np.random.randn(1000)
    filtered_signal = crossover(signal)

    assert filtered_signal.shape == (len(xover_freqs) + 1, len(signal))


def test_plot_filter(monkeypatch):
    """Test the plot_filter method of the CrossoverTwoOrMore class."""
    monkeypatch.setattr(plt, "show", lambda: None)
    xover_freqs = [500, 1000, 2000, 4000, 8000]
    sample_rate = 44100
    N = 4

    crossover = CrossoverTwoOrMore(
        freq_crossover=xover_freqs, N=N, sample_rate=sample_rate
    )

    # Check if plot_filter runs without error
    crossover.plot_filter()


def test_xover_methods():
    xover_freqs = np.array([250, 500, 1000, 1500, 2000]) * np.sqrt(2)
    sample_rate = 16000
    N = 4

    crossover = CrossoverTwoOrMore(
        freq_crossover=xover_freqs, N=N, sample_rate=sample_rate
    )

    np.random.seed(0)
    signal = np.random.randn(1000)

    filtered_1 = crossover.xover_1(signal)
    filtered_2 = crossover.xover_2(signal)
    filtered_3 = crossover.xover_3(signal)
    filtered_4 = crossover.xover_4(signal)
    filtered_5 = crossover.xover_5(signal)
    filtered_6 = crossover.xover_6(signal)

    assert filtered_1.shape == signal.shape
    assert filtered_2.shape == signal.shape
    assert filtered_3.shape == signal.shape
    assert filtered_4.shape == signal.shape
    assert filtered_5.shape == signal.shape
    assert filtered_6.shape == signal.shape

    # Basic checks to ensure the signals are filtered differently
    assert np.sum(filtered_1) == pytest.approx(
        -52.49701687, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(filtered_2) == pytest.approx(
        -0.36784226, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(filtered_3) == pytest.approx(
        -0.43473953, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(filtered_4) == pytest.approx(
        0.049283965, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(filtered_5) == pytest.approx(
        0.1732059956, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(filtered_6) == pytest.approx(
        0.32794628, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
