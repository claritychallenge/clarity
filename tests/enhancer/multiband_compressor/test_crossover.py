""" Tests for the CrossoverTwoOrMore class """

import numpy as np
import pytest
from matplotlib import pyplot as plt

from clarity.enhancer.multiband_compressor.crossover import (
    Crossover,
    compute_coefficients,
)


@pytest.fixture
def random_signal():
    """Generate a random signal for testing."""
    return np.random.randn(1000)


@pytest.fixture
def single_freq_crossover():
    """Return a Crossover object with a single frequency."""
    xover_freqs = 1000
    return Crossover(xover_freqs)


@pytest.fixture
def multi_freq_crossover():
    """Return a Crossover object with multiple frequencies."""
    xover_freqs = [1000, 2000, 3000]
    return Crossover(xover_freqs)


@pytest.fixture
def complex_multi_freq_crossover():
    """Return a Crossover object with multiple frequencies."""
    xover_freqs = [250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
    return Crossover(xover_freqs)


def test_initialization_single_frequency(single_freq_crossover):
    """Test the initialization of the Crossover class with a single frequency."""
    assert single_freq_crossover.xover_freqs.shape[0] == 1


def test_initialization_multiple_frequencies(multi_freq_crossover):
    """Test the initialization of the Crossover class with multiple frequencies."""
    assert multi_freq_crossover.xover_freqs.shape[0] == 3


def test_compute_coefficients():
    """Test the compute_coefficients method of the Crossover class."""
    bstore, astore, bstore_phi, astore_phi = compute_coefficients(
        np.array([1000, 2000, 3000]), 8000, 4
    )
    # Check that the coefficients are not None
    assert np.sum(bstore) == pytest.approx(
        6.70591633, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(astore) == pytest.approx(
        13.411832670, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(bstore_phi) == pytest.approx(
        28.748181628, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(astore_phi) == pytest.approx(
        28.748181628, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_filter_application(multi_freq_crossover, random_signal):
    """Test the application of the filter to a signal."""
    filtered_signal = multi_freq_crossover(random_signal)
    # Check the shape of the filtered signal
    assert filtered_signal.shape == (
        len(multi_freq_crossover.xover_freqs) + 1,
        random_signal.shape[0],
    )


def test_filter_with_different_lengths(multi_freq_crossover):
    """Test the application of the filter to signals of different lengths."""
    for length in [100, 1000, 10000]:
        signal = np.random.randn(length)
        filtered_signal = multi_freq_crossover(signal)
        # Check the shape of the filtered signal
        assert filtered_signal.shape == (
            len(multi_freq_crossover.xover_freqs) + 1,
            length,
        )


def test_plot_filter(multi_freq_crossover, monkeypatch):
    """Test the plot_filter method of the CrossoverTwoOrMore class."""
    monkeypatch.setattr(plt, "show", lambda: None)
    # Check if plot_filter runs without error
    multi_freq_crossover.plot_filter()


def test_complex_multi_freq_crossover(complex_multi_freq_crossover):
    """Test the application of the filter to a signal."""
    signal = np.random.randn(1000)
    filtered_signal = complex_multi_freq_crossover(signal)
    # Check the shape of the filtered signal
    assert filtered_signal.shape == (
        len(complex_multi_freq_crossover.xover_freqs) + 1,
        signal.shape[0],
    )
