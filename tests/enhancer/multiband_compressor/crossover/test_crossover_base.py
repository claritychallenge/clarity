""" Test the CrossoverBase Abstract class. """
import pytest
import numpy as np
from clarity.enhancer.multiband_compressor.crossover.crossover_base import CrossoverBase


@pytest.mark.parametrize(
    "freq_crossover",
    ([1000], [500, 1500]),
)
def test_initialization(freq_crossover):
    """Test the initialization of the CrossoverBase class."""
    sample_rate = 44100
    N = 4

    crossover = CrossoverBase(
        freq_crossover=freq_crossover, N=N, sample_rate=sample_rate
    )

    assert np.array_equal(crossover.xover_freqs, freq_crossover)
    assert crossover.order == N
    assert crossover.sample_rate == sample_rate


def test_invalid_order():
    """Test the initialization of the CrossoverBase class with an invalid order."""
    freq_crossover = 1000
    sample_rate = 44100
    invalid_N = 3

    with pytest.raises(
        ValueError, match="The order of the filter must be 4. 3 was provided."
    ):
        CrossoverBase(
            freq_crossover=freq_crossover, N=invalid_N, sample_rate=sample_rate
        )


def test_compute_coefficients_not_implemented():
    """Test the compute_coefficients method not implemented."""
    freq_crossover = 1000
    sample_rate = 44100
    N = 4

    crossover = CrossoverBase(
        freq_crossover=freq_crossover, N=N, sample_rate=sample_rate
    )

    with pytest.raises(
        NotImplementedError, match="This method must be implemented in the subclass."
    ):
        crossover.compute_coefficients()


def test_call_not_implemented():
    """Test the __call__ method not implemented."""
    freq_crossover = 1000
    sample_rate = 44100
    N = 4

    crossover = CrossoverBase(
        freq_crossover=freq_crossover, N=N, sample_rate=sample_rate
    )

    with pytest.raises(
        NotImplementedError, match="This method must be implemented in the subclass."
    ):
        crossover()
