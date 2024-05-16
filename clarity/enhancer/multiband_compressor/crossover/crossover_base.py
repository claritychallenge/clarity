"""Abstract Class for Crossover Filters."""
from __future__ import annotations
import numpy as np


class CrossoverBase:
    def __init__(
        self,
        freq_crossover: list | int,
        N: int = 4,
        sample_rate: float = 44100,
    ) -> None:
        """
        Initialize the crossover filter.

        Args:
            freq_crossover (list | int): The crossover frequencies (Hz).
            N (int): The order of the filter.
            sample_rate (float): The sample rate of the signal (Hz).
        """
        if isinstance(freq_crossover, int):
            freq_crossover = [freq_crossover]

        if N != 4:
            raise ValueError(f"The order of the filter must be 4. {N} was provided.")

        self.xover_freqs = np.array(freq_crossover)
        self.order = N

        self.sample_rate = sample_rate

    def compute_coefficients(self) -> None:
        """Compute the filter coefficients."""
        raise NotImplementedError("This method must be implemented in the subclass.")

    def __call__(self, *args, **kwargs):
        """Method to call the filter."""
        raise NotImplementedError("This method must be implemented in the subclass.")

    def plot_filter(self):
        """Method to plot the filter response."""
        pass
