"""Class compute crossover filter for one crossover frequency."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from clarity.enhancer.multiband_compressor.crossover.crossover_base import (
    CrossoverBase,
)
from scipy.signal import butter, sosfreqz, sosfilt

import warnings


class CrossoverOne(CrossoverBase):
    def __init__(
        self,
        freq_crossover: int,
        N: int = 4,
        sample_rate: float = 44100,
    ) -> None:
        """Initialize the crossover filter.

        Args:
            freq_crossover (int): The crossover frequency (Hz).
            N (int): The order of the filter.
            sample_rate (float): The sample rate of the signal (Hz).
        """
        super().__init__(freq_crossover, N, sample_rate)
        if N % 2:
            raise ValueError("The order 'N' must be an even number.")

        if len(self.freq_crossover) > 1:
            raise ValueError("Only one crossover frequency is allowed.")

        self.sos = self.compute_coefficients()

    def compute_coefficients(self) -> np.ndarray:
        """Compute the filter coefficients."""
        # check if the order is even

        # order of Butterworth filters
        N = int(self.order / 2)

        # normalized frequency (half-cycle / per sample)
        freq = np.atleast_1d(np.asarray(self.freq_crossover)) / self.sample_rate * 2

        # init neutral SOS matrix of shape (freq.size+1, SOS_dim_2, 6)
        n_sos = int(np.ceil(N / 2))  # number of lowpass sos
        SOS_dim_2 = n_sos if freq.size == 1 else N
        SOS = np.tile(
            np.array([1, 0, 0, 1, 0, 0], dtype="float64"), (freq.size + 1, SOS_dim_2, 1)
        )

        # get filter coefficients for lowpass
        # (and bandpass if more than one frequency is provided)
        for n in range(freq.size):
            # get coefficients
            kind = "lowpass" if n == 0 else "bandpass"
            f = freq[n] if n == 0 else freq[n - 1 : n + 1]
            sos = butter(N, f, kind, analog=False, output="sos")
            # write to sos matrix
            if n == 0:
                SOS[n, 0:n_sos] = sos
            else:
                SOS[n] = sos

        # get filter coefficients for the highpass
        sos = butter(N, freq[-1], "highpass", analog=False, output="sos")
        # write to sos matrix
        SOS[-1, 0:n_sos] = sos

        # Apply every Butterworth filter twice
        SOS = np.tile(SOS, (1, 2, 1))

        # invert phase in every second channel if the Butterworth order is odd
        # (realized by reversing b-coefficients of the first sos)
        if N % 2:
            SOS[np.arange(1, freq.size + 1, 2), 0, 0:3] *= -1

        return SOS

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Return the filter coefficients."""
        return sosfilt(self.sos, signal)

    def __str__(self):
        return f"Crossover filter with crossover frequency: {self.freq_crossover[0]} Hz"

    def plot(self):
        """Plot the frequency response of the filter."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax = plt.subplots(1, 1)

            h_total = None
            for i in range(len(self.sos)):
                w, h = sosfreqz(self.sos[i])
                if h_total is None:
                    h_total = h
                else:
                    h_total += h

                text = "bandpass"
                if i == 0:
                    text = "Lowpass"
                elif i == len(self.sos) - 1:
                    text = "Highpass"

                ax.plot(w, 20 * np.log10(abs(h)), label=text)

            ax.plot(w, 20 * np.log10(abs(h_total)), label="Sum")
            ax.set_xscale("log")
            ax.legend(loc="lower right", fontsize=12)
            ax.text(
                0.05,
                0.95,
                f"Crossover frequency: {self.freq_crossover[0]} Hz",
                fontsize=15,
                transform=ax.transAxes,
                va="top",
            )
            plt.title("Crossover filter frequency response")
            plt.xlabel("Frequency [radians / second]")
            plt.ylabel("Amplitude [dB]")
            plt.margins(0, 0.1)
            plt.grid(which="both", axis="both")
            plt.ylim(-50, 10)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # test the function
    crosover = CrossoverOne(250, 4, 44100)
    crosover.plot()
    print(crosover)
