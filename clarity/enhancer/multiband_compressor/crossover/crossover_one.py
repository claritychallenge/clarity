"""Class compute crossover filter for one crossover frequency."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz

from clarity.enhancer.multiband_compressor.crossover.crossover_base import CrossoverBase


class CrossoverOne(CrossoverBase):
    """Class compute crossover filter for one crossover frequency.
    This is based on the pyFar implementation.

    https://pyfar.readthedocs.io/en/v0.1.0/pyfar.dsp.filter.html
    """

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

        if len(self.xover_freqs) > 1:
            raise ValueError("Only one crossover frequency is allowed.")

        self.sos = self.compute_coefficients()

    def compute_coefficients(self) -> np.ndarray:
        """Compute the filter coefficients."""
        # check if the order is even

        # order of Butterworth filters
        N = int(self.order / 2)

        # normalized frequency (half-cycle / per sample)
        freq = np.atleast_1d(np.asarray(self.xover_freqs)) / self.sample_rate * 2

        # init neutral SOS matrix of shape (freq.size+1, SOS_dim_2, 6)
        n_sos = int(np.ceil(N / 2))  # number of lowpass sos
        SOS_dim_2 = n_sos if freq.size == 1 else N
        SOS = np.tile(
            np.array([1, 0, 0, 1, 0, 0], dtype="float64"), (freq.size + 1, SOS_dim_2, 1)
        )

        # get filter coefficients for lowpass
        sos = butter(N, freq[0], "lowpass", analog=False, output="sos")
        SOS[0, 0:n_sos] = sos

        # get filter coefficients for the highpass
        sos = butter(N, freq[0], "highpass", analog=False, output="sos")
        SOS[1, 0:n_sos] = sos

        # Apply every Butterworth filter twice
        SOS = np.tile(SOS, (1, 2, 1))

        # invert phase in every second channel if the Butterworth order is odd
        # (realized by reversing b-coefficients of the first sos)
        if N % 2:
            SOS[np.arange(1, freq.size + 1, 2), 0, 0:3] *= -1

        return SOS

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply the filter to the signal.

        Args:
            signal (np.ndarray): The input signal.

        Returns:
            np.ndarray: The filtered signal. Shape (2, len(signal))
            with the first row being the lowpass and the second row the highpass.
        """
        if signal.ndim > 1:
            raise ValueError("Only 1D signals are supported.")

        lowpass_signal = sosfilt(self.sos[0, 0], signal)
        lowpass_signal = sosfilt(self.sos[0, 1], lowpass_signal)

        highpass_signal = sosfilt(self.sos[1, 0], signal)
        highpass_signal = sosfilt(self.sos[1, 1], highpass_signal)

        return np.vstack([lowpass_signal, highpass_signal])

    def __str__(self):
        return f"Crossover filter with crossover frequency: {self.xover_freqs[0]} Hz"

    def plot_filter(self):
        """Method to plot the frequency response of the filter.
        This can help to validate the Class is generating the expected filters
        """
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

                text = "Highpass"
                if i == 0:
                    text = "Lowpass"

                ax.plot(w, 20 * np.log10(abs(h)), label=text)

            ax.plot(w, 20 * np.log10(abs(h_total)), label="Sum")
            ax.set_xscale("log")
            ax.legend(loc="lower right", fontsize=12)
            ax.text(
                0.05,
                0.95,
                f"Crossover frequency: {self.xover_freqs[0]} Hz",
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
    import librosa

    signal, sr = librosa.load(librosa.ex("choice"), sr=None)

    crosover = CrossoverOne(1000, 4, sr)
    crosover.plot_filter()
    filtered_signal = crosover(signal)
    fig, ax = plt.subplots(3, 1)
    ax[0].specgram(signal, Fs=sr)
    ax[1].specgram(filtered_signal[0], Fs=sr)
    ax[2].specgram(filtered_signal[1], Fs=sr)
    ax[0].set_title("Original signal")
    ax[1].set_title("Lowpass")
    ax[2].set_title("Highpass")
    plt.tight_layout()
    plt.show()
