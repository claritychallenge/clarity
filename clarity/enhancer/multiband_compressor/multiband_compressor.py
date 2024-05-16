"""An implementation for Multiband Dynamic Range Compressor."""
from __future__ import annotations

from clarity.enhancer.multiband_compressor.crossover import Crossover

import numpy as np


class Compressor:
    """Compressor
    Based in the compressor from [1].
    Code adapted from JUCE C++ source code.

    References:

    [1] Giannoulis, D., Massberg, M., & Reiss, J. D. (2012).
    Digital dynamic range compressor design - A tutorial and analysis.
    Journal of the Audio Engineering Society, 60(6), 399-408.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        ratio: float = 1.0,
        attack: float = 15.0,
        release: float = 100.0,
        makeUpGain: float = 0.0,
        sample_rate: float = 44100.0,
    ) -> None:
        if threshold > 0 or threshold < -60:
            raise ValueError("Threshold must be between 0 and -60 dB.")
        if ratio < 1 or ratio > 20:
            raise ValueError("Ratio must be between 1 and 20.")
        if attack < 0.1 or attack > 80.0:
            raise ValueError("Attack must be between 0.1 and 80. ms.")
        if release < 0.1 or release > 1000.0:
            raise ValueError("Release must be between 0.1 and 1000. ms.")
        if makeUpGain < 0 or makeUpGain > 24:
            raise ValueError("Make-up gain must be between 0 and 24 dB.")

        self.threshold = threshold
        self.ratio = ratio
        self.attackTime = attack
        self.release = release
        self.makeUpGain = makeUpGain
        self.sample_rate = sample_rate

    def processBlock(self, input_signal: np.ndarray) -> np.ndarray:
        """Process the signal.
        The method processes the input signal and returns the output signal.

        Args:
            input_signal (np.ndarray): The input signal. (channels, frames)

        Returns:
            np.ndarray: The output signal.
        """
        alphaAttack = np.exp(-1.0 / (0.001 * self.sample_rate * self.attackTime))
        alphaRelease = np.exp(-1.0 / (0.001 * self.sample_rate * self.release))

        channels = len(input_signal.shape)
        if channels not in [1, 2]:
            raise ValueError("Only mono and stereo signals are supported.")
        if input_signal.shape[0] < input_signal.shape[-1]:
            # Channel First
            input_signal = input_signal.T

        output_signal = np.zeros_like(input_signal)
        yL_prev = np.zeros(input_signal.shape[-1])

        for i in range(input_signal.shape[-1]):
            for j in range(channels):
                inputSignal = input_signal[j, i]
                if abs(inputSignal) < 0.000001:
                    x_g = -120
                else:
                    x_g = 20 * np.log10(abs(inputSignal))

                if x_g >= self.threshold:
                    y_g = self.threshold + (x_g - self.threshold) / self.ratio
                else:
                    y_g = x_g
                x_l = x_g - y_g

                if x_l > yL_prev[j]:
                    y_l = alphaAttack * yL_prev[j] + (1 - alphaAttack) * x_l
                else:
                    y_l = alphaRelease * yL_prev[j] + (1 - alphaRelease) * x_l

                c = 10 ** ((self.makeUpGain - y_l) / 20.0)
                yL_prev[j] = y_l

                output_signal[j, i] = inputSignal * c

        return output_signal


class MultibandCompressor:
    def __init__(
        self,
        center_frequencies: list | np.ndarray = None,
        crossover_frequencies: list | np.ndarray = None,
        order: int = 4,
        sample_rate: float = 44100,
    ) -> None:
        """
        Initialize the multiband compressor.

        Args:
            center_frequencies (list | np.ndarray): The center frequencies of the bands.
              If center_frequencies are not None, the crossover_frequencies are ignored
              and computed as center_frequencies[:-2] / sqrt(2).
            crossover_frequencies (list | np.ndarray): The crossover frequencies.
              Ignored if center_frequencies are provided.
            order (int): The order of the crossover filters.
            sample_rate (float): The sample rate of the signal.
        """
        if center_frequencies is None and crossover_frequencies is None:
            raise ValueError(
                "Either center_frequencies or crossover_frequencies must be provided."
            )

        self.center_frequencies = center_frequencies
        if center_frequencies is not None:
            self.xover_freqs = self.compute_crossover_freqs()
        else:
            self.xover_freqs = np.array(crossover_frequencies)

        # Compute the crossover filters
        self.crossover = Crossover(self.xover_freqs, order, sample_rate)
        self.sample_rate = sample_rate

        self.attack = list()
        self.release = list()
        self.kneewidth = list()
        self.threshold = list()
        self.compressor = list()

    def compute_crossover_freqs(self) -> list:
        """Compute the crossover frequencies.

        Args:

        Returns:
            np.ndarray: The crossover frequencies.
        """
        if len(self.center_frequencies) == 1:
            return [self.center_frequencies[0] * np.sqrt(2)]
        return [float(x) * np.sqrt(2) for x in self.center_frequencies[:-1]]

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Compress the signal."""
        split_signal = self.crossover(signal)
        compressed_signal = np.zeros_like(signal)
        for idx, sig in enumerate(split_signal):
            compressed_band, _, _ = self.compressor[idx].process(sig)
            compressed_signal += compressed_band

        return compressed_signal

    def __str__(self):
        out_text = "Multiband Compressor Summary:\n"
        out_text += f"Center Frequencies: {self.center_frequencies}\n"
        out_text += f"Filters: {self.crossover}\n"
        return out_text


if __name__ == "__main__":
    mbc = MultibandCompressor(center_frequencies=[250, 500, 1000, 2000, 4000, 8000])
    print(mbc.xover_freqs)
