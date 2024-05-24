"""An implementation for Multiband Dynamic Range Compressor."""

from __future__ import annotations

import numpy as np

from clarity.enhancer.multiband_compressor.crossover import Crossover
from clarity.enhancer.multiband_compressor.compressor_qmul import Compressor


class MultibandCompressor:
    def __init__(
        self,
        center_frequencies: list | np.ndarray,
        order: int = 4,
        sample_rate: float = 44100,
    ) -> None:
        """
        Initialize the multiband compressor.

        Args:
            center_frequencies (list | np.ndarray): The center frequencies of the bands.
              The crossover_frequencies are computed as
              center_frequencies[:-2] / sqrt(2).
            order (int): The order of the crossover filters.
            sample_rate (float): The sample rate of the signal.
        """

        self.center_frequencies = center_frequencies
        self.xover_freqs = self.compute_crossover_freqs()

        # Compute the crossover filters
        self.crossover = Crossover(self.xover_freqs, order, sample_rate)

        self.sample_rate = sample_rate

        self.attack: list = list()
        self.release: list = list()
        self.threshold: list = list()
        self.ratio: list = list()
        self.gain: list = list()
        self.compressor: list = list()

    def set_compressors(
        self,
        attack: list | float = 15.0,
        release: list | float = 100.0,
        threshold: list | float = 0.0,
        ratio: list | float = 1.0,
        gain: list | float = 0.0,
    ) -> None:
        """Set the compressors parameters.

        Parameters can be a float or a list with the same length as center_frequencies.

        Args:
            attack (list | float): The attack time in milliseconds.
            release (list | float): The release time in milliseconds.
            threshold (list | float): The threshold in dB.
            ratio (list | float): The ratio.
            gain (list | float): The make-up gain in dB.

        Raises:
            ValueError: If either of the parameters are a list with a different length
                as center_frequencies.
        """

        num_compressors = len(self.center_frequencies)
        if isinstance(attack, (int, float)):
            attack = [float(attack)] * num_compressors
        if isinstance(release, (int, float)):
            release = [float(release)] * num_compressors
        if isinstance(threshold, (int, float)):
            threshold = [float(threshold)] * num_compressors
        if isinstance(ratio, (int, float)):
            ratio = [float(ratio)] * num_compressors
        if isinstance(gain, (int, float)):
            gain = [float(gain)] * num_compressors

        if len(attack) != num_compressors:
            raise ValueError(
                "Attack must be a float or have the same length as center_frequencies."
            )
        if len(release) != num_compressors:
            raise ValueError(
                "Release must be a float or have the same length as center_frequencies."
            )
        if len(threshold) != num_compressors:
            raise ValueError(
                "Threshold must be a float or have the same "
                "length as center_frequencies."
            )
        if len(ratio) != num_compressors:
            raise ValueError(
                "Ratio must be a float or have the same length as center_frequencies."
            )
        if len(gain) != num_compressors:
            raise ValueError(
                "Gain must be a float or have the same length as center_frequencies."
            )

        self.compressor = [
            Compressor(
                attack=attack[i],
                release=release[i],
                threshold=threshold[i],
                ratio=ratio[i],
                gain=gain[i],
                sample_rate=self.sample_rate,
            )
            for i in range(num_compressors)
        ]

    def compute_crossover_freqs(self) -> list:
        """Compute the crossover frequencies.

        Args:

        Returns:
            np.ndarray: The crossover frequencies.
        """
        if len(self.center_frequencies) == 1:
            return [self.center_frequencies[0] * np.sqrt(2)]
        return [float(x) * np.sqrt(2) for x in self.center_frequencies[:-1]]

    def __call__(
        self, signal: np.ndarray, return_bands: bool = False
    ) -> np.ndarray | tuple:
        """Compress the signal.

        Args:
            signal (np.ndarray): The input signal.
            return_bands (bool): If True, return the compressed bands.

        Returns:
            np.ndarray: The compressed signal.
            np.ndarray: The compressed bands if return_bands is True.

        Raises:
            ValueError: If the compressors are not set.
        """

        if len(self.compressor) == 0:
            raise ValueError("Compressors not set. Use set_compressors method.")

        split_signal = self.crossover(signal)
        compressed_signal = np.zeros_like(split_signal)

        for idx, sig in enumerate(split_signal):
            compressed_signal[idx] = self.compressor[idx](sig[np.newaxis, :]).squeeze(0)

        if return_bands:
            return np.sum(compressed_signal, axis=0), compressed_signal
        return np.sum(compressed_signal, axis=0)

    def __str__(self):
        out_text = "Multiband Compressor Summary:\n"
        out_text += f"Center Frequencies: {self.center_frequencies}\n"
        out_text += f"Filters: {self.crossover}\n"
        return out_text


if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt
    from scipy.io import wavfile

    signal, sr = librosa.load(librosa.ex("brahms"), sr=None, duration=10, mono=False)

    mbc = MultibandCompressor(center_frequencies=[250, 500, 1000, 2000, 4000, 8000])
    print(mbc.xover_freqs)
    print(mbc)

    HL = np.array([20, 20, 30, 40, 50, 60])
    gains = np.maximum((HL - 20) / 3, 0)
    cratio = [1.32, 1.32, 1.67, 1.74, 1.45, 1.45]
    cattack_ms = [11, 11, 14, 13, 11, 11]
    crelease_ms = [80, 80, 80, 80, 100, 100]
    cthreshold = [-30, -30, -30, -30, -30, -30]

    mbc.set_compressors(
        attack=cattack_ms,
        release=crelease_ms,
        threshold=cthreshold,
        ratio=cratio,
        gain=gains,
    )
    compressed_signal, compressed_bands = mbc(signal, return_bands=True)
    fig, axes = plt.subplots(8, 1, figsize=(10, 10))
    axes[0].specgram(signal, Fs=sr, NFFT=512, noverlap=256)
    axes[0].set_title("Original Signal")
    wavfile.write("original_signal.wav", sr, signal)
    for idx, band in enumerate(compressed_bands):
        axes[idx + 1].specgram(band, Fs=sr, NFFT=512, noverlap=256)
        axes[idx + 1].set_title(f"Band {idx + 1}")
        wavfile.write(f"band_{idx}.wav", sr, band)

    axes[-1].specgram(compressed_signal, Fs=sr, NFFT=512, noverlap=256)
    axes[-1].set_title("Compressed Signal")
    wavfile.write("compressed_signal.wav", sr, compressed_signal)
    plt.tight_layout()
    plt.show()
