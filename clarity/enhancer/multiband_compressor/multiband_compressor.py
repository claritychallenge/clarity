"""An implementation for Multiband Dynamic Range Compressor."""

from __future__ import annotations

import numpy as np

from clarity.enhancer.multiband_compressor.crossover import Crossover
from clarity.enhancer.multiband_compressor.compressor_qmul import Compressor


class MultibandCompressor:
    def __init__(
        self,
        crossover_frequencies: int | float | list | np.ndarray,
        order: int = 4,
        sample_rate: float = 44100,
        compressors_params: dict | None = None,
    ) -> None:
        """
        Initialize the multiband compressor.

        Args:
            crossover_frequencies (list | np.ndarray): The crossover frequencies for the
                different compressors.
            order (int): The order of the crossover filters. Expected to be 4
            sample_rate (float): The sample rate of the signal.
            compressors_params (dict): A dictionary with the compressors parameters.
                By default, all compressors are initialized with the same parameters.
                if a dictionary is provided, it must have the following keys:
                - attack (float | list): The attack time in milliseconds.
                - release (float | list): The release time in milliseconds.
                - threshold (float | list): The threshold in dB.
                - ratio (float | list): The ratio.
                - gain (float | list): The make-up gain in dB.
                If the values are floats, all compressors will have the same value. If
                the values are lists, the length must be the same as the number of
                compressors (len(crossover_frequencies) + 1).
        """

        if isinstance(crossover_frequencies, (int, float)):
            crossover_frequencies = [crossover_frequencies]

        self.xover_freqs = np.array(crossover_frequencies)
        self.num_compressors = len(self.xover_freqs) + 1

        # Compute the crossover filters
        self.crossover = Crossover(self.xover_freqs, order, sample_rate)

        self.sample_rate = sample_rate

        # Set the params for the compressors
        if compressors_params is None:
            self.attack: float = 15.0
            self.release: float = 100.0
            self.threshold: float = 0.0
            self.ratio: float = 1.0
            self.gain: float = 0.0
        else:
            self.attack = compressors_params.get("attack", 15.0)
            self.release = compressors_params.get("release", 100.0)
            self.threshold = compressors_params.get("threshold", 0.0)
            self.ratio = compressors_params.get("ratio", 1.0)
            self.gain = compressors_params.get("gain", 0.0)

        # Initialize the compressors
        self.compressor: list = list()
        self.set_compressors(
            attack=self.attack,
            release=self.release,
            threshold=self.threshold,
            ratio=self.ratio,
            gain=self.gain,
        )

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

        Returns:
            list[Compressor]: The compressors.

        Raises:
            ValueError: If the parameters are not the same length as
            crossover frequencies + 1.
        """

        if isinstance(attack, (int, float)):
            attack = [float(attack)] * self.num_compressors
        if isinstance(release, (int, float)):
            release = [float(release)] * self.num_compressors
        if isinstance(threshold, (int, float)):
            threshold = [float(threshold)] * self.num_compressors
        if isinstance(ratio, (int, float)):
            ratio = [float(ratio)] * self.num_compressors
        if isinstance(gain, (int, float)):
            gain = [float(gain)] * self.num_compressors

        if len(attack) != self.num_compressors:
            raise ValueError(
                "Attack must be a float or have the same "
                "length as crossover frequencies + 1. "
                f"{len(attack)} was provided, {self.num_compressors} expected."
            )
        if len(release) != self.num_compressors:
            raise ValueError(
                "Release must be a float or have the same "
                "length as crossover frequencies + 1. "
                f"{len(release)} was provided, {self.num_compressors} expected."
            )
        if len(threshold) != self.num_compressors:
            raise ValueError(
                "Threshold must be a float or have the same "
                "length as crossover frequencies + 1. "
                f"{len(threshold)} was provided, {self.num_compressors} expected."
            )
        if len(ratio) != self.num_compressors:
            raise ValueError(
                "Ratio must be a float or have the same "
                "length as crossover frequencies + 1. "
                f"{len(ratio)} was provided, {self.num_compressors} expected."
            )
        if len(gain) != self.num_compressors:
            raise ValueError(
                "Gain must be a float or have the same length as "
                "crossover frequencies + 1. "
                f"{len(gain)} was provided, {self.num_compressors} expected."
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
            for i in range(self.num_compressors)
        ]

    def __call__(
        self, signal: np.ndarray, return_bands: bool = False
    ) -> np.ndarray | tuple:
        """Compress the signal.

        Args:
            signal (np.ndarray): The input signal.
                (channel, samples)
            return_bands (bool): If True, return the compressed bands.

        Returns:
            np.ndarray: The compressed signal.
                (channel, samples)
            np.ndarray: The compressed bands if return_bands is True.
                (bands, channel, samples)
        """

        bands_signal = self.crossover(signal[0])[:, np.newaxis, :]
        if signal.shape[0] > 1:
            for sig in signal[1:]:
                bands_signal = np.hstack(
                    (bands_signal, self.crossover(sig)[:, np.newaxis, :])
                )

        compressed_bands = np.zeros_like(bands_signal)
        for idx, sig in enumerate(bands_signal):
            compressed_bands[idx, :, :] = self.compressor[idx](sig)

        compressed_signal = np.sum(compressed_bands, axis=0)
        if return_bands:
            return compressed_signal, compressed_bands
        return compressed_signal

    def __str__(self):
        """Return the string representation of the object."""
        out_text = "Multiband Compressor Summary:\n"
        out_text += f"-Crossover Filter: Frequencies {str(self.crossover)}\n"
        out_text += f"-Compressors: {self.num_compressors}\n"
        for idx, comp in enumerate(self.compressor):
            out_text += f"  -Compressor {idx + 1}: "
            out_text += f"{comp}\n"

        return out_text


if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt

    signal, sr = librosa.load(librosa.ex("brahms"), sr=None, duration=10, mono=False)
    # if signal.ndim == 1:
    #     signal = signal[np.newaxis, :]
    signal = np.vstack((signal, signal * 0.8))

    HL = np.array([20, 20, 30, 40, 50, 60])
    mbc = MultibandCompressor(
        crossover_frequencies=np.array([250, 500, 1000, 2000, 4000]) * np.sqrt(2),
        order=4,
        sample_rate=sr,
        compressors_params={
            "attack": [11, 11, 14, 13, 11, 11],
            "release": [80, 80, 80, 80, 100, 100],
            "threshold": [-40, -40, -40, -40, -40, -40],
            "ratio": [4, 4, 4, 4, 4, 4],
            "gain": np.maximum((HL - 20) / 3, 0),
        },
    )

    compressed_signal, compressed_bands = mbc(signal, return_bands=True)

    for chan in range(signal.shape[0]):
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        axes[0].specgram(signal[chan], Fs=sr, NFFT=512, noverlap=256)
        axes[0].set_title("Original Signal")

        axes[1].specgram(compressed_signal[chan], Fs=sr, NFFT=512, noverlap=256)
        axes[1].set_title("Compressed Signal")

        axes[2].plot(signal[chan])
        axes[2].plot(compressed_signal[chan])
        axes[2].set_title("Time Domain")

        fig.suptitle(f"Channel {chan}", fontsize=20)

        plt.tight_layout()
        plt.show()
        plt.close()
