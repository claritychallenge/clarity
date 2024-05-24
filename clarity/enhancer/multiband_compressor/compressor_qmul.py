""""Module for the Compressor class."""

from __future__ import annotations

import warnings

import numpy as np


class Compressor:
    """Compressor
    Based in the compressor from [1].
    Code adapted from JUCE C++ source code.
    Optimization using IIR filters.

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
        gain: float = 0.0,
        sample_rate: float = 44100.0,
    ) -> None:
        if threshold > 0 or threshold < -60:
            warnings.warn(
                "Threshold outside the recommended range [0.0, -60.0] dB."
                f" {threshold} dB was provided."
            )
        if ratio < 1 or ratio > 20:
            warnings.warn(
                "Ratio outside the recommended range [1.0, 20.0]."
                f" {ratio} was provided."
            )

        if attack < 0.1 or attack > 80.0:
            warnings.warn(
                "Attack outside the recommended range [0.1, 80.0] ms."
                f" {attack} ms was provided."
            )

        if release < 0.1 or release > 1000.0:
            warnings.warn(
                "Release outside the recommended range [0.1, 1000.0] ms."
                f" {release} ms was provided."
            )

        if gain < 0 or gain > 24:
            warnings.warn(
                "Make-up gain outside the recommended range [0.0, 24.0] dB."
                f" {gain} dB was provided."
            )

        self.threshold = float(threshold)
        self.ratio = float(ratio)
        self.attack = float(attack)
        self.release = float(release)
        self.gain = float(gain)
        self.sample_rate = float(sample_rate)
        self.alpha_attack = np.exp(-1.0 / (0.001 * self.sample_rate * self.attack))
        self.alpha_release = np.exp(-1.0 / (0.001 * self.sample_rate * self.release))

    def __call__(self, input_signal: np.ndarray) -> np.ndarray:
        """Process the signal.
        The method processes the input signal and returns the output signal.

        Args:
            input_signal (np.ndarray): The input signal. (channels, samples)

        Returns:
            np.ndarray: The output signal.(channels, samples)
        """

        # The lines below are constant and so could move to the constructor
        alpha_attack = np.exp(-1.0 / (0.001 * self.sample_rate * self.attack))
        alpha_release = np.exp(-1.0 / (0.001 * self.sample_rate * self.release))

        # Compute the instantaneous desired levels
        x_g = 20 * np.log10(np.abs(input_signal))
        x_g[x_g < -120] = -120
        y_g = self.threshold + (x_g - self.threshold) / self.ratio
        y_g[x_g < self.threshold] = x_g[x_g < self.threshold]
        y_l = x_g - y_g

        # Do the filtering - cannot easily vectorise this part
        filtered = np.zeros(input_signal.shape)
        for channel, filtered_signal in enumerate(filtered):
            out = 0
            for i, sample in enumerate(y_l[channel]):
                alpha = alpha_attack if sample > out else alpha_release
                out = alpha * out + (1 - alpha) * sample
                filtered_signal[i] = out

        # Compute the gains and apply to the input signal
        c = 10 ** ((self.gain - filtered) / 20.0)
        output_signal = input_signal * c

        return output_signal

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return (
            f"Compressor: threshold={self.threshold}, ratio={self.ratio}, "
            f"attack={self.attack}, release={self.release}, gain={self.gain}, "
            f"sample_rate={self.sample_rate}"
        )


if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt

    signal, sr = librosa.load(librosa.ex("brahms"), sr=None, duration=10, mono=False)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    compressor = Compressor(
        threshold=-30.0,
        ratio=4.0,
        attack=10.0,
        release=100.0,
        gain=1.25,
        sample_rate=sr,
    )

    compressed_signal = compressor(signal)
    fig, axes = plt.subplots(2, 1)
    axes[0].specgram(signal[0], Fs=sr, NFFT=512, noverlap=256)
    axes[0].set_title("original signal")

    axes[1].specgram(compressed_signal[0], Fs=sr, NFFT=512, noverlap=256)
    axes[1].set_title("Compressed signal")
    plt.yticks([x for x in range(0, int(sr / 2), 1000)])

    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(signal[0])
    plt.plot(compressed_signal[0])
    plt.show()
