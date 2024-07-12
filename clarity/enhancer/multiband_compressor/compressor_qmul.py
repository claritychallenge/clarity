""""Module for the Compressor class."""

from __future__ import annotations

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

    Example:
    >>> import librosa
    >>> import matplotlib.pyplot as plt

    >>> signal, sr = librosa.load(
    ...     librosa.ex("brahms"),
    ...     sr=None,
    ...     duration=10,
    ...     mono=False
    ... )
    >>> if signal.ndim == 1:
    >>>      signal = signal[np.newaxis, :]

    >>> compressor = Compressor(
    ...    threshold=-30.0,
    ...    ratio=4.0,
    ...    attack=10.0,
    ...    release=100.0,
    ...    makeup_gain=1.25,
    ...    sample_rate=sr,
    ...    knee_width=10.0,
    ...)

    >>> compressed_signal = compressor(signal)
    >>> fig, axes = plt.subplots(2, 1)
    >>> axes[0].specgram(signal[0], Fs=sr, NFFT=512, noverlap=256)
    >>> axes[0].set_title("original signal")
    >>> axes[1].specgram(compressed_signal[0], Fs=sr, NFFT=512, noverlap=256)
    >>> axes[1].set_title("Compressed signal")
    >>> plt.yticks([x for x in range(0, int(sr / 2), 1000)])
    >>> plt.tight_layout()
    >>> plt.show()
    >>> plt.close()

    >>> plt.figure()
    >>> plt.plot(signal[0])
    >>> plt.plot(compressed_signal[0])
    >>> plt.show()
    """

    def __init__(
        self,
        threshold: float = 0.0,
        ratio: float = 1.0,
        attack: float = 15.0,
        release: float = 100.0,
        makeup_gain: float = 0.0,
        knee_width: float = 0.0,
        sample_rate: float = 44100.0,
    ) -> None:
        """Constructor for the Compressor class.

        Args:
            threshold (float): The threshold level in dB.
            ratio (float): The compression ratio.
            attack (float): The attack time in ms.
            release (float): The release time in ms.
            makeup_gain (float): The make-up gain in dB.
            knee_width (float): The knee width in dB.
            sample_rate (float): The sample rate in Hz.

        Notes:
            Original implementation recommends ranges for each parameter.
            We are not enforcing these ranges in this implementation.
            The ranges are:
            - threshold in the range [0.0, -60.0] dB,
            - ratio in the range [1.0, 20.0],
            - attack in the range [0.1, 80.0] ms,
            - release in the range [0.1, 1000.0] ms,
            - makeup_gain in the range [0.0, 24.0] dB.
            - knee_width in the range [0.0, 10.0] dB.
        """

        self.threshold = float(threshold)
        self.ratio = float(ratio)
        self.attack = float(attack)
        self.release = float(release)
        self.makeup_gain = float(makeup_gain)
        self.sample_rate = float(sample_rate)
        self.knee_width = float(knee_width)

        self.eps = 1e-12

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

        # Compute the instantaneous desired levels
        input_signal[input_signal == 0] = self.eps
        x_g = 20 * np.log10(np.abs(input_signal))
        x_g[x_g < -120] = -120
        y_g = self.threshold + (x_g - self.threshold) / self.ratio

        if self.knee_width == 0:
            # If knee width is zero, apply hard knee
            y_g[x_g < self.threshold] = x_g[x_g < self.threshold]
        else:
            # Apply soft knee
            index = 2 * np.abs(x_g - self.threshold) <= self.knee_width
            y_g[index] = x_g[index] + (
                (1 / self.ratio - 1)
                * (x_g[index] - self.threshold + self.knee_width / 2) ** 2
            ) / (2 * self.knee_width)

            index = 2 * (x_g - self.threshold) < -self.knee_width
            y_g[index] = x_g[index]

        y_l = x_g - y_g

        # Do the filtering - cannot easily vectorise this part
        filtered = np.zeros(input_signal.shape)
        for channel, filtered_signal in enumerate(filtered):
            out = 0
            for i, sample in enumerate(y_l[channel]):
                alpha = self.alpha_attack if sample > out else self.alpha_release
                out = alpha * out + (1 - alpha) * sample
                filtered_signal[i] = out

        # Compute the gains and apply to the input signal
        c = 10 ** ((self.makeup_gain - filtered) / 20.0)
        output_signal = input_signal * c

        return output_signal

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return (
            f"Compressor: threshold={self.threshold}, ratio={self.ratio},"
            f" attack={self.attack}, release={self.release}, makeup"
            f" gain={self.makeup_gain}, sample_rate={self.sample_rate}"
        )
