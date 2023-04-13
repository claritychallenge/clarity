"""Compressor Class"""
from __future__ import annotations

from typing import Any, Final

import numpy as np

EPS: Final = 1e-8


class Compressor:
    def __init__(
        self,
        fs: float = 44100.0,
        attack: float = 5.0,
        release: float = 20.0,
        threshold: float = 1.0,
        attenuation: float = 0.0001,
        rms_buffer_size: float = 0.2,
        makeup_gain: float = 1.0,
        **_kwargs,
    ) -> None:
        """Instantiate the Compressor Class.

        Args:
            fs (float): (default = 44100.0)
            attack (float): (default = 5.0)
            release float: (default = 20.0)
            threshold (float): (default = 1.0)
            attenuation (float): (default = 0.0001)
            rms_buffer_size (float): (default = 0.2)
            makeup_gain (float): (default = 1.0)
        """
        self.fs = fs
        self.rms_buffer_size = rms_buffer_size
        self.set_attack(attack)
        self.set_release(release)
        self.threshold = threshold
        self.attenuation = attenuation
        self.makeup_gain = makeup_gain

        # window for computing rms
        self.win_len = int(self.rms_buffer_size * self.fs)
        self.window = np.ones(self.win_len)

    def set_attack(self, t_msec: float) -> None:
        """DESCRIPTION

        Args:
            t_msec (float): DESCRIPTION

        Returns:
            float: DESCRIPTION
        """
        t_sec = t_msec / 1000.0
        reciprocal_time = 1.0 / t_sec
        self.attack = reciprocal_time / self.fs

    def set_release(self, t_msec: float) -> None:
        """DESCRIPTION

        Args:
            t_msec (float): DESCRIPTION

        Returns:
            float: DESCRIPTION
        """
        t_sec = t_msec / 1000.0
        reciprocal_time = 1.0 / t_sec
        self.release = reciprocal_time / self.fs

    def process(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Any]]:
        """DESCRIPTION

        Args:
            signal (np.array): DESCRIPTION

        Returns:
            np.array: DESCRIPTION
        """
        padded_signal = np.concatenate((np.zeros(self.win_len - 1), signal))
        rms = np.sqrt(
            np.convolve(padded_signal**2, self.window, mode="valid") / self.win_len
            + EPS
        )
        comp_ratios: list[float] = []
        curr_comp: float = 1.0
        for rms_i in rms:
            if rms_i > self.threshold:
                temp_comp = (rms_i * self.attenuation) + (
                    (1.0 - self.attenuation) * self.threshold
                )
                curr_comp = curr_comp * (1.0 - self.attack) + (temp_comp * self.attack)
            else:
                curr_comp = self.release + curr_comp * (1 - self.release)
            comp_ratios.append(curr_comp)
        return (signal * np.array(comp_ratios) * self.makeup_gain), rms, comp_ratios
