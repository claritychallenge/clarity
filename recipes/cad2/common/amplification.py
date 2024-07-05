"""Hearing aid amplification module for CAD2."""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from clarity.enhancer.multiband_compressor import MultibandCompressor
from clarity.utils.audiogram import Listener
from recipes.cad2.common.gain_table import CamfitGainTable


class HearingAid:
    def __init__(self, compressor_params: dict, gain_table_params: dict) -> None:
        """Hearing aid amplification module for CAD2.

        Args:
            compressor_params (dict): Parameters for the multiband compressor.
            gain_table_params (dict): Parameters for the gain table.

        Example:
            compressor_params = {
                "crossover_frequencies": [1000, 2000, 4000],
                "sample_rate": 16000,
            }
            gain_table_params = {
                "noisegate_levels": 40
                "noisegate_slope": 0
                "cr_level": 0
                "max_output_level": 100
            }
            hearing_aid = HearingAid(compressor_params, gain_table_params)
        """

        self.compressor = MultibandCompressor(**compressor_params)
        self.gain_table = CamfitGainTable(**gain_table_params)
        self.mbc = []

    def set_compressors(self, listener: Listener) -> None:
        """Set the compressors for the listener.

        Args:
            listener (Listener): Listener audiogram.
        """

        gaintable_left_ear, gaintable_right_ear = self.gain_table.process(
            listener.audiogram_left,
            listener.audiogram_right,
            interpolate=False,
        )
        gain_left, cr_left = self.compute_55_65_85_params(gaintable_left_ear)
        gain_right, cr_right = self.compute_55_65_85_params(gaintable_right_ear)

        for i in range(2):
            self.mbc.append(
                self.compressor.set_compressors(
                    attack=[11, 11, 14, 13, 11, 11],
                    release=[80, 80, 80, 80, 100, 100],
                    threshold=-40,
                    ratio=(cr_left if i == 0 else cr_right).tolist(),
                    makeup_gain=(gain_left if i == 0 else gain_right).tolist(),
                    knee_width=0,
                )
            )

    @staticmethod
    def compute_55_65_85_params(gain_table: ndarray) -> tuple[ndarray, ndarray]:
        """Compute the gain and compression ratio using
        levels at  55. 65 and 85 dB SPL from CAMFIT gaintable.
        """
        level_55 = gain_table[65]  # 55 dB SPL
        level_85 = gain_table[95]  # 85 dB SPL
        gain = gain_table[75]  # 65 dB SPL
        cr = np.divide(
            (85 - 55),
            (85 + level_85) - (55 + level_55),
            out=np.ones_like(level_55),
            where=(85 + level_85) - (55 + level_55) != 0,
        )

        gain = np.interp(
            [250, 500, 1000, 2000, 4000, 6000],
            [177, 297, 500, 841, 1414, 2378, 4000, 6727, 11314],
            gain,
        )
        cr = np.interp(
            [250, 500, 1000, 2000, 4000, 6000],
            [177, 297, 500, 841, 1414, 2378, 4000, 6727, 11314],
            cr,
        )

        return np.minimum(gain, 24), cr

    def __call__(self, signal: ndarray) -> ndarray:
        """Apply hearing aid to the signal.

        Args:
            signal (ndarray): Signal to enhance.
            listener (Listener): Listener audiogram.

        Returns:
            ndarray: Enhanced signal.
        """
        enhanced_signal = self.compressor(signal)
        return enhanced_signal


if __name__ == "__main__":
    compressor_params = {
        "crossover_frequencies": np.array([250, 500, 1000, 2000, 4000]) * np.sqrt(2),
        "sample_rate": 44100,
    }
    gain_table_params = {
        "noisegate_levels": 45,
        "noisegate_slope": 0,
        "cr_level": 0,
        "max_output_level": 100,
    }
    hearing_aid = HearingAid(compressor_params, gain_table_params)
    listener = Listener.from_dict(
        {
            "name": "listener_14",
            "audiogram_cfs": [250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
            "audiogram_levels_l": [20, 35, 40, 50, 60, 65, 75, 70],
            "audiogram_levels_r": [80, 80, 80, 70, 70, 70, 80, 80],
        }
    )
    signal = np.random.randn(44100)
    hearing_aid.set_compressors(listener)
    enhanced_signal = hearing_aid(signal[np.newaxis, :])
    print(enhanced_signal)
