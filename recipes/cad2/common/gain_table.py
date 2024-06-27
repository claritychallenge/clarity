import numpy as np
from numpy import ndarray

from clarity.enhancer.gha.gha_utils import (
    FittingParams,
    gainrule_camfit_compr,
    multifit_apply_noisegate,
)

# from clarity.enhancer.gha.gainrule_camfit import multifit_apply_noisegate
from clarity.utils.audiogram import Audiogram


class CamfitGainTable:
    def __init__(
        self,
        noisegate_levels: float | ndarray = 45,
        noisegate_slope: float | ndarray = 1.0,
        cr_level: float = 0.0,
        max_output_level: float = 100.0,
        interpolation_freqs: ndarray = np.array(
            [250, 500, 1000, 2000, 4000, 6000, 8000],
        ),
    ):
        self.noisegate_levels = noisegate_levels
        self.noisegate_slope = noisegate_slope
        self.cr_level = cr_level
        self.max_output_level = max_output_level
        self.interpolation_freqs = interpolation_freqs

        self.sFitmodel: FittingParams = {
            "frequencies": np.array(
                [177, 297, 500, 841, 1414, 2378, 4000, 6727, 11314]
            ),
            "edge_frequencies": np.array(
                [
                    1.0000e-08,
                    229.2793,
                    385.3570,
                    648.4597,
                    1.0905e03,
                    1.8337e03,
                    3.0842e03,
                    5.1873e03,
                    8.7241e03,
                    10000001,
                ]
            ),
            "levels": np.arange(
                -10, 110 + 1
            ),  # Levels SPL at which to compute the gains
            "channels": 2,
            "side": "lr",
        }

    def process(
        self,
        audiogram_left: Audiogram,
        audiogram_right: Audiogram,
        interpolate: bool = True,
    ) -> tuple[ndarray, ndarray]:
        """Method to process the CAMFIT gain table.

        Args:
            audiogram_left (Audiogram): Audiogram for the left ear.
            audiogram_right (Audiogram): Audiogram for the right ear.
            interpolate (bool, optional): Whether to interpolate the gain table to the desired frequencies. Defaults to True.
                If False, will return the gain table for CAMFIT frequencies.

        Returns:
            tuple[ndarray, ndarray]: Processed gain tables for the left and right ears.
        """

        gain_table, noisegate_levels, noisegate_slope = gainrule_camfit_compr(
            audiogram_left,
            audiogram_right,
            self.sFitmodel,
            self.noisegate_levels,
            self.noisegate_slope,
            self.cr_level,
            self.max_output_level,
        )

        gain_table_corrected = multifit_apply_noisegate(
            gain_table,
            self.sFitmodel["frequencies"],
            self.sFitmodel["levels"],
            noisegate_levels,
            noisegate_slope,
        )

        if not interpolate:
            return gain_table_corrected[:, :, 0], gain_table_corrected[:, :, 1]

        return self.interpolate(gain_table_corrected[:, :, 0]), self.interpolate(
            gain_table_corrected[:, :, 1]
        )

    def interpolate(self, gain_table: ndarray) -> ndarray:
        """Method to interpolate the CAMFIT gain table to the desired frequencies."""
        return np.array(
            [
                np.interp(
                    x=self.interpolation_freqs,
                    xp=[177, 297, 500, 841, 1414, 2378, 4000, 6727, 11314],
                    fp=x,
                )
                for x in gain_table
            ]
        )
