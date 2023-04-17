from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from scipy.interpolate import interp1d

from clarity.enhancer.gha.gainrule_camfit import gainrule_camfit_compr

if TYPE_CHECKING:
    from numpy import ndarray

    from clarity.utils.audiogram import Audiogram


class Gaintable(TypedDict):
    """Gaintable for a given audiogram."""

    sGt_uncorr: ndarray
    sGt: ndarray
    noisegatelevel: ndarray
    noisegateslope: ndarray
    frequencies: ndarray
    levels: ndarray
    channels: int


class FittingParams(TypedDict):
    """Fitting parameters for gaintable calculation."""

    frequencies: ndarray
    edge_frequencies: ndarray
    levels: ndarray
    channels: int
    side: str


def get_gaintable(
    audiogram_left: Audiogram,
    audiogram_right: Audiogram,
    noisegate_levels: float | ndarray,
    noisegate_slope: float | ndarray,
    cr_level: float,
    max_output_level: float,
) -> Gaintable:
    """Compute a gaintable for a given audiogram.

    Replaces MATLAB GUI interface of original OpenMHA software for
    gaintable_camfit_compr table calculation. Assumes two channels and
    that audiogram frequencies are identical at two ears.

    Args:
        audiogram_left (Audiogram): the audiogram for the left ear
        audiogram_right (Audiogram): the audiogram for the right ear
        audf (list): audiogram frequencies for fitting
        noisegatelevels (ndarray): compression threshold levels for each frequency band
        noisegateslope (float): determines slope of gains below compression threshold
        cr_level (int): overall input level in dB for calculation of compression ratios
        max_output_level (int): maximum output level in dB

    Returns:
        dict: dim ndarray of gains

    """

    # Fixed centre frequencies for amplification bands
    sFitmodel: FittingParams = {
        "frequencies": np.array(
            [177, 297, 500, 841, 1414, 2378, 4.0000e03, 6.7270e03, 11314]
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
        "levels": np.arange(-10, 110 + 1),  # Levels SPL at which to compute the gains
        "channels": 2,
        "side": "lr",
    }

    # Calculate gains and compression ratios
    gain_table, noisegate_levels, noisegate_slope = gainrule_camfit_compr(
        audiogram_left,
        audiogram_right,
        sFitmodel,
        noisegate_levels,
        noisegate_slope,
        cr_level,
        max_output_level,
    )

    gain_table_corrected = multifit_apply_noisegate(
        gain_table,
        sFitmodel["frequencies"],
        sFitmodel["levels"],
        noisegate_levels,
        noisegate_slope,
    )

    # Reshape sGt here to suit cfg file input
    sGt_data: Gaintable = {
        "sGt_uncorr": gain_table,  # sGt without noisegate correction
        "sGt": np.transpose(np.reshape(gain_table_corrected, (121, 18), order="F")),
        "noisegatelevel": noisegate_levels,
        "noisegateslope": noisegate_slope,
        "frequencies": sFitmodel["frequencies"],
        "levels": sFitmodel["levels"],
        "channels": sFitmodel["channels"],
    }
    return sGt_data


def format_gaintable(
    gaintable: Gaintable,
    noisegate_corr: bool = True,
) -> str:
    """
    Format gaintable for insertion into cfg file as long string.

    Args:
        gaintable (ndarray): The gaintable to format
        noisegate_corr (boolean, optional): apply noisegate correction or do not
            (default: True)

    Returns:
        str: gaintable formatted for insertion into OpenMHA
            cfg file

    """
    if noisegate_corr:
        sGt = gaintable["sGt"]
    else:
        logging.warning("Noise gate correction not being applied to gain table.")
        sGt = gaintable["sGt_uncorr"]

    # Do not apply gains to 9th and 18th row
    logging.info("No application of gains to 9th and 18th row of table.")
    sGt[[8, -1], :] = 0

    # Format for inclusion in cfg file
    v = "["
    for k in range(0, np.shape(sGt)[0]):
        v += f"{sGt[k, :]};"
    v += "]"
    v = v.replace("\n", "")
    v = re.sub(" +", " ", v)  # remove extra white spaces
    # v = re.sub(". ", " ", v)  # remove point after integers

    formatted_sGt = v

    return formatted_sGt


def multifit_apply_noisegate(
    gain_table: ndarray,
    sFit_model_frequencies: ndarray,
    sFit_model_levels: ndarray,
    noisegate_level: ndarray,
    noisegate_slope: ndarray,
) -> ndarray:
    """Apply noisegate to the gain table

    Based on OpenMHA subfunction of libmultifit.m.

    Args:
        gain_table (ndarray): gain array
        sFit_model_frequencies (list): FFT filterbank frequencies
        sFit_model_levels (ndarray): levels at which to calculate gains
        noisegate_level (ndarray): chosen compression threshold
        noisegate_slope (ndarray): determines slope below compression threshold

    Returns:
        ndarray: gain table with noisegate applied

    """
    gain_table_corrected = gain_table.copy()
    for i in [0, 1]:
        for kf in range(len(sFit_model_frequencies)):
            gain_noisegate = interp1d(
                sFit_model_levels,
                gain_table_corrected[:, kf, i],
                fill_value="extrapolate",
            )(noisegate_level[kf, i])
            idx = [
                i for i, x in enumerate(sFit_model_levels < noisegate_level[kf, i]) if x
            ]
            gain_table_corrected[idx, kf, i] = (
                sFit_model_levels[idx] - noisegate_level[kf, i]
            ) * noisegate_slope[kf, i] + gain_noisegate

    return gain_table_corrected
