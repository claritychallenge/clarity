import copy
import logging
import re

import numpy as np
from scipy.interpolate import interp1d

from clarity.enhancer.gha.gainrule_camfit import gainrule_camfit_compr


def get_gaintable(
    audiogram, noisegatelevels, noisegateslope, cr_level, max_output_level
):
    """Compute a gaintable for a given audiogram.

    Replaces MATLAB GUI interface of original OpenMHA software for
    gaintable_camfit_compr table calculation. Assumes two channels and
    that audiogram frequencies are identical at two ears.

    Args:
        audiogram (Audiogram): the audiogram for which to compute the gain table
        audf (list): audiogram frequencies for fitting
        noisegatelevels (ndarray): compression threshold levels for each frequency band
        noisegateslope (int): determines slope of gains below compression threshold
        cr_level (int): overall input level in dB for calculation of compression ratios
        max_output_level (int): maximum output level in dB

    Returns:
        dict: dim ndarray of gains

    """
    # Initialise parameters
    num_channels = 2  # only configured for two channels
    sFitmodel = {}

    # Fixed centre frequencies for amplification bands
    sFitmodel["frequencies"] = [
        177,
        297,
        500,
        841,
        1414,
        2378,
        4.0000e03,
        6.7270e03,
        11314,
    ]
    sFitmodel["edge_frequencies"] = [
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

    # Input levels in SPL for which to compute the gains
    sFitmodel["levels"] = np.arange(-10, 110 + 1)
    sFitmodel["channels"] = num_channels
    sFitmodel["side"] = "lr"

    # Calculate gains and compression ratios
    output = gainrule_camfit_compr(
        audiogram,
        sFitmodel,
        noisegatelevels,
        noisegateslope,
        cr_level,
        max_output_level,
    )

    sGt = {}
    sGt["sGt_uncorr"] = copy.deepcopy(output["sGt"])  # sGt without noisegate correction

    output = multifit_apply_noisegate(
        output["sGt"],
        sFitmodel["frequencies"],
        sFitmodel["levels"],
        output["noisegatelevel"],
        output["noisegateslope"],
    )

    # Reshape sGt here to suit cfg file input
    sGt["sGt"] = np.transpose(np.reshape(output["sGt"], (121, 18), order="F"))

    sGt["noisegatelevel"] = output["noisegatelevel"]
    sGt["noisegateslope"] = output["noisegateslope"]
    sGt["frequencies"] = sFitmodel["frequencies"]
    sGt["levels"] = sFitmodel["levels"]
    sGt["channels"] = num_channels

    return sGt


def format_gaintable(gaintable, noisegate_corr=True):
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
    sGt, sFit_model_frequencies, sFit_model_levels, noisegate_level, noisegate_slope
):
    """Apply noisegate.

    Based on OpenMHA subfunction of libmultifit.m.

    Args:
        sGt (ndarray): gain array
        sFit_model_frequencies (list): FFT filterbank frequencies
        sFit_model_levels (ndarray): levels at which to calculate gains
        noisegate_level (ndarray): chosen compression threshold
        noisegate_slope (ndarray): determines slope below compression threshold

    Returns:
        ndarray: Noise signal

    """

    for i in [0, 1]:
        for kf in range(len(sFit_model_frequencies)):
            gain_noisegate = interp1d(
                sFit_model_levels, sGt[:, kf, i], fill_value="extrapolate"
            )(noisegate_level[kf, i])
            idx = [
                i for i, x in enumerate(sFit_model_levels < noisegate_level[kf, i]) if x
            ]
            sGt[idx, kf, i] = (
                sFit_model_levels[idx] - noisegate_level[kf, i]
            ) * noisegate_slope[kf, i] + gain_noisegate

    output = {}
    output["sGt"] = {}
    output["sGt"] = sGt
    output["noisegatelevel"] = noisegate_level
    output["noisegateslope"] = noisegate_slope

    return output
