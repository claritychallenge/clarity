import copy
import logging

import numpy as np
from scipy.interpolate import interp1d


def compute_proportion_overlap(a1, a2, b1, b2):
    """Compute proportion of overlap of two ranges.

    For ranges (a1, a2) and (b1, b2), express the extent of the overlap
    in the range as a proportion of the extent of range (b1, b2).add()

    e.g (4, 9) and (6, 15) -> overlap (6,9), so proportion is (9-6)/(15-6)

    """
    left = max(a1, b1)
    right = min(a2, b2)
    if left > right:
        overlap = 0.0
    else:
        overlap = float(right - left) / (b2 - b1)
    return overlap


def isothr(vsDesF):
    """Calculate conversion factors of HL thresholds to SPL thresholds.

    Translation of OpenMHA isothr.m. Calculates conversion factors of HL
    thresholds to SPL thresholds. Values from 20 Hz to 12500 Hz are taken
    from ISO 226:2003(E). Values from 14000 Hz to 18000 Hz are taken from ISO
    389-7:2005 (reference thresholds of hearing for free field listening).
    Values at 0 and 20000 Hz are not taken from the ISO Threshold contour.

    Args:
        vsDesF (list): centre frequencies for the amplification bands as 177,
            297, 500, 841,  1414,  2378, 4000, 6727, 11314 Hz

    Returns:
        ndarray: conversion factors

    """
    iso226_389 = np.zeros((34, 2))
    iso226_389[:, 0] = [
        0,
        20,
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        12500,
        14000,
        16000,
        18000,
        20000,
    ]
    iso226_389[:, 1] = [
        80.0,
        78.5,
        68.7,
        59.5,
        51.1,
        44.0,
        37.5,
        31.5,
        26.5,
        22.1,
        17.9,
        14.4,
        11.4,
        8.6,
        6.2,
        4.4,
        3.0,
        2.2,
        2.4,
        3.5,
        1.7,
        -1.3,
        -4.2,
        -6.0,
        -5.4,
        -1.5,
        6.0,
        12.6,
        13.9,
        12.3,
        18.4,
        40.2,
        73.2,
        70.0,
    ]
    vThr = iso226_389[:, 1]
    vsF = iso226_389[:, 0]

    if isinstance(vsDesF, list):
        vsDesF = np.array(vsDesF)

    if np.size(vsDesF[vsDesF < 50]) > 0:
        logging.warning("Frequency values below 50 Hz set to 50 Hz")
        vsDesF[vsDesF < 50] = 50

    vIsoThrDB = interp1d(vsF, vThr, fill_value="extrapolate")(vsDesF)

    return vIsoThrDB


def freq_interp_sh(f_in, y_in, f):
    """Linear interpolation on logarithmic frequency scale.

    Has samples and hold on edges.

    Args:
        f_in (ndarray): audiogram frequencies (Hz)
        y_in (ndarray): audiogram levels
        f (list): FFT filterbank frequencies

    Returns:
        ndarray: interpolated levels corresponding to filterbank frequencies

    """

    # Checks
    if np.size(f[0]) > 1:
        f = f[0]
    if np.size(f_in[0]) > 1:
        f_in = f_in[0]
    if np.size(y_in[0]) > 1:
        y_in = y_in[0]

    vals = np.pad(
        f_in.astype(float), 1, constant_values=((0.5 * f_in[0], 2 * f_in[-1]))
    )
    yvals = np.pad(y_in, 1, constant_values=((y_in[0], y_in[-1])))

    y = interp1d(np.log(vals), yvals, fill_value="extrapolate")(np.log(f))
    y = np.expand_dims(y, 0)

    return y


def gains(compr_thr_inputs, compr_thr_gains, compression_ratios, levels):
    """Based on OpenMHA gains subfunction of gainrule_camfit_compr.

    Args:
        compr_thr_inputs (ndarray): levels for speech in dynamic compression (dc)
            bands minus minima distance (38 dB)
        compr_thr_gains (ndarray): interpolated audiogram levels plus conversion
            factors of HL thresholds to SPL thresholds (output of isothr) minus
            compr_thr_inputs
        compression_ratios (ndarray): compression ratios according to CAMFIT compressive
        levels (ndarray): set of levels over which to calculate gains e.g. -10:110 dB

    Returns:
        ndarray: set of uncorrected gains as 2-d numpy array

    """
    levels = np.transpose(np.tile(levels, (np.size(compr_thr_inputs), 1)))
    compr_thr_inputs = np.tile(compr_thr_inputs, (np.shape(levels)[0], 1))
    compr_thr_gains = np.tile(compr_thr_gains, (np.shape(levels)[0], 1))
    compression_ratios = np.tile(compression_ratios, (np.shape(levels)[0], 1))

    compr_thr_outputs = compr_thr_inputs + compr_thr_gains
    outputs = (levels - compr_thr_inputs) / compression_ratios + compr_thr_outputs
    g = outputs - levels

    return g


def gainrule_camfit_linear(
    audiogram, sFitmodel, noisegatelevels=45, noisegateslope=1, max_output_level=100
):
    """Apply linear Cambridge rule for hearing aid fittings 'CAMFIT'.

    Based on OpenMHA gainrule_camfit_linear.m. Applies linear Cambridge rule for
    hearing aid fittings 'CAMFIT'. Implemented as described in B. Moore and B.
    Glasberg (1998), "Use of a loudness model for hearing-aid fitting. I. Linear
    hearing aids" Brit. J. Audiol. (32) 317-335.

    The gain rule limits the gains so that in each band 100 dB output level
    is not exceeded.
    The Cambridge formula defines intercepts only up to 5 kHz. Because the
    intercepts do not vary much between 1kHz and 5kHz anyway (these are all
    within 0dB +/- 1dB), we extend the last intercept of +1dB at 5kHz to all higher
    frequencies. This function assumes audiogram frequencies are identical for
    two ears.

    The original function is part of the HörTech Open Master Hearing Aid (openMHA)
    Copyright © 2007 2009 2011 2013 2015 2017 2018 HörTech gGmbH
    openMHA is free software: see licencing conditions at http://www.openmha.org/

    Args:
        sAud (dict): contains the subject-specific hearing threshold levels in
            dB HL for the left and right ears, and the audiogram frequencies
        sFitmodel (dict): contains the center frequencies for the amplification
            bands and input levels in SPL for which to compute the gains
        noisegatelevels (ndarray): compression threshold levels for each frequency
            band (default: 45)
        noisegateslope (int): determines slope of gains below compression threshold
            (default: 1)

    Returns:
        dict: dictionary containing gain table, noise gate, and noise
            gate expansion slope fields

    """

    intercept_frequencies = np.array(
        [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 5005]
    )
    intercepts = np.array([-11, -10, -8, -6, 0, -1, 1, -1, 0, 1, 1])
    try:
        if np.size(sFitmodel["frequencies"][0][0]) > 1:
            intercepts = freq_interp_sh(
                intercept_frequencies, intercepts, sFitmodel["frequencies"][0][0]
            )
    except Exception:  # noqa E722
        intercepts = freq_interp_sh(
            intercept_frequencies, intercepts, sFitmodel["frequencies"]
        )

    sFitmodel_frequencies = sFitmodel["frequencies"]
    sFitmodel_levels = sFitmodel["levels"]
    if np.size(sFitmodel_frequencies) == 1:
        sFitmodel_frequencies = sFitmodel_frequencies[0][0]
        sFitmodel_levels = sFitmodel_levels[0][0]

    if np.shape(sFitmodel_levels)[0] != 1:
        sFitmodel_levels = np.transpose(sFitmodel_levels)

    # Interpolate audiogram
    # num levels x num freqs x L, R
    sGt = np.zeros((np.size(sFitmodel_levels), np.size(sFitmodel_frequencies), 2))

    noisegate_level = np.zeros((np.size(sFitmodel_frequencies), 2))
    noisegate_slope = np.zeros((np.size(sFitmodel_frequencies), 2))
    insertion_gains_out = np.zeros((np.size(sFitmodel_frequencies), 2))

    for i, levels in enumerate([audiogram.levels_l, audiogram.levels_r]):
        htlside = freq_interp_sh(audiogram.cfs, levels, sFitmodel_frequencies)

        insertion_gains = htlside * 0.48 + intercepts

        # According to B. Moore (1998), "Use of a loudness model for hearing-aid
        # fitting. II. Hearing aids with multi-channel compression" Brit. J.
        # Audiol. (33) 157-170, p. 159, do not permit negative insertion gains in
        # practice.
        insertion_gains[insertion_gains < 0] = 0

        # Set all gains to 0 for 0dB HL flat audiogram
        insertion_gains = insertion_gains * htlside.any()  # any(htlside)

        sGt[:, :, i] = np.tile(insertion_gains, (len(sFitmodel_levels), 1))

        if np.size(insertion_gains[0]) == np.size(htlside):
            insertion_gains = insertion_gains[0]

        insertion_gains_out[:, i] = insertion_gains
        output_levels = np.tile(sFitmodel_levels, (np.size(insertion_gains), 1))
        output_levels = sGt[:, :, i] + np.transpose(output_levels)

        # Where output level is greater than max_output_level, reduce gain
        safe_output_levels = copy.deepcopy(output_levels)
        safe_output_levels[safe_output_levels > max_output_level] = max_output_level

        sGt[:, :, i] = sGt[:, :, i] - (output_levels - safe_output_levels)

        noisegate_level[:, i] = noisegatelevels * np.ones(
            np.size(sFitmodel_frequencies)
        )
        noisegate_slope[:, i] = noisegateslope * np.ones(np.size(sFitmodel_frequencies))

    # overall_level = 10 * np.log(np.sum(10 ** (insertion_gains_out / 10), axis=0))

    output = {}
    output["sGt"] = sGt
    output["noisegate_level"] = noisegate_level
    output["noisegate_slope"] = noisegate_slope
    output["insertion_gains"] = insertion_gains_out

    return output


def gainrule_camfit_compr(
    audiogram,
    sFitmodel,
    noisegatelevels=45,
    noisegateslope=1,
    level=0,
    max_output_level=100,
):
    """Applies compressive Cambridge rule for hearing aid fittings 'CAMFIT'.

    Translation of OpenMHA gainrule_camfit_compr.m.
    Applies compressive Cambridge rule for
    hearing aid fittings 'CAMFIT'. Computes gains for compression according to Moore
    et al. (1999) "Use of a loudness model for hearing aid fitting: II. Hearing aids
    with multi-channel compression." Brit. J. Audiol. (33) 157-170

    The gain rule limits the gains so that in each band 100 dB output level is
    not exceeded. This function assumes audiogram frequencies are identical for the
    two ears. In this implementation, any negative gains are set to 0 dB.

    The original function is part of the HörTech Open Master Hearing Aid (openMHA)
    Copyright © 2007 2009 2011 2013 2015 2017 2018 HörTech gGmbH
    openMHA is free software: see licencing conditions at http://www.openmha.org/

    Args:
        audiogram (Audiogram): the audiogram for which to make the fit
        sFitmodel (dict): contains the center frequencies for the amplification
            bands and input levels in SPL for which to compute the gains
        noisegatelevels (ndarray): compression threshold levels for each frequency
            band (default: 45)
        noisegateslope (int): determines slope of gains below compression threshold
        level (int): input level in each band for compression rate calculation
            (default: 0 for variable level depending on insertion gains)
        max_output_level (int): maximum output level in dB

    Returns:
        dict: dictionary containing gain table, noise gate, and noise
            gate expansion slope fields

    """

    # International long-term average speech spectrum for speech with an overall
    # level of 70dB in third-octave frequency bands, taken from Byrne et al.
    # (1994) J. Acoust. Soc. Am. 96(4) 2108-2120
    # Average across males and females SG
    LTASS_freq = np.array(
        [
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
            6300,
            8000,
            10000,
            12500,
            16000,
        ]
    )
    LTASS_edge_freq = np.zeros((26))
    LTASS_edge_freq[1:-1] = np.sqrt(LTASS_freq[0:-1] * LTASS_freq[1:])
    LTASS_edge_freq[-1] = 16000 * np.power(2, (1 / 6))
    LTASS_lev = np.array(
        [
            38.6,
            43.5,
            54.4,
            57.7,
            56.8,
            60.2,
            60.3,
            59.0,
            62.1,
            62.1,
            60.5,
            56.8,
            53.7,
            53.0,
            52.0,
            48.7,
            48.1,
            46.8,
            45.6,
            44.5,
            44.3,
            43.7,
            43.4,
            41.3,
            40.7,
        ]
    )
    LTASS_intensity = np.power(10, LTASS_lev / 10)

    frequencies = np.array(sFitmodel["frequencies"])
    edge_freq = np.array(sFitmodel["edge_frequencies"])

    sFitmodel_frequencies = np.array(sFitmodel["frequencies"])
    sFitmodel_levels = np.array(sFitmodel["levels"])

    speech_level_65_in_dc_bands = np.zeros(np.shape(frequencies))

    for band, (f_range_a, f_range_b) in enumerate(zip(edge_freq[:-1], edge_freq[1:])):
        portion = [
            compute_proportion_overlap(f_range_a, f_range_b, ltass_a, ltass_b)
            for (ltass_a, ltass_b) in zip(LTASS_edge_freq[:-1], LTASS_edge_freq[1:])
        ]
        intensity_sum = np.inner(LTASS_intensity, portion)  # weighted sum
        speech_level_70_in_dc_band = 10 * np.log10(intensity_sum)
        speech_level_65_in_dc_bands[band] = speech_level_70_in_dc_band - 5

    # minima in lowest level speech that needs to be understood is 38 dB below
    # speech_level_65_in_dc_bands
    minima_distance = 38

    # Conversion factors of HL thresholds to SPL thresholds
    Conv = isothr(frequencies)

    # Get speech minima
    Lmin = speech_level_65_in_dc_bands - minima_distance

    # Interpolate audiogram and get absolute thresholds in dB HL at centre frequencies
    # and gains required for speech minima
    htl = np.zeros((np.size(frequencies), 2))
    Gmin = np.zeros(np.shape(htl))

    for i, levels in enumerate([audiogram.levels_l, audiogram.levels_r]):
        htl[:, i] = freq_interp_sh(audiogram.cfs, levels, frequencies)
        Gmin[:, i] = htl[:, i] + Conv - Lmin

    # Get input levels
    Lmid = speech_level_65_in_dc_bands

    # Calculate gains at centre frequencies
    Gmid = gainrule_camfit_linear(
        audiogram, sFitmodel, noisegatelevels, noisegateslope, max_output_level
    )
    insertion_gains = Gmid["insertion_gains"]
    Gmid = Gmid["sGt"]

    # Calculate compression ratios
    compression_ratio = np.zeros((np.size(frequencies), 2))
    sGt = np.zeros((len(sFitmodel_levels), len(sFitmodel_frequencies), 2))

    noisegate_level = np.zeros((np.size(sFitmodel_frequencies), 2))
    noisegate_slope = np.zeros((np.size(sFitmodel_frequencies), 2))

    # Find index corresponding to input level in dB for compression rate calculation
    if level != 0:
        cr_idx = [i for (i, val) in enumerate(sFitmodel_levels) if val == level]

    for i, levels in enumerate([audiogram.levels_l, audiogram.levels_r]):
        if level != 0:
            tmp = Lmid + Gmid[cr_idx, :, i].flatten() - Lmin - Gmin[:, i]
        else:
            tmp = Lmid + insertion_gains[:, i] - Lmin - Gmin[:, i]
        idx = [i for i, x in enumerate(tmp < 13) if x]
        tmp[idx] = 13
        compression_ratio[:, i] = minima_distance / tmp
        compression_ratio[:, i][compression_ratio[:, i] < 1] = 1

        sGt[:, :, i] = gains(
            Lmin, Gmin[:, i], compression_ratio[:, i], sFitmodel_levels
        )
        # Set negative gains to zero
        sGt[:, :, i][sGt[:, :, i] < 0] = 0
        # where output level is greater than max_output_level, reduce gain
        tmp = np.tile(sFitmodel_levels, (len(Gmin[:, i]), 1))
        tmp = np.transpose(tmp)
        output_levels = sGt[:, :, i] + tmp

        safe_output_levels = copy.deepcopy(output_levels)
        safe_output_levels[safe_output_levels > max_output_level] = max_output_level
        sGt[:, :, i] = sGt[:, :, i] - (output_levels - safe_output_levels)

        # set all gains to 0 for 0dB HL flat audiogram
        sGt[:, :, i] = sGt[:, :, i] * levels.any()

        noisegate_level[:, i] = noisegatelevels * np.ones(
            np.size(sFitmodel_frequencies)
        )
        noisegate_slope[:, i] = noisegateslope * np.ones(np.size(sFitmodel_frequencies))

    logging.info("Noisegate levels are %s", noisegate_level)

    output = {}
    output["sGt"] = sGt
    output["noisegatelevel"] = noisegate_level
    output["noisegateslope"] = noisegate_slope

    return output
