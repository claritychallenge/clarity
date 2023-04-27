"""HASPI intelligibility Index"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import numpy as np

from clarity.evaluator.haspi.eb import ear_model
from clarity.evaluator.haspi.ebm import (
    cepstral_correlation_coef,
    env_filter,
    fir_modulation_filter,
    modulation_cross_correlation,
)
from clarity.evaluator.haspi.ip import get_neural_net, nn_feed_forward_ensemble
from clarity.utils.audiogram import Audiogram, Listener

if TYPE_CHECKING:
    from numpy import ndarray


# HASPI assumes the following audiogram frequencies:
HASPI_AUDIOGRAM_FREQUENCIES: Final = np.array([250, 500, 1000, 2000, 4000, 6000])


def haspi_v2(  # pylint: disable=too-many-arguments too-many-locals
    reference: ndarray,
    reference_sample_rate: float,
    processed: ndarray,
    processed_sample_rate: float,
    audiogram: Audiogram,
    level1: float = 65.0,
    f_lp: float = 320.0,
    itype: int = 0,
) -> tuple[float, ndarray]:
    """
    Compute the HASPI intelligibility index using the
    auditory model followed by computing the envelope cepstral
    correlation and BM vibration high-level covariance. The reference
    signal presentation level for NH listeners is assumed to be 65 dB
    SPL. The same model is used for both normal and impaired hearing. This
    version of HASPI uses a modulation filterbank followed by an ensemble of
    neural networks to compute the estimated intelligibility.

    **NB** - The original HASPI model derivation included a bug which meant that
    although the 'shift' parameter used in band centre frequency calculations was set to
    '0.02' it was never actually applied. To replicate this behaviour ear_model is
    called with 'shift' set to None.  For discussion please refer to the discussion in
    `Issue #105 <https://github.com/claritychallenge/clarity/issues/105>`
    for further details.

    Args:
        reference (np.ndarray): Clear input reference speech signal with no noise or
            distortion. If a hearing loss is specified, no amplification should be
            provided.
        reference_sample_rate (int): Sampling rate in Hz for signal x
        processed (np.ndarray): Output signal with noise, distortion, HA gain, and/or
            processing.
        processed_sample_rate (int): Sampling rate in Hz for signal y.
        hearing_loss (np.ndarray): (1,6) vector of hearing loss at the 6 audiometric
            frequencies [250, 500, 1000, 2000, 4000, 6000] Hz.
        level1 (int): Optional input specifying level in dB SPL that corresponds to a
            signal RMS = 1. Default is 65 dB SPL if argument not provided.
        f_lp (int):
        itype (int): Intelligibility model

    Returns:
        tuple(Intel: float, raw: nd-array)
        Intel: Intelligibility estimated by passing the cepstral coefficients
              through a modulation filterbank followed by an ensemble of
              neural networks.
        raw: vector of 10 cep corr modulation filterbank outputs, averaged
              over basis functions 2-6.

    Updates:
        James M. Kates, 5 August 2013.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    if not audiogram.has_frequencies(HASPI_AUDIOGRAM_FREQUENCIES):
        logging.warning(
            "Audiogram does not have all HASPI frequency measurements"
            "Measurements will be interpolated"
        )

    # Adjust audiogram to match the standard frequencies
    audiogram = audiogram.resample(HASPI_AUDIOGRAM_FREQUENCIES)

    # Auditory model for intelligibility
    # Reference is no processing, normal hearing
    reference_env, _, processed_env, _, _, _, fsamp = ear_model(
        reference,
        reference_sample_rate,
        processed,
        processed_sample_rate,
        audiogram.levels,
        itype,
        level1,
        # shift=0.02 # See comment in docstring
        shift=None,
    )

    # Envelope modulation features

    # LP filter and subsample the envelope
    fsub = 8.0 * f_lp  # subsample to span 2 octaves above the cutoff frequency
    reference_lp, processed_lp = env_filter(
        reference_env, processed_env, f_lp, fsub, fsamp
    )

    # Compute the cepstral coefficients as a function of subsampled time
    nbasis = 6  # Use 6 basis functions
    thr = 2.5  # Silence threshold in dB SL
    dither = 0.1  # Dither in dB RMS to add to envelope signals
    reference_cep, processed_cep = cepstral_correlation_coef(
        reference_lp, processed_lp, thr, dither, nbasis
    )

    # Cepstral coefficients filtered at each modulation rate
    # Band center frequencies [2, 6, 10, 16, 25, 40, 64, 100, 160, 256] Hz
    # Band edges [0, 4, 8, 12.5, 20.5, 30.5, 52.4, 78.1, 128, 200, 328] Hz
    reference_mod, processed_mod, _ = fir_modulation_filter(
        reference_cep, processed_cep, fsub
    )

    # Cross-correlation between the cepstral coefficients for the degraded and
    # ref signals at each modulation rate, averaged over basis functions 2-6
    average_correlation_matrix = modulation_cross_correlation(
        reference_mod, processed_mod
    )

    # Intelligibility prediction
    # Get the neural network parameters and the weights for an ensemble of 10 networks
    (
        neural_net_params,
        weights_hidden,
        weights_out,
        normalization_factor,
    ) = get_neural_net()

    # Average the neural network outputs for the modulation filterbank values
    model = nn_feed_forward_ensemble(
        average_correlation_matrix, neural_net_params, weights_hidden, weights_out
    )
    model = model / normalization_factor

    # Return the intelligibility estimate and raw modulation filter outputs
    return model[0], average_correlation_matrix


def haspi_v2_be(  # pylint: disable=too-many-arguments
    reference_left: ndarray,
    reference_right: ndarray,
    processed_left: ndarray,
    processed_right: ndarray,
    sample_rate: float,
    listener: Listener,
    level: float = 100.0,
) -> float:
    """Better ear HASPI.

    Calculates HASPI for left and right ear and selects the better result.

    Args:
        ref_left (np.ndarray): left channel of reference signal
        ref_right (np.ndarray): right channel of reference signal
        proc_left (np.ndarray): left channel of processed signal
        proc_right (np.ndarray): right channel of processed signal
        sample_rate (int): sampling rate for both signal
        audiogram_left (): left ear audiogram
        audiogram_right (): right ear audiogram
        level: level in dB SPL corresponding to RMS=1

    Returns:
        float: beHASPI score

    Updates:
        Zuzanna Podwinska, March 2022
    """

    score_left, _ = haspi_v2(
        reference_left,
        sample_rate,
        processed_left,
        sample_rate,
        listener.audiogram_left,
        level,
    )
    score_right, _ = haspi_v2(
        reference_right,
        sample_rate,
        processed_right,
        sample_rate,
        listener.audiogram_right,
        level,
    )

    return max(score_left, score_right)
