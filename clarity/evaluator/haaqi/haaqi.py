"""Matlab's haaqi version 1 to python version."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import numpy as np

from clarity.evaluator.haspi import eb
from clarity.utils.audiogram import Audiogram

if TYPE_CHECKING:
    from numpy import ndarray

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

logger = logging.getLogger(__name__)

# HAAQI assumes the following audiogram frequencies:
HAAQI_AUDIOGRAM_FREQUENCIES: Final = np.array([250, 500, 1000, 2000, 4000, 6000])


def haaqi_v1(
    reference: ndarray,
    reference_freq: float,
    processed: ndarray,
    processed_freq: float,
    audiogram: Audiogram,
    equalisation: int,
    level1: float = 65.0,
    silence_threshold: float = 2.5,
    add_noise: float = 0.0,
    segment_covariance: int = 16,
) -> tuple[float, float, float, list[float]]:
    """
    Compute the HAAQI music quality index using the auditory model followed by
    computing the envelope cepstral correlation and Basilar Membrane vibration
    average short-time coherence signals.

    The reference signal presentation level for NH listeners is assumed
    to be 65 dB SPL. The same model is used for both normal and
    impaired hearing.

    Arguments:
        reference (ndarray):  Input reference speech signal with no noise or distortion.
            If a hearing loss is specified, NAL-R equalization is optional
        reference_freq (int): Sampling rate in Hz for reference signal.
        processed (np.ndarray):  Output signal with noise, distortion, HA gain,
            and/or processing.
        processed_freq (int): Sampling rate in Hz for processed signal.
        hearing_loss (np.ndarray): (1,6) vector of hearing loss at the 6 audiometric
            frequencies [250, 500, 1000, 2000, 4000, 6000] Hz.
        equalisation (int): hearing loss equalization mode for reference signal:
            1 = no EQ has been provided, the function will add NAL-R
            2 = NAL-R EQ has already been added to the reference signal
        level1 (int): Optional input specifying level in dB SPL that corresponds to a
           signal RMS = 1. Default is 65 dB SPL if argument not provided.
           Default: 65
        silence_threshold (float): Silence threshold sum across bands,
            dB above auditory threshold. Default : 2.5
        add_noise (float): Additive noise dB SL to condition cross-covariances.
            Defaults to 0.0
        segment_covariance (int): Segment size for the covariance calculation.
            Defaults to 16

    Returns:
        combined : Quality is the polynomial sum of the nonlinear and linear terms
        nonlinear : Nonlinear quality component = .245(BMsync5) + .755(CepHigh)^3
        linear : Linear quality component = std of spectrum and norm spectrum
        raw : Vector of raw values = [cephigh, bmsync5, dloud, dnorm]

    James M. Kates, 5 August 2013 (HASQI_v2).
    Version for HAAQI_v1, 19 Feb 2015.

    Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    if not audiogram.has_frequencies(HAAQI_AUDIOGRAM_FREQUENCIES):
        logging.warning(
            "Audiogram does not have all HAAQI frequency measurements"
            "Measurements will be interpolated"
        )

    audiogram = audiogram.resample(HAAQI_AUDIOGRAM_FREQUENCIES)

    # Auditory model for quality
    # Reference is no processing or NAL-R, impaired hearing
    (
        reference_db,
        reference_basilar_membrane,
        processed_db,
        processed_basilar_membrane,
        reference_sl,
        processed_sl,
        sample_rate,
    ) = eb.ear_model(
        reference,
        reference_freq,
        processed,
        processed_freq,
        audiogram.levels,
        equalisation,
        level1,
    )

    # ---------------------------------------
    # Envelope and long-term average spectral features
    # Smooth the envelope outputs: 250 Hz sub-sampling rate
    segment_size = 8  # Averaging segment size in msec
    reference_smooth = eb.env_smooth(reference_db, segment_size, sample_rate)
    processed_smooth = eb.env_smooth(processed_db, segment_size, sample_rate)

    # Mel cepstrum correlation after passing through modulation filterbank
    _, _, mel_cepstral_high, _ = eb.melcor9(
        reference_smooth, processed_smooth, silence_threshold, add_noise, segment_size
    )  # 8 modulation freq bands

    # Linear changes in the long-term spectra
    # dloud  vector: [sum abs diff, std dev diff, max diff] spectra
    # dnorm  vector: [sum abs diff, std dev diff, max diff] norm spectra
    # dslope vector: [sum abs diff, std dev diff, max diff] slope
    dloud_stats, dnorm_stats, _ = eb.spectrum_diff(reference_sl, processed_sl)

    # Temporal fine structure (TFS) correlation measurements
    # Compute the time-frequency segment covariances
    signal_cross_covariance, reference_mean_square, _ = eb.bm_covary(
        reference_basilar_membrane,
        processed_basilar_membrane,
        segment_covariance,
        sample_rate,
    )

    # Average signal segment cross-covariance
    # avecov=weighted ave of cross-covariances, using only data above threshold
    # syncov=ave cross-covariance with added IHC loss of synchronization at HF
    _, ihc_sync_covariance = eb.ave_covary2(
        signal_cross_covariance, reference_mean_square, silence_threshold
    )
    basilar_membrane_sync5 = ihc_sync_covariance[
        4
    ]  # Ave segment coherence with IHC loss of sync

    # Extract and normalize the spectral features
    # Dloud:std
    d_loud = dloud_stats[1] / 2.5  # Loudness difference std
    d_loud = 1.0 - d_loud  # 1=perfect, 0=bad
    d_loud = min(d_loud, 1)
    d_loud = max(d_loud, 0)

    # Dnorm:std
    d_norm = dnorm_stats[1] / 25  # Slope difference std
    d_norm = 1.0 - d_norm  # 1=perfect, 0=bad
    d_norm = min(d_norm, 1)
    d_norm = max(d_norm, 0)

    # Construct the models
    # Nonlinear model - Combined envelope and TFS
    nonlinear_model = 0.754 * (mel_cepstral_high**3) + 0.246 * basilar_membrane_sync5

    # Linear model
    linear_model = 0.329 * d_loud + 0.671 * d_norm

    # Combined model
    combined_model = (
        0.336 * nonlinear_model
        + 0.001 * linear_model
        + 0.501 * (nonlinear_model**2)
        + 0.161 * (linear_model**2)
    )  # Polynomial sum

    # Raw data
    raw = [mel_cepstral_high, basilar_membrane_sync5, d_loud, d_norm]

    return combined_model, nonlinear_model, linear_model, raw


def compute_haaqi(
    processed_signal: ndarray,
    reference_signal: ndarray,
    processed_sample_rate: float,
    reference_sample_rate: float,
    audiogram: Audiogram,
    equalisation: int = 1,
    level1: float = 65.0,
) -> float:
    """Compute HAAQI metric

    Args:
        processed_signal (np.ndarray): Output signal with noise, distortion, HA gain,
            and/or processing.
        reference_signal (np.ndarray): Input reference speech signal with no noise
            or distortion. If a hearing loss is specified, NAL-R equalization
            is optional
        processed_sample_rate (float): Sampling rate in Hz for processed signal.
        reference_sample_rate (float): Sampling rate in Hz for reference signal.
        audiogram (Audiogram): Audiogram object.
        equalisation (int): hearing loss equalization mode for reference signal:
            1 = no EQ has been provided, the function will add NAL-R
            2 = NAL-R EQ has already been added to the reference signal
            Defaults to 1.
        level1 (float): Optional input specifying level in dB SPL
            that corresponds to a signal RMS = 1.
            Default is 65 dB SPL.
    """

    if len(reference_signal) == 0:
        if len(processed_signal) == 0:
            # No scoring if no music
            return 1.0
        logger.error("If `Reference` is empty, `Processed` must be empty as well")
        return 0.0

    score, _, _, _ = haaqi_v1(
        reference=reference_signal,
        reference_freq=reference_sample_rate,
        processed=processed_signal,
        processed_freq=processed_sample_rate,
        audiogram=audiogram,
        equalisation=equalisation,
        level1=level1,
    )
    return score
