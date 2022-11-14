"""Matlab's haaqi version 1 to python version."""
import numpy as np

from clarity.evaluator.haspi import eb


def haaqi_v1(
    x: np.ndarray,
    fx: int,
    y: np.ndarray,
    fy: int,
    hl: np.ndarray,
    eq: int,
    level1: int = 65,
):
    """
    Compute the HAAQI music quality index using the
    auditory model followed by computing the envelope cepstral
    correlation and BM vibration average short-time coherence signals.
    The reference signal presentation level for NH listeners is assumed
    to be 65 dB SPL. The same model is used for both normal and
    impaired hearing.

    Arguments:
        x (ndarray):  Clear input reference speech signal with no noise or distortion.
            If a hearing loss is specified, NAL-R equalization is optional
        fx: Sampling rate in Hz for signal x
        y:  Output signal with noise, distortion, HA gain, and/or processing.
        fy: Sampling rate in Hz for signal y.
        hl: (1,6) vector of hearing loss at the 6 audiometric frequencies
        [250, 500, 1000, 2000, 4000, 6000] Hz.
        eq: Flag to provide equalization for the hearing loss to signal x:
            1 = no EQ has been provided, the function will add NAL-R
            2 = NAL-R EQ has already been added to the reference signal
        level1: Optional input specifying level in dB SPL that corresponds to a
           signal RMS = 1. Default is 65 dB SPL if argument not provided.
           Default: 65

    Returns:
        combined : Quality is the polynomial sum of the nonlin and linear terms
        nonlin : Nonlinear quality component = .245(BMsync5) + .755(CepHigh)^3
        linear : Linear quality component = std of spectrum and norm spectrum
        raw : Vector of raw values = [cephigh, bmsync5, dloud, dnorm]

    James M. Kates, 5 August 2013 (HASQI_v2).
    Version for HAAQI_v1, 19 Feb 2015.

    Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Auditory model for quality
    # Reference is no processing or NAL-R, impaired hearing
    xenv, xbm, yenv, ybm, xsl, ysl, fsamp = eb.EarModel(x, fx, y, fy, hl, eq, level1)

    # ---------------------------------------
    # Envelope and long-term average spectral features
    # Smooth the envelope outputs: 250 Hz sub-sampling rate
    segsize = 8  # Averaging segment size in msec
    xdb = eb.env_smooth(xenv, segsize, fsamp)
    ydb = eb.env_smooth(yenv, segsize, fsamp)

    # Mel cepstrum correlation after passing through modulation filterbank
    thr = 2.5  # Silence threshold: sum across bands, dB above aud threshold
    addnoise = 0.0  # dditive noise in dB SL to condition cross-covariances
    _, _, cephigh, _ = eb.melcor9(
        xdb, ydb, thr, addnoise, segsize
    )  # 8 modulation freq bands

    # Linear changes in the long-term spectra
    # dloud  vector: [sum abs diff, std dev diff, max diff] spectra
    # dnorm  vector: [sum abs diff, std dev diff, max diff] norm spectra
    # dslope vector: [sum abs diff, std dev diff, max diff] slope
    dloud_vector, dnorm_vector, _ = eb.spect_diff(xsl, ysl)

    # Temporal fine structure (TFS) correlation measurements
    # Compute the time-frequency segment covariances
    segcov = 16  # Segment size for the covariance calculation
    sigcov, sigmsx, _ = eb.bm_covary(xbm, ybm, segcov, fsamp)

    # Average signal segment cross-covariance
    # avecov=weighted ave of cross-covariances, using only data above threshold
    # syncov=ave cross-covariance with added IHC loss of synchronization at HF
    thr = 2.5  # Threshold in dB SL for including time-freq tile
    _, syncov = eb.ave_covary2(sigcov, sigmsx, thr)
    bmsync5 = syncov[4]  # Ave segment coherence with IHC loss of sync

    # Extract and normalize the spectral features
    # Dloud:std
    d = dloud_vector[1]  # Loudness difference std
    d = d / 2.5  # Scale the value
    d = 1.0 - d  # 1=perfect, 0=bad
    d = min(d, 1)
    d = max(d, 0)
    dloud = d

    # Dnorm:std
    d = dnorm_vector[1]  # Slope difference std
    d = d / 25  # Scale the value
    d = 1.0 - d  # 1=perfect, 0=bad
    d = min(d, 1)
    d = max(d, 0)
    dnorm = d

    # Construct the models
    # Nonlinear model
    nonlin_model = 0.754 * (cephigh**3) + 0.246 * bmsync5  # Combined envelope and TFS

    # Linear model
    lin_model = 0.329 * dloud + 0.671 * dnorm  # Linear fit

    # Combined model
    combined = (
        0.336 * nonlin_model
        + 0.001 * lin_model
        + 0.501 * (nonlin_model**2)
        + 0.161 * (lin_model**2)
    )  # Polynomial sum

    # Raw data
    raw = [cephigh, bmsync5, dloud, dnorm]

    return combined, nonlin_model, lin_model, raw
