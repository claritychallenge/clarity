from clarity.evaluator.haspi import eb


def hasqi_v2(x, fx, y, fy, HL, eq=1, level1=65):
    """
    Function to compute the HASQI version 2 quality index using the
    auditory model followed by computing the envelope cepstral
    correlation and BM vibration average short-time coherence signals.
    The reference signal presentation level for NH listeners is assumed
    to be 65 dB SPL. The same model is used for both normal and
    impaired hearing.

    Arguments:
    x			Clear input reference speech signal with no noise or distortion.
              If a hearing loss is specified, NAL-R equalization is optional
    fx        Sampling rate in Hz for signal x
    y			Output signal with noise, distortion, HA gain, and/or processing.
    fy        Sampling rate in Hz for signal y.
    HL		(1,6) vector of hearing loss at the 6 audiometric frequencies
                  [250, 500, 1000, 2000, 4000, 6000] Hz.
    eq        Flag to provide equalization for the hearing loss to signal x:
                1 = no EQ has been provided, the function will add NAL-R
                2 = NAL-R EQ has already been added to the reference signal
    Level1    Optional input specifying level in dB SPL that corresponds to a
              signal RMS = 1. Default is 65 dB SPL if argument not provided.
    Returns:
    Combined  Quality estimate is the product of the nonlinear and linear terms
    Nonlin    Nonlinear quality component = (cepstral corr)^2 x seg BM coherence
    Linear    Linear quality component = std of spectrum and spectrum slope
    raw       Vector of raw values = [CepCorr, BMsync5, Dloud, Dslope]

    James M. Kates, 5 August 2013.
    Translated from MATLAB to Python by Gerardo Roa Dabike, October 2022.
    """

    # Auditory model for quality
    # Reference is no processing or NAL-R, impaired hearing
    xenv, xBM, yenv, yBM, xSL, ySL, fsamp = eb.EarModel(x, fx, y, fy, HL, eq, level1)

    # Envelope and long-term average spectral features
    # Smooth the envelope outputs: 125 Hz sub-sampling rate
    segsize = 16  # Averaging segment size in msec
    xdB = eb.env_smooth(xenv, segsize, fsamp)
    ydB = eb.env_smooth(yenv, segsize, fsamp)

    # Mel cepstrum correlation using smoothed envelopes
    # m1=ave of coefficients 2-6
    # xy=vector of coefficients 1-6
    thr = 2.5  # Silence threshold: sum across bands, dB above aud threshold
    addnoise = 0.0  # Additive noise in dB SL to condition cross-covariances
    cep_corr, xy = eb.melcor(xdB, ydB, thr, addnoise)

    # Linear changes in the log-term spectra
    # dloud  vector: [sum abs diff, std dev diff, max diff] spectra
    # dnorm  vector: [sum abs diff, std dev diff, max diff] norm spectra
    # dslope vector: [sum abs diff, std dev diff, max diff] slope
    dloud, dnorm, dslope = eb.spect_diff(xSL, ySL)

    # Temporal fine structure correlation measurements
    # Compute the time-frequency segment covariances
    segcov = 16  # Segment size for the covariance calculation
    sigcov, sigMSx, sigMSy = eb.bm_covary(xBM, yBM, segcov, fsamp)

    # Average signal segment cross-covariance
    # avecov=weighted ave of cross-covariances, using only data above threshold
    # syncov=ave cross-covariance with added IHC loss of synchronization at HF
    thr = 2.5  # Threshold in dB SL for including time-freq tile
    avecov, syncov = eb.ave_covary2(sigcov, sigMSx, thr)
    bm_sync5 = syncov[4]  # Ave segment coherence with IHC loss of sync

    # Extract and normalize the spectral features
    # Dloud:std
    d = dloud[1]  # Loudness difference std
    d = d / 2.5  # Scale the value
    d = 1.0 - d  # 1=perfect, 0=bad
    d = min(d, 1)
    d = max(d, 0)
    d_loud = d

    # Dslope:std
    d = dslope[1]  # Slope difference std
    d = 1.0 - d
    d = min(d, 1)
    d = max(d, 0)
    d_slope = d

    # Construct the models
    # Nonlinear model
    non_lin = (
        cep_corr**2
    ) * bm_sync5  # Combined envelope and temporal fine structure
    # Linear model
    linear = 0.579 * d_loud + 0.421 * d_slope  # Linear fit

    # Combined model
    combined = non_lin * linear  # Product of nonlinear x linear

    # Raw data
    raw = [cep_corr, bm_sync5, d_loud, d_slope]
    return combined, non_lin, linear, raw


def hasqi_v2_be(
    xl, xr, yl, yr, fs_signal, audiogram_l, audiogram_r, audiogram_cfs, level=100
) -> float:
    """Better ear HASQI.

    Calculates HASQI for left and right ear and selects the better result.

    Args:
        xl: left channel of reference signal
        xr: right channel of reference signal
        yl: left channel of processed signal
        yr: right channel of processed signal
        fs_signal: sampling rate for both signal
        audiogram_l: left ear audiogram
        audiogram_r: right ear audiogram
        audiogram_cfs: audiogram frequencies
        level: level in dB SPL corresponding to RMS=1

    Returns:
        float: beHASQI score

    Gerardo Roa Dabike, November 2022
    """

    # HASQI assumes the following audiogram frequencies:
    aud = [250, 500, 1000, 2000, 4000, 6000]

    # Adjust listener.audiogram_levels_l and _r to match the frequencies above
    hl_l = [
        audiogram_l[i] for i in range(len(audiogram_cfs)) if audiogram_cfs[i] in aud
    ]
    hl_r = [
        audiogram_r[i] for i in range(len(audiogram_cfs)) if audiogram_cfs[i] in aud
    ]

    score_l, _, _, _ = hasqi_v2(xl, fs_signal, yl, fs_signal, hl_l, eq=1, level1=level)
    score_r, _, _, _ = hasqi_v2(xr, fs_signal, yr, fs_signal, hl_r, eq=1, level1=level)

    return max(score_l, score_r)
