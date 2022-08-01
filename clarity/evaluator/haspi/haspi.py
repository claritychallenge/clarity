from clarity.evaluator.haspi import eb, ebm, ip


def haspi_v2(x, fx, y, fy, HL, Level1=65):
    """
    Compute the HASPI intelligibility index using the
    auditory model followed by computing the envelope cepstral
    correlation and BM vibration high-level covariance. The reference
    signal presentation level for NH listeners is assumed to be 65 dB
    SPL. The same model is used for both normal and impaired hearing. This
    version of HASPI uses a modulation filterbank followed by an ensemble of
    neural networks to compute the estimated intelligibility.

    Args:
        x (): Clear input reference speech signal with no noise or distortion.
              If a hearing loss is specified, no amplification should be provided.
        fx (): Sampling rate in Hz for signal x
        y (): Output signal with noise, distortion, HA gain, and/or processing.
        fy (): Sampling rate in Hz for signal y.
        HL (): (1,6) vector of hearing loss at the 6 audiometric frequencies
                    [250, 500, 1000, 2000, 4000, 6000] Hz.
        Level1 (): Optional input specifying level in dB SPL that corresponds to a
              signal RMS = 1. Default is 65 dB SPL if argument not provided.

    Returns:
    Intel     Intelligibility estimated by passing the cepstral coefficients
              through a modulation filterbank followed by an ensemble of
              neural networks.
    raw       vector of 10 cep corr modulation filterbank outputs, averaged
              over basis funct 2-6.

    James M. Kates, 5 August 2013.

    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Auditory model for intelligibility
    # Reference is no processing, normal hearing
    itype = 0  # Intelligibility model
    xenv, _, yenv, _, _, _, fsamp = eb.EarModel(x, fx, y, fy, HL, itype, Level1)

    # Envelope modulation features

    # LP filter and subsample the envelope
    fLP = 320
    fsub = 8 * fLP  # subsample to span 2 octaves above the cutoff frequency
    xLP, yLP = ebm.EnvFilt(xenv, yenv, fLP, fsub, fsamp)

    # Compute the ceptstral coefficients as a function of subsampled time
    nbasis = 6  # Use 6 basis functions
    thr = 2.5  # Silence threshold in dB SL
    dither = 0.1  # Dither in dB RMS to add to envelope signals
    xcep, ycep = ebm.CepCoef(xLP, yLP, thr, dither, nbasis)

    # Cepstral coeffifiencts filtered at each modulation rate
    # Band center frequencies [2, 6, 10, 16, 25, 40, 64, 100, 160, 256] Hz
    # Band edges [0, 4, 8, 12.5, 20.5, 30.5, 52.4, 78.1, 128, 200, 328] Hz
    xmod, ymod, _ = ebm.ModFilt(xcep, ycep, fsub)

    # Cross-correlation between the cepstral coefficients for the degraded and
    # ref signals at each modulation rate, averaged over basis functions 2-6
    aveCM = ebm.ModCorr(xmod, ymod)

    # Intelligibility prediction
    # Get the neural network parameters and the weights for an ensemble of 10 networks
    NNparam, Whid, Wout, b = ip.GetNeuralNet()

    # Average the neural network outputs for the modulation filterbank values
    model = ip.NNfeedfwdEns(aveCM, NNparam, Whid, Wout)
    model = model[0] / b

    # Return the intelligibility estimate and raw modulation filter outputs
    Intel = model
    raw = aveCM

    return Intel, raw


def haspi_v2_be(
    xl, xr, yl, yr, fs_signal, audiogram_l, audiogram_r, audiogram_cfs, Level=100
) -> float:
    """Better ear HASPI.

    Calculates HASPI for left and right ear and selects the better result.

    Args:
        xl: left channel of reference signal
        xr: right channel of reference signal
        yl: left channel of processed signal
        yr: right channel of processed signal
        fs_signal: sampling rate for both signal
        audiogram_l: left ear audiogram
        audiogram_r: right ear audiogram
        audiogram_cfs: audiogram frequencies
        Level: level in dB SPL corresponding to RMS=1

    Returns:
        float: beHASPI score

    Zuzanna Podwinska, March 2022
    """

    # HASPI assumes the following audiogram frequencies:
    aud = [250, 500, 1000, 2000, 4000, 6000]

    # Adjust listener.audiogram_levels_l and _r to match the frequencies above
    HL_l = [
        audiogram_l[i] for i in range(len(audiogram_cfs)) if audiogram_cfs[i] in aud
    ]
    HL_r = [
        audiogram_r[i] for i in range(len(audiogram_cfs)) if audiogram_cfs[i] in aud
    ]

    score_l, _ = haspi_v2(xl, fs_signal, yl, fs_signal, HL_l, Level)
    score_r, _ = haspi_v2(xr, fs_signal, yr, fs_signal, HL_r, Level)

    return max(score_l, score_r)
