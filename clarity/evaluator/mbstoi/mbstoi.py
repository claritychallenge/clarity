import numpy as np
import logging
import math
from scipy.signal import resample

from clarity.evaluator.mbstoi.mbstoi_utils import (
    ec,
    stft,
    remove_silent_frames,
    thirdoct,
)


def mbstoi(xl, xr, yl, yr, fs_signal, gridcoarseness=1):
    """A Python implementation of the Modified Binaural Short-Time
    Objective Intelligibility (MBSTOI) measure as described in:
    A. H. Andersen, J. M. de Haan, Z.-H. Tan, and J. Jensen, “Refinement
    and validation of the binaural short time objective intelligibility
    measure for spatially diverse conditions,” Speech Communication,
    vol. 102, pp. 1-13, Sep. 2018. A. H. Andersen, 10/12-2018

    All title, copyrights and pending patents in and to the original MATLAB
    Software are owned by Oticon A/S and/or Aalborg University. Please see
    details at http://ah-andersen.net/code/

    Args:
        xl (ndarray): clean speech signal from left ear
        xr (ndarray): clean speech signal from right ear.
        yl (ndarray): noisy/processed speech signal from left ear.
        yr (ndarray): noisy/processed speech signal from right ear.
        gridcoarseness (integer): grid coarseness as denominator of ntaus and ngammas (default: 1)

    Returns
        float: MBSTOI index d

    """

    # Basic STOI parameters
    fs = 10000  # Sample rate of proposed intelligibility measure in Hz
    N_frame = 256  # Window support in samples
    K = 512  # FFT size in samples
    J = 15  # Number of one-third octave bands
    mn = 150  # Centre frequency of first 1/3 octave band in Hz
    N = 30  # Number of frames for intermediate intelligibility measure (length analysis window)
    dyn_range = 40  # Speech dynamic range in dB

    # Values to define EC grid
    tau_min = -0.001  # Minimum interaural delay compensation in seconds. B: -0.01.
    tau_max = 0.001  # Maximum interaural delay compensation in seconds. B: 0.01.
    ntaus = math.ceil(100 / gridcoarseness)  # Number of tau values to try out
    gamma_min = -20  # Minimum interaural level compensation in dB
    gamma_max = 20  # Maximum interaural level compensation in dB
    ngammas = math.ceil(40 / gridcoarseness)  # Number of gamma values to try out

    # Constants for jitter
    # ITD compensation standard deviation in seconds. Equation 6 Andersen et al. 2018 Refinement
    sigma_delta_0 = 65e-6
    # ILD compensation standard deviation.  Equation 5 Andersen et al. 2018
    sigma_epsilon_0 = 1.5
    # Constant for level shift deviation in dB. Equation 5 Andersen et al. 2018
    alpha_0_db = 13
    # Constant for time shift deviation in seconds. Equation 6 Andersen et al. 2018
    tau_0 = 1.6e-3
    # Constant for level shift deviation. Power for calculation of sigma delta gamma in equation 5 Andersen et al. 2018.
    p = 1.6

    # Prepare signals, ensuring that inputs are column vectors
    xl = xl.flatten()
    xr = xr.flatten()
    yl = yl.flatten()
    yr = yr.flatten()

    # Resample signals to 10 kHz
    if fs_signal != fs:

        logging.debug(f"Resampling signals with sr={fs} for MBSTOI calculation.")
        # Assumes fs_signal is 44.1 kHz
        el = len(xl)
        xl = resample(xl, int(el * (fs / fs_signal) + 1))
        xr = resample(xr, int(el * (fs / fs_signal) + 1))
        yl = resample(yl, int(el * (fs / fs_signal) + 1))
        yr = resample(yr, int(el * (fs / fs_signal) + 1))

    # Remove silent frames
    [xl, xr, yl, yr] = remove_silent_frames(
        xl, xr, yl, yr, dyn_range, N_frame, N_frame / 2
    )

    # Handle case when signals are zeros
    if (
        abs(np.log10(np.linalg.norm(xl) / np.linalg.norm(yl))) > 5.0
        or abs(np.log10(np.linalg.norm(xr) / np.linalg.norm(yr))) > 5.0
    ):
        sii = 0

    # STDFT and filtering
    # Get 1/3 octave band matrix
    [H, cf, fids, freq_low, freq_high] = thirdoct(
        fs, K, J, mn
    )  # (fs, nfft, num_bands, min_freq)
    cf = 2 * math.pi * cf  # This is now the angular frequency in radians per sec

    # Apply short time DFT to signals and transpose
    xl_hat = stft(xl, N_frame, K).transpose()
    xr_hat = stft(xr, N_frame, K).transpose()
    yl_hat = stft(yl, N_frame, K).transpose()
    yr_hat = stft(yr, N_frame, K).transpose()

    # Take single sided spectrum of signals
    idx = int(K / 2 + 1)
    xl_hat = xl_hat[0:idx, :]
    xr_hat = xr_hat[0:idx, :]
    yl_hat = yl_hat[0:idx, :]
    yr_hat = yr_hat[0:idx, :]

    # Compute intermediate correlation via EC search
    logging.info(f"Starting EC evaluation")
    # Here intermeduiate correlation coefficients are evaluated for a discrete set of
    # gamma and tau values (a "grid") and the highest value is chosen.
    d = np.zeros((J, np.shape(xl_hat)[1] - N + 1))
    p_ec_max = np.zeros((J, np.shape(xl_hat)[1] - N + 1))

    # Interaural compensation time and level values
    taus = np.linspace(tau_min, tau_max, ntaus)
    gammas = np.linspace(gamma_min, gamma_max, ngammas)

    # Jitter incorporated below - Equations 5 and 6 in Andersen et al. 2018
    sigma_epsilon = (
        np.sqrt(2) * sigma_epsilon_0 * (1 + (abs(gammas) / alpha_0_db) ** p) / 20
    )
    gammas = gammas / 20
    sigma_delta = np.sqrt(2) * sigma_delta_0 * (1 + (abs(taus) / tau_0))

    logging.info(f"Processing EC stage")
    d, p_ec_max = ec(
        xl_hat,
        xr_hat,
        yl_hat,
        yr_hat,
        J,
        N,
        fids,
        cf.flatten(),
        taus,
        ntaus,
        gammas,
        ngammas,
        d,
        p_ec_max,
        sigma_epsilon,
        sigma_delta,
    )

    # Compute the better ear STOI
    logging.info(f"Computing better ear intermediate correlation coefficients")
    # Arrays for the 1/3 octave envelope
    Xl = np.zeros((J, np.shape(xl_hat)[1]))
    Xr = np.zeros((J, np.shape(xl_hat)[1]))
    Yl = np.zeros((J, np.shape(xl_hat)[1]))
    Yr = np.zeros((J, np.shape(xl_hat)[1]))

    # Apply 1/3 octave bands as described in Eq.(1) of the STOI article
    for k in range(np.shape(xl_hat)[1]):
        Xl[:, k] = np.dot(H, abs(xl_hat[:, k]) ** 2)
        Xr[:, k] = np.dot(H, abs(xr_hat[:, k]) ** 2)
        Yl[:, k] = np.dot(H, abs(yl_hat[:, k]) ** 2)
        Yr[:, k] = np.dot(H, abs(yr_hat[:, k]) ** 2)

    # Arrays for better-ear correlations
    dl_interm = np.zeros((J, len(range(N, len(xl_hat[1]) + 1))))
    dr_interm = np.zeros((J, len(range(N, len(xl_hat[1]) + 1))))
    pl = np.zeros((J, len(range(N, len(xl_hat[1]) + 1))))
    pr = np.zeros((J, len(range(N, len(xl_hat[1]) + 1))))

    # Compute temporary better-ear correlations
    for m in range(N, np.shape(xl_hat)[1]):
        Xl_seg = Xl[:, (m - N) : m]
        Xr_seg = Xr[:, (m - N) : m]
        Yl_seg = Yl[:, (m - N) : m]
        Yr_seg = Yr[:, (m - N) : m]

        for n in range(J):
            xln = Xl_seg[n, :] - np.sum(Xl_seg[n, :]) / N
            xrn = Xr_seg[n, :] - np.sum(Xr_seg[n, :]) / N
            yln = Yl_seg[n, :] - np.sum(Yl_seg[n, :]) / N
            yrn = Yr_seg[n, :] - np.sum(Yr_seg[n, :]) / N
            pl[n, m - N] = np.sum(xln * xln) / np.sum(yln * yln)
            pr[n, m - N] = np.sum(xrn * xrn) / np.sum(yrn * yrn)
            dl_interm[n, m - N] = np.sum(xln * yln) / (
                np.linalg.norm(xln) * np.linalg.norm(yln)
            )
            dr_interm[n, m - N] = np.sum(xrn * yrn) / (
                np.linalg.norm(xrn) * np.linalg.norm(yrn)
            )

    # Get the better ear intermediate coefficients
    idx = np.isfinite(dl_interm)
    dl_interm[~idx] = 0
    idx = np.isfinite(dr_interm)
    dr_interm[~idx] = 0
    p_be_max = np.maximum(pl, pr)
    dbe_interm = np.zeros((np.shape(dl_interm)))

    idx = pl > pr
    dbe_interm[idx] = dl_interm[idx]
    dbe_interm[~idx] = dr_interm[~idx]

    # Compute STOI measure
    # Whenever a single ear provides a higher correlation than the corresponding EC
    # processed alternative,the better-ear correlation is used.
    idx = p_be_max > p_ec_max
    d[idx] = dbe_interm[idx]
    sii = np.mean(d)

    logging.info("MBSTOI processing complete")

    return sii
