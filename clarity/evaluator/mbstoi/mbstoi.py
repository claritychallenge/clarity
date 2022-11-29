"""Modified Binaural Short-Time Objective Intelligibility (MBSTOI) Measure"""
import importlib.resources as pkg_resources
import logging
import math

import numpy as np
import yaml  # type: ignore
from scipy.signal import resample

from clarity.evaluator.mbstoi.mbstoi_utils import (
    equalisation_cancellation,
    remove_silent_frames,
    stft,
    thirdoct,
)

# pylint: disable=too-many-locals


# basic stoi parameters from file
params_file = pkg_resources.open_text(__package__, "parameters.yaml")
basic_stoi_parameters = yaml.safe_load(params_file.read())


def mbstoi(
    left_ear_clean: np.ndarray,
    right_ear_clean: np.ndarray,
    left_ear_noisy: np.ndarray,
    right_ear_noisy: np.ndarray,
    fs_signal,
    gridcoarseness: int = 1,
    sample_rate: int = 10000,
    n_frame: int = 256,
    fft_size_in_samples: int = 512,
    n_third_octave_bands: int = 15,
    centre_freq_first_third_octave_hz: int = 150,
    n_frames: int = 30,
    dyn_range: int = 40,
    tau_min: float = -0.001,
    tau_max: float = 0.001,
    gamma_min: int = -20,
    gamma_max: int = 20,
    sigma_delta_0: float = 65e-6,
    sigma_epsilon_0: float = 1.5,
    alpha_0_db: int = 13,
    tau_0: float = 1.6e-3,
    level_shift_deviation: float = 1.6,
) -> float:
    """Implementation of the Modified Binaural Short-Time Objective Intelligibility (mbstoi) measure.

    Args:
        left_ear_clean (np.ndarray): Clean speech signal from left ear.
        right_ear_clean (np.ndarray): Clean speech signal from right ear.
        left_ear_noisy (np.ndarray) : Noisy/processed speech signal from left ear.
        right_ear_noisy (np.ndarray) : Noisy/processed speech signal from right ear.
        fs_signal (int) : Frequency sample rate of signal.
        gridcoarseness (int) : Grid coarseness as denominator of ntaus and ngammas (default: 1).
        sample_rate (int) :  Sample Rate.
        n_frame (int) :  Number of Frames.
        fft_size_in_samples (int) :  ??? size in samples.
        n_third_octave_bands (int) : Number of third octave bands.
        centre_freq_first_third_octave_hz (int) :  150,
        n_frames (int) :  Number of Frames.
        dyn_range (int) : Dynamic Range.
        tau_min (float) : Min Tau the ???
        tau_max (float) : Max Tau the ???
        gamma_min (int) : Minimum gamma the ???
        gamma_max (int) : Maximum gamma the ???
        sigma_delta_0 (float) : ???
        sigma_epsilon_0 (float) : ???
        alpha_0_db (int) : ???
        tau_0 (float) : ???
        level_shift_deviation (float) : ???

    Returns:
        float : mbtsoi index d.

    Notes:
        All title, copyrights and pending patents pertaining to mbtsoi[1]_ in and to the original Matlab software are
        owned by oticon a/s and/or Aalborg University. please see details at
        `http://ah-andersen.net/code/<http://ah-andersen.net/code/>`


    .. [1] A. H. Andersen, J. M. de Haan, Z.-H. Tan, and J. Jensen (2018) Refinement and validation of the binaural
    short time objective intelligibility measure for spatially diverse conditions. Speech Communication vol. 102,
    pp. 1-13 `doi:10.1016/j.specom.2018.06.001 <https://doi.org/10.1016/j.specom.2018.06.001>`_
    """

    n_taus = math.ceil(100 / gridcoarseness)  # number of tau values to try out
    n_gammas = math.ceil(40 / gridcoarseness)  # number of gamma values to try out

    # prepare signals, ensuring that inputs are column vectors
    left_ear_clean = left_ear_clean.flatten()
    right_ear_clean = right_ear_clean.flatten()
    left_ear_noisy = left_ear_noisy.flatten()
    right_ear_noisy = right_ear_noisy.flatten()

    # Resample signals to 10 kHz
    if fs_signal != sample_rate:

        logging.debug(
            "Resampling signals with sr=%s for MBSTOI calculation.", sample_rate
        )
        # Assumes fs_signal is 44.1 kHz
        length_left_ear_clean = len(left_ear_clean)
        left_ear_clean = resample(
            left_ear_clean, int(length_left_ear_clean * (sample_rate / fs_signal) + 1)
        )
        right_ear_clean = resample(
            right_ear_clean, int(length_left_ear_clean * (sample_rate / fs_signal) + 1)
        )
        left_ear_noisy = resample(
            left_ear_noisy, int(length_left_ear_clean * (sample_rate / fs_signal) + 1)
        )
        right_ear_noisy = resample(
            right_ear_noisy, int(length_left_ear_clean * (sample_rate / fs_signal) + 1)
        )

    # Remove silent frames
    (
        left_ear_clean,
        right_ear_clean,
        left_ear_noisy,
        right_ear_noisy,
    ) = remove_silent_frames(
        left_ear_clean,
        right_ear_clean,
        left_ear_noisy,
        right_ear_noisy,
        dyn_range,
        n_frame,
        n_frame / 2,
    )

    # Handle case when signals are zeros
    if (
        abs(np.log10(np.linalg.norm(left_ear_clean) / np.linalg.norm(left_ear_noisy)))
        > 5.0
        or abs(
            np.log10(np.linalg.norm(right_ear_clean) / np.linalg.norm(right_ear_noisy))
        )
        > 5.0
    ):
        sii = 0

    # STDFT and filtering
    # Get 1/3 octave band matrix
    [
        octave_band_matrix,
        centre_frequencies,
        frequency_band_edges_indices,
        _freq_low,
        _freq_high,
    ] = thirdoct(
        sample_rate,
        fft_size_in_samples,
        n_third_octave_bands,
        centre_freq_first_third_octave_hz,
    )  # (fs, nfft, num_bands, min_freq)
    centre_frequencies = (
        2 * math.pi * centre_frequencies
    )  # This is now the angular frequency in radians per sec

    # Apply short time DFT to signals and transpose
    left_ear_clean_hat = stft(left_ear_clean, n_frame, fft_size_in_samples).transpose()
    right_ear_clean_hat = stft(
        right_ear_clean, n_frame, fft_size_in_samples
    ).transpose()
    left_ear_noisy_hat = stft(left_ear_noisy, n_frame, fft_size_in_samples).transpose()
    right_ear_noisy_hat = stft(
        right_ear_noisy, n_frame, fft_size_in_samples
    ).transpose()

    # Take single sided spectrum of signals
    idx = int(fft_size_in_samples / 2 + 1)
    left_ear_clean_hat = left_ear_clean_hat[0:idx, :]
    right_ear_clean_hat = right_ear_clean_hat[0:idx, :]
    left_ear_noisy_hat = left_ear_noisy_hat[0:idx, :]
    right_ear_noisy_hat = right_ear_noisy_hat[0:idx, :]

    # Compute intermediate correlation via EC search
    logging.info("Starting EC evaluation")
    # Here intermeduiate correlation coefficients are evaluated for a discrete set of
    # gamma and tau values (a "grid") and the highest value is chosen.
    intermediate_intelligibility_measure_grid = np.zeros(
        (n_third_octave_bands, np.shape(left_ear_clean_hat)[1] - n_frames + 1)
    )
    p_ec_max = np.zeros(
        (n_third_octave_bands, np.shape(left_ear_clean_hat)[1] - n_frames + 1)
    )

    # Interaural compensation time and level values
    taus = np.linspace(tau_min, tau_max, n_taus)
    gammas = np.linspace(gamma_min, gamma_max, n_gammas)

    # Jitter incorporated below - Equations 5 and 6 in Andersen et al. 2018
    sigma_epsilon = (
        np.sqrt(2)
        * sigma_epsilon_0
        * (1 + (abs(gammas) / alpha_0_db) ** level_shift_deviation)
        / 20
    )
    gammas = gammas / 20

    sigma_delta = np.sqrt(2) * sigma_delta_0 * (1 + (abs(taus) / tau_0))

    logging.info("Processing Equalisation Cancellation stage")
    updated_intermediate_intelligibility_measure, p_ec_max = equalisation_cancellation(
        left_ear_clean_hat,
        right_ear_clean_hat,
        left_ear_noisy_hat,
        right_ear_noisy_hat,
        n_third_octave_bands,
        n_frames,
        frequency_band_edges_indices,
        centre_frequencies.flatten(),
        taus,
        n_taus,
        gammas,
        n_gammas,
        intermediate_intelligibility_measure_grid,
        p_ec_max,
        sigma_epsilon,
        sigma_delta,
    )

    # Compute the better ear STOI
    logging.info("Computing better ear intermediate correlation coefficients")
    # Arrays for the 1/3 octave envelope
    left_ear_clean_third_octave_band = np.zeros(
        (n_third_octave_bands, np.shape(left_ear_clean_hat)[1])
    )
    right_ear_clean_third_octave_band = np.zeros(
        (n_third_octave_bands, np.shape(left_ear_clean_hat)[1])
    )
    left_ear_noisy_third_octave_band = np.zeros(
        (n_third_octave_bands, np.shape(left_ear_clean_hat)[1])
    )
    right_ear_noisy_third_octave_band = np.zeros(
        (n_third_octave_bands, np.shape(left_ear_clean_hat)[1])
    )

    # Apply 1/3 octave bands as described in Eq.(1) of the STOI article
    for k in range(np.shape(left_ear_clean_hat)[1]):
        left_ear_clean_third_octave_band[:, k] = np.dot(
            octave_band_matrix, abs(left_ear_clean_hat[:, k]) ** 2
        )
        right_ear_clean_third_octave_band[:, k] = np.dot(
            octave_band_matrix, abs(right_ear_clean_hat[:, k]) ** 2
        )
        left_ear_noisy_third_octave_band[:, k] = np.dot(
            octave_band_matrix, abs(left_ear_noisy_hat[:, k]) ** 2
        )
        right_ear_noisy_third_octave_band[:, k] = np.dot(
            octave_band_matrix, abs(right_ear_noisy_hat[:, k]) ** 2
        )

    # Arrays for better-ear correlations
    dl_interm = np.zeros(
        (n_third_octave_bands, len(range(n_frames, len(left_ear_clean_hat[1]) + 1)))
    )
    dr_interm = np.zeros(
        (n_third_octave_bands, len(range(n_frames, len(left_ear_clean_hat[1]) + 1)))
    )
    left_improved = np.zeros(
        (n_third_octave_bands, len(range(n_frames, len(left_ear_clean_hat[1]) + 1)))
    )
    right_improved = np.zeros(
        (n_third_octave_bands, len(range(n_frames, len(left_ear_clean_hat[1]) + 1)))
    )

    # Compute temporary better-ear correlations
    for m in range(
        n_frames, np.shape(left_ear_clean_hat)[1]
    ):  # pylint: disable=invalid-name
        left_ear_clean_seg = left_ear_clean_third_octave_band[:, (m - n_frames) : m]
        right_ear_clean_seg = right_ear_clean_third_octave_band[:, (m - n_frames) : m]
        left_ear_noisy_seg = left_ear_noisy_third_octave_band[:, (m - n_frames) : m]
        right_ear_noisy_seg = right_ear_noisy_third_octave_band[:, (m - n_frames) : m]

        for n in range(n_third_octave_bands):  # pylint: disable=invalid-name
            left_ear_clean_n = (
                left_ear_clean_seg[n, :] - np.sum(left_ear_clean_seg[n, :]) / n_frames
            )
            right_ear_clean_n = (
                right_ear_clean_seg[n, :] - np.sum(right_ear_clean_seg[n, :]) / n_frames
            )
            left_ear_noisy_n = (
                left_ear_noisy_seg[n, :] - np.sum(left_ear_noisy_seg[n, :]) / n_frames
            )
            right_ear_noisy_n = (
                right_ear_noisy_seg[n, :] - np.sum(right_ear_noisy_seg[n, :]) / n_frames
            )
            left_improved[n, m - n_frames] = np.sum(
                left_ear_clean_n * left_ear_clean_n
            ) / np.sum(left_ear_noisy_n * left_ear_noisy_n)
            right_improved[n, m - n_frames] = np.sum(
                right_ear_clean_n * right_ear_clean_n
            ) / np.sum(right_ear_noisy_n * right_ear_noisy_n)
            dl_interm[n, m - n_frames] = np.sum(left_ear_clean_n * left_ear_noisy_n) / (
                np.linalg.norm(left_ear_clean_n) * np.linalg.norm(left_ear_noisy_n)
            )
            dr_interm[n, m - n_frames] = np.sum(
                right_ear_clean_n * right_ear_noisy_n
            ) / (np.linalg.norm(right_ear_clean_n) * np.linalg.norm(right_ear_noisy_n))

    # Get the better ear intermediate coefficients
    idx = np.isfinite(dl_interm)
    dl_interm[~idx] = 0
    idx = np.isfinite(dr_interm)
    dr_interm[~idx] = 0
    p_be_max = np.maximum(left_improved, right_improved)
    dbe_interm = np.zeros((np.shape(dl_interm)))

    idx = left_improved > right_improved
    dbe_interm[idx] = dl_interm[idx]
    dbe_interm[~idx] = dr_interm[~idx]

    # Compute STOI measure
    # Whenever a single ear provides a higher correlation than the corresponding EC
    # processed alternative,the better-ear correlation is used.
    idx = p_be_max > p_ec_max
    updated_intermediate_intelligibility_measure[idx] = dbe_interm[idx]
    sii = np.mean(updated_intermediate_intelligibility_measure)

    logging.info("MBSTOI processing complete")

    return sii
