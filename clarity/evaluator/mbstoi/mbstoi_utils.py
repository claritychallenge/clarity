"""Utilities for MBSTOI processing."""
import logging

import numpy as np
from scipy.signal import find_peaks

EPS = np.finfo("float").eps


def equalisation_cancellation(
    left_ear_clean_hat: np.ndarray,
    right_ear_clean_hat: np.ndarray,
    left_ear_noisy_hat: np.ndarray,
    right_ear_noisy_hat: np.ndarray,
    n_third_octave_bands: int,
    n_frames: int,
    frequency_band_edges_indices: np.ndarray,
    centre_frequencies: np.ndarray,
    taus: np.ndarray,
    n_taus: int,
    gammas: np.ndarray,
    n_gammas: int,
    intermediate_intelligibility_measure_grid: np.ndarray,
    p_ec_max: np.ndarray,
    sigma_epsilon: np.ndarray,
    sigma_delta: np.ndarray,
):
    """Run the equalisation-cancellation (EC) stage of the MBSTOI metric.

    The EC loop evaluates one huge equation in every iteration (see referenced notes for details). The left and right
    ear signals are level adjusted by gamma (in dB) and time shifted by tau relative to one-another and are thereafter
    subtracted. The processed signals are treated similarly. To obtain performance similar to that of humans,the EC
    stage adds jitter. We are searching for the level and time adjustments that maximise the intermediate correlation
    coefficients d. Could add location of source and interferer to this to reduce search space.

    Args:
        left_ear_clean_hat (np.ndarray) : Clean left ear short-time DFT coefficients (single-sided) per frequency
    bin and frame.
        right_ear_clean_hat (np.ndarray) : Clean right ear short-time DFT coefficients (single-sided) per frequency bin and frame.
        left_ear_noisy_hat (np.ndarray) : Noisy/processed left ear short-time DFT coefficients (single-sided) per
    frequency bin and frame.
        right_ear_noisy_hat (np.ndarray) : Noisy/processed right eat short-time DFT coefficients (single-sided) per
    frequency bin and frame.
    n_third_octave_bands (int) : Number of one-third octave bands.
    n_frames (int) : Number of frames for intermediate intelligibility measure.
    fids (np.ndarray) : Indices of frequency band edges.
    cf (np.ndarray) : Centre frequencies.
    taus (np.ndarray) : Interaural delay (tau) values.
    n_taus (int) : Number of tau values.
    gammas (np.ndarray) : Interaural level difference (gamma) values.
    ngammas (int) : Number of gamma values.
    intermediate_intelligibility_measure_grid (np.ndarray) : Grid for intermediate intelligibility measure.
    p_ec_max (np.ndarray) : Empty grid for maximum values.
    sigma_epsilon (np.ndarray) : Jitter for gammas.
    sigma_delta (np.ndarray) : Jitter for taus.

    Returns:
        intermediate_intelligibility_measure_grid (np.ndarray) : updated grid for intermediate intelligibility measure
        p_ec_max (np.ndarray) : grid containing maximum values.
    """
    taus = np.expand_dims(taus, axis=0)
    sigma_delta = np.expand_dims(sigma_delta, axis=0)
    sigma_epsilon = np.expand_dims(sigma_epsilon, axis=0)
    gammas = np.expand_dims(gammas, axis=0)
    epsexp = np.exp(2 * np.log(10) ** 2 * sigma_epsilon**2)

    # per frequency band
    for i in range(n_third_octave_bands):  # pylint: disable=invalid-name
        tauexp = np.exp(-1j * centre_frequencies[i] * taus)
        tauexp2 = np.exp(-1j * 2 * centre_frequencies[i] * taus)
        deltexp = np.exp(-2 * centre_frequencies[i] ** 2 * sigma_delta**2)
        epsdelexp = np.exp(
            0.5
            * (
                np.ones((n_taus, 1))
                * (
                    np.log(10) ** 2 * sigma_epsilon**2
                    - centre_frequencies[i] ** 2 * np.transpose(sigma_delta) ** 2
                )
                * np.ones((1, n_gammas))
            )
        )
        # per frame
        for jj in range(
            np.shape(intermediate_intelligibility_measure_grid)[1]
        ):  # pylint: disable=invalid-name
            seg_xl = left_ear_clean_hat[
                int(frequency_band_edges_indices[i, 0] - 1) : int(
                    frequency_band_edges_indices[i, 1]
                ),
                jj : (jj + n_frames),
            ]
            seg_xr = right_ear_clean_hat[
                int(frequency_band_edges_indices[i, 0] - 1) : int(
                    frequency_band_edges_indices[i, 1]
                ),
                jj : (jj + n_frames),
            ]
            seg_yl = left_ear_noisy_hat[
                int(frequency_band_edges_indices[i, 0] - 1) : int(
                    frequency_band_edges_indices[i, 1]
                ),
                jj : (jj + n_frames),
            ]
            seg_yr = right_ear_noisy_hat[
                int(frequency_band_edges_indices[i, 0] - 1) : int(
                    frequency_band_edges_indices[i, 1]
                ),
                jj : (jj + n_frames),
            ]

            # All normalised by subtracting mean
            left_ear_clean = np.sum(np.conj(seg_xl) * seg_xl, axis=0)
            left_ear_clean = np.expand_dims(left_ear_clean, axis=0)
            left_ear_clean = left_ear_clean - np.mean(left_ear_clean)
            right_ear_clean = np.sum(np.conj(seg_xr) * seg_xr, axis=0)
            right_ear_clean = np.expand_dims(right_ear_clean, axis=0)
            right_ear_clean = right_ear_clean - np.mean(right_ear_clean)
            rhox = np.sum(np.conj(seg_xr) * seg_xl, axis=0)
            rhox = np.expand_dims(rhox, axis=0)
            rhox = rhox - np.mean(rhox)
            left_ear_noisy = np.sum(np.conj(seg_yl) * seg_yl, axis=0)
            left_ear_noisy = np.expand_dims(left_ear_noisy, axis=0)
            left_ear_noisy = left_ear_noisy - np.mean(left_ear_noisy)
            right_ear_noisy = np.sum(np.conj(seg_yr) * seg_yr, axis=0)
            right_ear_noisy = np.expand_dims(right_ear_noisy, axis=0)
            right_ear_noisy = right_ear_noisy - np.mean(right_ear_noisy)
            rhoy = np.sum(np.conj(seg_yr) * seg_yl, axis=0)
            rhoy = np.expand_dims(rhoy, axis=0)
            rhoy = rhoy - np.mean(rhoy)

            # Evaluate parts of intermediate correlation - EC stage exhaustive search
            # over ITD/ILD values. These correspond to equations 7 and 8 in
            # Andersen et al. 2018

            # Calculate Exy
            firstpart = firstpartfunc(
                left_ear_clean,
                left_ear_noisy,
                right_ear_clean,
                right_ear_noisy,
                n_taus,
                gammas,
                epsexp,
            )
            secondpart = secondpartfunc(
                left_ear_clean, left_ear_noisy, rhoy, rhox, tauexp, epsdelexp, gammas
            )
            thirdpart = thirdpartfunc(
                right_ear_clean, right_ear_noisy, rhoy, rhox, tauexp, epsdelexp, gammas
            )
            fourthpart = fourthpartfunc(rhox, rhoy, tauexp2, n_gammas, deltexp)
            exy = np.real(firstpart - secondpart - thirdpart + fourthpart)

            # Calculate Exx
            firstpart = firstpartfunc(
                left_ear_clean,
                left_ear_clean,
                right_ear_clean,
                right_ear_clean,
                n_taus,
                gammas,
                epsexp,
            )
            secondpart = secondpartfunc(
                left_ear_clean, left_ear_clean, rhox, rhox, tauexp, epsdelexp, gammas
            )
            thirdpart = thirdpartfunc(
                right_ear_clean, right_ear_clean, rhox, rhox, tauexp, epsdelexp, gammas
            )
            fourthpart = fourthpartfunc(rhox, rhox, tauexp2, n_gammas, deltexp)
            exx = np.real(firstpart - secondpart - thirdpart + fourthpart)

            # Calculate Eyy
            firstpart = firstpartfunc(
                left_ear_noisy,
                left_ear_noisy,
                right_ear_noisy,
                right_ear_noisy,
                n_taus,
                gammas,
                epsexp,
            )
            secondpart = secondpartfunc(
                left_ear_noisy, left_ear_noisy, rhoy, rhoy, tauexp, epsdelexp, gammas
            )
            thirdpart = thirdpartfunc(
                right_ear_noisy, right_ear_noisy, rhoy, rhoy, tauexp, epsdelexp, gammas
            )
            fourthpart = fourthpartfunc(rhoy, rhoy, tauexp2, n_gammas, deltexp)
            eyy = np.real(firstpart - secondpart - thirdpart + fourthpart)

            # Ensure that intermediate correlation will be sensible and compute it
            # If all minimum values are less than 1e-40, set d[i,jj] to -1
            if np.min(abs(exx * eyy), axis=0).all() < 1e-40:
                intermediate_intelligibility_measure_grid[i, jj] = -1
                continue

            proportion = np.divide(exx, eyy)
            tmp = proportion.max(axis=0)
            idx1 = proportion.argmax(axis=0)

            # Return overall maximum and index
            p_ec_max[i, jj] = tmp.max()
            idx2 = tmp.argmax()
            intermediate_intelligibility_measure_grid[i, jj] = np.divide(
                exy[idx1[idx2], idx2],
                np.sqrt(exx[idx1[idx2], idx2] * eyy[idx1[idx2], idx2]),
            )

    return (intermediate_intelligibility_measure_grid, p_ec_max)


# pylint: disable=invalid-name,fixme
def firstpartfunc(
    L1: np.ndarray,
    L2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    n_taus: int,
    gammas: np.ndarray,
    epsexp,
):
    # FixMe : Complete Docstring
    """Need a description

    Args:
        L1 (???) : ???
        L2 (???) : ???
        R1 (???) : ???
        R2 (???) : ???
        n_taus (???) : ???
        gammas (???) : ???

    Returns:
    """
    result = (
        np.ones((n_taus, 1))
        * (
            (
                10 ** (2 * gammas) * np.sum(L1 * L2)
                + 10 ** (-2 * gammas) * np.sum(R1 * R2)
            )
            * epsexp
        )
        + np.sum(L1 * R2)
        + np.sum(R1 * L2)
    )
    return result


def secondpartfunc(
    L1: np.ndarray, L2: np.ndarray, rho1, rho2, tauexp, epsdelexp, gammas: np.ndarray
):
    # FixMe : Complete Docstring
    """Need a description

    Args:
        L1 (???) : ???
        L2 (???) : ???
        rho1 (???) : ???
        rho2 (???) : ???
        tauexp (???) : ???
        epsdelexp (???) : ???
        gammas (???) : ???

    Returns:
    """
    result = (
        2
        * (
            np.transpose(
                np.dot(L1, np.real(np.transpose(rho1) * tauexp))
                + np.dot(L2, np.real(np.transpose(rho2) * tauexp))
            )
            * 10**gammas
        )
        * epsdelexp
    )
    return result


def thirdpartfunc(R1, R2, rho1, rho2, tauexp, epsdelexp, gammas: np.ndarray):
    # FixMe : Complete docstring
    """Need a description

    Args:
        R1 (???) : ???
        R2 (???) : ???
        rho1 (???) : ???
        rho2 (???) : ???
        tauexp (???) : ???
        epsdelexp (???) : ???
        gammas (???) : ???

    Returns:
    """
    result = (
        2
        * np.transpose(
            np.dot(R1, np.real(np.dot(np.transpose(rho1), tauexp)))
            + np.dot(R2, np.real(np.transpose(rho2) * tauexp))
        )
        * 10**-gammas
        * epsdelexp
    )
    return result


def fourthpartfunc(rho1, rho2, tauexp2, n_gammas, deltexp):
    # FixMe : Complete docstring
    """Need a description

    Args:
    rho1 (???) : ???
    rho2 (???) : ???
    tauexp2 (???) : ???
    n_gammas (???) : ???
    deltexp (???) : ???

    Returns
    """
    result = (
        2
        * np.transpose(
            np.real(np.dot(rho1, np.conj(np.transpose(rho2))))
            + deltexp * np.real(np.dot(rho1, np.transpose(rho2) * tauexp2))
        )
        * np.ones((1, n_gammas))
    )

    return result


# pylint: enable=invalid-name,fixme


def stft(signal: np.ndarray, win_size: int, fft_size: int) -> np.ndarray:
    """Short-time Fourier transform based on MBSTOI Matlab code.

    Args:
        signal (np.ndarray) : Input signal
        win_size (int) : The size of the window and the signal frames.
        fft_size (int) : The size of the fft in samples (zero-padding or not).

    Returns:
        stft_out (np.ndarray) : The short-time Fourier transform of signal.
    """
    hop = int(win_size / 2)
    frames = list(range(0, len(signal) - win_size, hop))
    stft_out = np.zeros((len(frames), fft_size), dtype=np.complex128)

    hanning_window = np.hanning(win_size + 2)[1:-1]
    signal = signal.flatten()

    # pylint: disable=invalid-name
    for i, frame in enumerate(frames):
        ii = list(range(frame, (frame + win_size), 1))
        stft_out[i, :] = np.fft.fft(signal[ii] * hanning_window, n=fft_size, axis=0)
    # pylint: enable=invalid-name

    return stft_out


def remove_silent_frames(
    left_ear_clean: np.ndarray,
    right_ear_clean: np.ndarray,
    left_ear_noisy: np.ndarray,
    right_ear_noisy: np.ndarray,
    dynamic_range: np.ndarray = 40,
    frame_length: int = 256,
    hop: float = 128,
):
    """Remove silent frames of x and y based on x

    A frame is excluded if its energy is lower than max(energy) - dyn_range
    The frame exclusion is based solely on x, the clean speech signal
    Based on mpariente/pystoi/utils.py

    Args:
        left_ear_clean (np.ndarray) : Clean input signal left channel.
        right_ear_clean (np.ndarray) : Clean input signal right channel.
        left_ear_noisy (np.ndarray) : Degraded/processed signal left channel.
        right_ear_noisy (np.ndarray) : Degraded/processed signal right channel.
        dyn_range (np.ndarray) : Range, energy range to determine which frame is silent (default : 40).
        framelen (int) : Window size for energy evaluation (default : 256).
        hop (int) : Hop size for energy evaluation (default : 128).

    Returns :
        xl_sil (np.ndarray): left_ear_clean without the silent frames.
        xr_sil (np.ndarray): right_ear_clean without the silent frames.
        yl_sil (np.ndarray): left_ear_noisy without the silent frames in xl_sil.
        yr_sil (np.ndarray): right_ear_noisy without the silent frames in rl_sil.
    """
    dyn_range = int(dynamic_range)
    hop = int(hop)

    # Compute Mask
    hanning_window = np.hanning(frame_length + 2)[1:-1]

    xl_frames = np.array(
        [
            hanning_window * left_ear_clean[i : i + frame_length]
            for i in range(0, len(left_ear_clean) - frame_length, hop)
        ]
    )
    xr_frames = np.array(
        [
            hanning_window * right_ear_clean[i : i + frame_length]
            for i in range(0, len(right_ear_clean) - frame_length, hop)
        ]
    )
    yl_frames = np.array(
        [
            hanning_window * left_ear_noisy[i : i + frame_length]
            for i in range(0, len(left_ear_noisy) - frame_length, hop)
        ]
    )
    yr_frames = np.array(
        [
            hanning_window * right_ear_noisy[i : i + frame_length]
            for i in range(0, len(right_ear_noisy) - frame_length, hop)
        ]
    )

    # Compute energies in dB
    xl_energies = 20 * np.log10(np.linalg.norm(xl_frames, axis=1) + EPS)
    xr_energies = 20 * np.log10(np.linalg.norm(xr_frames, axis=1) + EPS)

    # Find boolean mask of energies lower than dynamic_range dB
    # with respect to maximum clean speech energy frame
    maskxl = (np.max(xl_energies) - dyn_range - xl_energies) < 0
    maskxr = (np.max(xr_energies) - dyn_range - xr_energies) < 0

    mask = maskxl | maskxr

    # Remove silent frames by masking
    xl_frames = xl_frames[mask]
    xr_frames = xr_frames[mask]
    yl_frames = yl_frames[mask]
    yr_frames = yr_frames[mask]

    # init zero arrays to hold x, y with silent frames removed
    n_sil = (len(xl_frames) - 1) * hop + frame_length
    xl_sil = np.zeros(n_sil)
    xr_sil = np.zeros(n_sil)
    yl_sil = np.zeros(n_sil)
    yr_sil = np.zeros(n_sil)

    for i in range(xl_frames.shape[0]):
        xl_sil[range(i * hop, i * hop + frame_length)] += xl_frames[i, :]
        xr_sil[range(i * hop, i * hop + frame_length)] += xr_frames[i, :]
        yl_sil[range(i * hop, i * hop + frame_length)] += yl_frames[i, :]
        yr_sil[range(i * hop, i * hop + frame_length)] += yr_frames[i, :]

    return xl_sil, xr_sil, yl_sil, yr_sil


def thirdoct(frequency_sampling: int, nfft: int, num_bands: int, min_freq: int):
    """Returns the 1/3 octave band matrix and its center frequencies based on mpariente/pystoi.

    Args:
        fs (int) : Frequency sampling rate.
        n_fft (int) : Number of FFT. FFT == ???
        num_bands (int) : Number of one-third octave bands.
        min_freq (int) : Center frequencey of the lowest one-third octave band.

    Returns:
        octave_band_matrix (np.ndarray) :
        centre_frequencies (np.ndarray) : Centre frequencies.
        frequency_band_edges_indices (np.ndarray) : Indices of Frequency Band Edges
        freq_low (float) : Lowest frequency.
        freq_high (float) : Highest frequency
    """
    # pylint: disable=invalid-name
    f = np.linspace(0, frequency_sampling, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = np.array(range(num_bands)).astype(float)
    # pylint: enable=invalid-name
    centre_frequencies = np.power(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * np.power(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2.0, (2 * k + 1) / 6)
    octave_band_matrix = np.zeros((num_bands, len(f)))  # a verifier
    frequency_band_edges_indices = np.zeros((num_bands, 2))

    for i in range(len(centre_frequencies)):  # pylint: disable=invalid-name
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        octave_band_matrix[i, fl_ii:fh_ii] = 1
        frequency_band_edges_indices[i, :] = [fl_ii + 1, fh_ii]

    centre_frequencies = centre_frequencies[np.newaxis, :]

    return (
        octave_band_matrix,
        centre_frequencies,
        frequency_band_edges_indices,
        freq_low,
        freq_high,
    )


def find_delay_impulse(ddf: np.ndarray, initial_value: int = 22050):
    """Find binaural delay in signal ddf given initial location of unit impulse, initial_value.

    Args:
        ddf (np.ndarray) :
        initial_value: (int) : Initial value (default: 22050)

    Returns:

    """
    pk0 = find_peaks(ddf[:, 0])
    pk1 = find_peaks(ddf[:, 1])
    delay = np.zeros((2, 1))
    if len(pk0[0]) > 0:
        # m = np.max(ddf[pk0[0], 0])
        pkmax0 = np.argmax(ddf[:, 0])
        delay[0] = int(pkmax0 - initial_value)
    else:
        logging.error("Error in selecting peaks.")
    if len(pk1[0]) > 0:
        pkmax1 = np.argmax(ddf[:, 1])
        delay[1] = int(pkmax1 - initial_value)
    else:
        logging.error("Error in selecting peaks.")
    return delay
