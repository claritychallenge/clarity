"""HASPI EBM module"""
from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import convolve, convolve2d

if TYPE_CHECKING:
    from numpy import ndarray


def env_filter(
    reference_db: ndarray,
    processed_db: ndarray,
    filter_cutoff: float,
    freq_sub_sample: float,
    freq_samp: float,
) -> tuple[ndarray, ndarray]:
    """
    Lowpass filter and subsample the envelope in dB SL produced by the model
    of the auditory periphery. The LP filter uses a von Hann raised cosine
    window to ensure that there are no negative envelope values produced by
    the filtering operation.

    Args:
        reference_db (np.ndarray): env in dB SL for the ref signal in each auditory band
        processed_db (np.ndarray): env in dB SL for the degraded signal in each auditory
            band
        filter_cutoff ():  LP filter cutoff frequency for the filtered envelope, Hz
        freq_sub_samp ():  subsampling frequency in Hz for the LP filtered envelopes
        freq_samp ():  sampling rate in Hz for the signals xdB and ydB

    Returns:
        tuple: reference_env - LP filtered and subsampled reference signal envelope
           Each frequency band is a separate column.
           processed_env - LP filtered and subsampled degraded signal envelope


    Updates:
        James M. Kates, 12 September 2019.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Check the filter design parameters
    if freq_sub_sample > freq_samp:
        raise ValueError("upsampling rate too high.")

    if filter_cutoff > 0.5 * freq_sub_sample:
        raise ValueError("LP cutoff frequency too high.")

    # Check the data matrix orientation
    # Require each frequency band to be a separate column
    nrow = reference_db.shape[0]  # number of rows
    ncol = reference_db.shape[1]  # number of columnts
    if ncol > nrow:
        reference_db = reference_db.T
        processed_db = processed_db.T
    nsamp = reference_db.shape[0]

    # Compute the lowpass filter length in samples to give -3 dB at fcut Hz
    tfilt = 1000 * (1 / filter_cutoff)  # filter length in ms
    tfilt = 0.7 * tfilt  # Empirical adjustment to the filter length
    nfilt = round(0.001 * tfilt * freq_samp)  # Filter length in samples
    nhalf = floor(nfilt / 2)
    nfilt = int(2 * nhalf)  # Force an even filter length

    # Design the FIR LP filter using a von Hann window to ensure that there are no
    # negative envelope values. The MATLAB code uses the hanning() function, which
    # returns the Hann window without the first and last zero-weighted window samples,
    # unlike np.hann and scipy.signal.windows.hann; the code below replicates this
    # behaviour
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, nfilt / 2 + 1) / (nfilt + 1)))
    benv = np.concatenate((window, np.flip(window)))
    benv = benv / np.sum(benv)

    # LP filter for the envelopes at fsamp
    reference_env = convolve2d(reference_db, np.expand_dims(benv, 1), "full")
    reference_env = reference_env[nhalf : nhalf + nsamp, :]
    processed_env = convolve2d(processed_db, np.expand_dims(benv, 1), "full")
    processed_env = processed_env[nhalf : nhalf + nsamp, :]

    # Subsample the LP filtered envelopes
    space = floor(freq_samp / freq_sub_sample)
    index = np.arange(0, nsamp, space)

    return reference_env[index, :], processed_env[index, :]


def cepstral_correlation_coef(
    reference_db: ndarray,
    processed_db: ndarray,
    thresh_cep: float,
    thresh_nerve: float,
    nbasis: int,
) -> tuple[ndarray, ndarray]:
    """
    Compute the cepstral correlation coefficients between the reference signal
    and the distorted signal log envelopes. The silence portions of the
    signals are removed prior to the calculation based on the envelope of the
    reference signal. For each time sample, the log spectrum in dB SL is
    fitted with a set of half-cosine basis functions. The cepstral coefficients
    then form the input to the cepstral correlation
    calculation.

    Args:
        reference_db (): subsampled reference signal envelope in dB SL in each band
        processed_db (): subsampled distorted output signal envelope
        thresh_cep (): threshold in dB SPL to include sample in calculation
        thresh_nerve (): additive noise RMS for IHC firing (in dB)
        nbasis: number of cepstral basis functions to use

    Returns:
        tuple: refernce_cep cepstral coefficient matrix for the ref signal
            (nsamp,nbasis) processed_cep cepstral coefficient matrix for the output
            signal (nsamp,nbasis) each column is a separate basis function, from low to
            high

    Updates:
        James M. Kates, 23 April 2015.
        Gammawarp version to fit the basis functions, 11 February 2019.
        Additive noise for IHC firing rates, 24 April 2019.
            Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Processing parameters
    nbands = reference_db.shape[1]

    # Mel cepstrum basis functions
    freq = np.arange(0, nbasis)
    k = np.arange(0, nbands)
    cepm = np.zeros((nbands, nbasis))

    for n in range(nbasis):
        basis = np.cos(freq[n] * np.pi * k / (nbands - 1))
        cepm[:, n] = basis / np.sqrt(np.sum(basis**2))

    # Find the reference segments that lie sufficiently above the quiescent rate
    x_linear = 10 ** (
        reference_db / 20
    )  # Convert envelope dB to linear (specific loudness)
    xsum = np.sum(x_linear, 1) / nbands  # Proportional to loudness in sones
    xsum = 20 * np.log10(xsum)  # Convert back to dB (loudness in phons)
    index = np.where(xsum > thresh_cep)[0]  # Identify those segments above threshold
    nsamp = len(index)  # Number of segments above threshold

    # Exit if not enough segments above zero
    if nsamp <= 1:
        raise ValueError("Signal below threshold")

    # Remove the silent samples
    reference_db = reference_db[index, :]
    processed_db = processed_db[index, :]

    # Add low-level noise to provide IHC firing jitter
    reference_db = add_noise(reference_db, thresh_nerve)
    processed_db = add_noise(processed_db, thresh_nerve)

    # Compute the mel cepstrum coefficients using only those samples above threshold
    reference_cep = reference_db @ cepm
    processed_cep = processed_db @ cepm

    # Remove the average value from the cepstral coefficients. The cepstral
    # cross-correlation will thus be a cross-covariance, and there is no effect of the
    # absolute signal level in dB.
    for n in range(nbasis):
        x = reference_cep[:, n]
        x = x - np.mean(x)
        reference_cep[:, n] = x
        y = processed_cep[:, n]
        y = y - np.mean(y)
        processed_cep[:, n] = y

    return reference_cep, processed_cep


def add_noise(reference_db: ndarray, thresh_db: float) -> ndarray:
    """
    Add independent random Gaussian noise to the subsampled signal envelope
    in each auditory frequency band.

    Args:
        reference_db (): subsampled envelope in dB re:auditory threshold
        thresh_db (): additive noise RMS level (in dB)

    Returns:
      () envelope with threshold noise added, in dB re:auditory threshold

    Updates:
        James M. Kates, 23 April 2019.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Additive noise sequence
    # Gaussian noise with RMS=1, then scaled
    noise = thresh_db * np.random.standard_normal(reference_db.shape)

    # Add the noise to the signal envelope
    return reference_db + noise


def fir_modulation_filter(
    reference_envelope: ndarray,
    processed_envelope: ndarray,
    freq_sub_sampling: float,
    center_frequencies: ndarray | None = None,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Apply a FIR modulation filterbank to the reference envelope signals
    contained in matrix reference_envelope and the processed signal envelope
    signals in matrix processed_envelope. Each column in reference_envelope
    and processed_envelope is a separate filter band or cepstral coefficient
    basis function. The modulation filters use a lowpass filter for the
    lowest modulation rate, and complex demodulation followed by a lowpass
    filter for the remaining bands. The onset and offset transients are
    removed from the FIR convolutions to temporally align the modulation
    filter outputs.

    Args:
        reference_envelope (np.ndarray) : matrix containing the subsampled reference
            envelope values. Each column is a different frequency band or cepstral basis
            function arranged from low to high.
        processed_envelope (np.ndarray): matrix containing the subsampled processed
            envelope values
        freq_sub_sampling (): envelope sub-sampling rate in Hz
        center_frequencies (np.ndarray): Center Frequencies

    Returns:
        tuple:
            reference_modulation ():  a cell array containing the reference signal
                output of the modulation filterbank. reference_modulation is of size
                [nchan,nmodfilt] where nchan is the number of frequency channels or
                cepstral basis functions in reference_envelope, and nmodfilt is the
                number of modulation filters used in the analysis. Each cell contains a
                column vector of length nsamp, where nsamp is the number of samples in
                each envelope sequence contained in the columns of reference_envelope.
            processed_modulation (): cell array containing the processed signal output
                of the modulation filterbank.
            center_frequencies (): vector of modulation rate filter center frequencies

    Updates:
        James M. Kates, 14 February 2019.
        Two matrix version of gwarp_ModFiltWindow, 19 February 2019.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Input signal properties
    nsamp = reference_envelope.shape[0]
    nchan = reference_envelope.shape[1]

    # Modulation filter band cf and edges, 10 bands
    # Band spacing uses the factor of 1.6 used by Dau
    if center_frequencies is None:
        center_frequencies = np.array(
            [2, 6, 10, 16, 25, 40, 64, 100, 160, 256]
        )  # Band center frequencies
    nmod = len(center_frequencies)
    edge = np.zeros(nmod + 1)
    edge[0:3] = [0, 4, 8]
    for k in range(3, nmod + 1):
        edge[k] = (center_frequencies[k - 1] ** 2) / edge[k - 1]

    # Allowable filters based on envelope subsampling rate
    fn_yq = 0.5 * freq_sub_sampling
    index = edge < fn_yq
    edge = edge[index]  # Filter upper bands edges less than Nyquist rate
    nmod = len(edge) - 1
    center_frequencies = center_frequencies[:nmod]

    # Assign FIR filter lengths. Setting t0=0.2 gives a filter Q of about 1.25
    # to match Ewert et al. (2002), and t0=0.33 corresponds to Q=2 (Dau et al 1997a).
    # Moritz et al. (2015) used t0=0.29. General relation Q=6.25*t0, compromise with
    # t0=0.24 which gives Q=1.5

    t_0 = 0.24  # Filter length in seconds for the lowest modulation frequency band
    t = np.zeros(nmod)  # pylint: disable=invalid-name
    t[0] = t_0
    t[1] = t_0
    t[2:nmod] = (
        t_0 * center_frequencies[2] / center_frequencies[2:nmod]
    )  # Constant-Q filters above 10 Hz
    nfir = 2 * np.floor(
        t * freq_sub_sampling / 2
    )  # Force even filter lengths in samples
    nhalf = nfir / 2

    # Design the family of lowpass windows
    filter_coefficients = []
    for k in range(nmod):
        coefficient = np.hanning(nfir[k] + 1)
        coefficient /= np.sum(coefficient)
        filter_coefficients.append(coefficient)

    # Pre-compute the cosine and sine arrays
    cosine = []  # cosine array, one frequency per list element
    sine = []  # sine array, one frequency per list element
    n = np.arange(1, nsamp + 1)
    for k in range(nmod):
        if k == 0:
            cosine.append(1)
            sine.append(0)
        else:
            cosine.append(
                np.sqrt(2) * np.cos(np.pi * n * center_frequencies[k] / fn_yq)
            )
            sine.append(np.sqrt(2) * np.sin(np.pi * n * center_frequencies[k] / fn_yq))

    # Convolve the input and output envelopes with the modulation filters
    reference_modulation = np.zeros((nchan, nmod, nsamp))
    processed_modulation = np.zeros((nchan, nmod, nsamp))
    for k in range(nmod):  # Loop over the modulation filters
        coefficient = filter_coefficients[k]
        transient_duration = int(nhalf[k])  # Transient duration for the filter
        _cosine = cosine[k]  # Cosine and sine for complex modulation
        _sine = sine[k]
        for m in range(nchan):  # Loop over the input signal vectors
            # Reference signal
            # Extract the frequency or cepstral coefficient band
            reference_cepstral_coef = reference_envelope[:, m]
            # Complex demodulation, then LP filter
            reference_complex_demodulation = convolve(
                (
                    reference_cepstral_coef * _cosine
                    - 1j * reference_cepstral_coef * _sine
                ),
                coefficient,
            )
            # Truncate the filter transients
            reference_complex_demodulation = reference_complex_demodulation[
                transient_duration : transient_duration + nsamp
            ]
            # Modulate back up to the carrier freq
            xfilt = (
                np.real(reference_complex_demodulation) * _cosine
                - np.imag(reference_complex_demodulation) * _sine
            )
            # Save the filtered signal
            reference_modulation[m, k, :] = xfilt

            # Processed signal
            # Extract the frequency or cepstral coefficient band
            processed_cepstral_coef = processed_envelope[:, m]
            # Complex demodulation, then LP filter
            processed_complex_demodulation = convolve(
                (
                    processed_cepstral_coef * _cosine
                    - 1j * processed_cepstral_coef * _sine
                ),
                coefficient,
            )
            # Truncate the filter transients
            processed_complex_demodulation = processed_complex_demodulation[
                transient_duration : transient_duration + nsamp
            ]
            # Modulate back up to the carrier freq
            yfilt = (
                np.real(processed_complex_demodulation) * _cosine
                - np.imag(processed_complex_demodulation) * _sine
            )
            processed_modulation[m, k, :] = yfilt  # Save the filtered signal

    return reference_modulation, processed_modulation, center_frequencies


def modulation_cross_correlation(
    reference_modulation: ndarray, processed_modulation: ndarray
) -> ndarray:
    """
    Compute the cross-correlations between the input signal time-frequency
    envelope and the distortion time-frequency envelope. The cepstral
    coefficients or envelopes in each frequency band have been passed
    through the modulation filterbank using function ebm_ModFilt.

    Args:
       reference_modulation (np.array): cell array containing the reference signal
           output of the modulation filterbank. Xmod is of size [nchan,nmodfilt] where
           nchan is the number of frequency channels or cepstral basis functions in
           Xenv, and nmodfilt is the number of modulation filters used in the analysis.
           Each cell contains a column vector of length nsamp, where nsamp is the
           number of samples in each envelope sequence contained in the columns of
           Xenv.
       processed_modulation (np.ndarray): subsampled distorted output signal envelope

    Output:
        float: aveCM modulation correlations averaged over basis functions 2-6
             vector of size nmodfilt

    Updates:
       James M. Kates, 21 February 2019.
       Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Processing parameters
    nchan = reference_modulation.shape[0]  # Number of basis functions
    nmod = reference_modulation.shape[1]  # Number of modulation filters
    small = 1e-30  # Zero threshold

    # Compute the cross-covariance matrix
    covariance_matrix = np.zeros((nchan, nmod))
    for m in range(nmod):
        for j in range(nchan):
            # Index j gives the input reference band
            x_j = reference_modulation[j, m]  # Input freq band j, modulation freq m
            x_j -= np.mean(x_j)
            xsum = np.sum(x_j**2)
            # Processed signal band
            y_j = processed_modulation[j, m]  # Processed freq band j, modulation freq m
            y_j -= np.mean(y_j)
            ysum = np.sum(y_j**2)
            # Cross-correlate the reference and processed signals
            if (xsum < small) or (ysum < small):
                covariance_matrix[j, m] = 0
            else:
                covariance_matrix[j, m] = np.abs(np.sum(x_j * y_j)) / np.sqrt(
                    xsum * ysum
                )

    return np.mean(covariance_matrix[1:6], 0)
