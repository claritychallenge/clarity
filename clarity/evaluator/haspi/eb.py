"""Module for HASPI, HASQI, HAAQI EBs"""
from __future__ import annotations

# pylint: disable=import-error
import logging
from typing import TYPE_CHECKING

import numpy as np
from numba import njit  # type: ignore # <-- silence mypy no attribute error
from scipy.signal import (
    butter,
    cheby2,
    convolve,
    correlate,
    firwin,
    group_delay,
    lfilter,
    resample_poly,
)

from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


def ear_model(
    reference: ndarray,
    reference_freq: float,
    processed: ndarray,
    processed_freq: float,
    hearing_loss: ndarray,
    itype: int,
    level1: float,
    nchan: int = 32,
    m_delay: int = 1,
    shift: float | None = None,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, float]:
    """
    Function that implements a cochlear model that includes the middle ear,
    auditory filter bank, Outer Hair Cell (OHC) dynamic-range compression,
    and Inner Hair Cell (IHC) attenuation.

    The inputs are the reference and processed signals that are to be
    compared. The reference is at the reference intensity (e.g. 65 dB SPL
    or with NAL-R amplification) and has no other processing. The processed
    signal is the hearing-aid output, and is assumed to have the same or
    greater group delay compared to the reference.

    The function outputs the envelopes of the signals after OHC compression
    and IHC loss attenuation.

    Arguments:
        reference (np.ndarray): reference signal: should be adjusted to 65 dB SPL
            (itype=0 or 1) or to 65 dB SPL plus NAL-R gain (itype=2)
        reference_freq (int): sampling rate for the reference signal, Hz
        processed (np.ndarray): processed signal (e.g. hearing-aid output) includes
            HA gain
        processed_freq (int): sampling rate for the processed signal, Hz
        hearing_loss (np.ndarray): audiogram giving the hearing loss in dB at 6
            audiometric frequencies: [250, 500, 1000, 2000, 4000, 6000] Hz
        itype (int): purpose for the calculation:
             0=intelligibility: reference is normal hearing and must not
               include NAL-R EQ
             1=quality: reference does not include NAL-R EQ
             2=quality: reference already has NAL-R EQ applied
        level1:   level calibration: signal RMS=1 corresponds to Level1 dB SPL
        nchan (int): auditory frequency bands
        m_delay (int): Compensate for the gammatone group delay.
        shift (float): Basal shift of the basilar membrane length

    Returns:
        reference_db (): envelope for the reference in each band
        reference_basilar_membrane (): BM motion for the reference in each band
        processed_db (): envelope for the processed signal in each band
        processed_basilar_membrane (): BM motion for the processed signal in each band
        reference_sl (): compressed RMS average reference in each band converted
            to dB SL
        processed_sl (): compressed RMS average output in each band converted to dB SL
        freq_sample (): sampling rate in Hz for the model outputs

    Updates:
    James M. Kates, 27 October 2011.
    Basilar Membrane added 30 Dec 2011.
    Revised 19 June 2012.
    Remove match of reference RMS level to processed 29 August 2012.
    IHC adaptation added 1 October 2012.
    Basilar Membrane envelope converted to dB SL, 2 Oct 2012.
    Filterbank group delay corrected, 14 Dec 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    Updated by Gerardo Roa Dabike, September 2022.
    """

    # Processing parameters
    # OHC and IHC parameters for the hearing loss
    # Auditory filter center frequencies span 80 to 8000 Hz.
    _center_freq = center_frequency(nchan)  # Center frequencies on an ERB scale

    # Cochlear model parameters for the processed signal
    (
        attn_ohc_y,
        bandwidth_min_y,
        low_knee_y,
        compression_ratio_y,
        attn_ihc_y,
    ) = loss_parameters(hearing_loss, _center_freq)

    # The cochlear model parameters for the reference are the same as for the hearing
    # loss if calculating quality, but are for normal hearing if calculating
    # intelligibility.
    if itype == 0:
        hearing_loss_x = np.zeros(len(hearing_loss))
    else:
        hearing_loss_x = hearing_loss
    [
        attn_ohc_x,
        bandwidth_min_x,
        low_knee_x,
        compression_ratio_x,
        attn_ihc_x,
    ] = loss_parameters(hearing_loss_x, _center_freq)

    # Compute center frequencies for the control
    _center_freq_control = center_frequency(nchan, shift)
    # Maximum BW for the control
    _, bandwidth_1, _, _, _ = loss_parameters(np.full(6, 100), _center_freq_control)

    # Input signal adjustments
    # Convert the signals to 24 kHz sampling rate.
    # Using 24 kHz guarantees that all of the cochlear filters have the same shape
    # independent of the incoming signal sampling rates
    reference_24hz, _ = resample_24khz(reference, reference_freq)
    processed_24hz, freq_sample = resample_24khz(processed, processed_freq)

    # Check file sizes
    min_signal_length = min(len(reference_24hz), len(processed_24hz))
    reference_24hz = reference_24hz[:min_signal_length]
    processed_24hz = processed_24hz[:min_signal_length]

    # Bulk broadband signal alignment
    reference_24hz, processed_24hz = input_align(reference_24hz, processed_24hz)
    nsamp = len(reference_24hz)

    # For HASQI, here add NAL-R equalization if the quality reference doesn't
    # already have it.
    if itype == 1:
        nfir = 140  # Length in samples of the FIR NAL-R EQ filter (24-kHz rate)
        enhancer = NALR(nfir, freq_sample)
        audiogram = Audiogram(
            levels=hearing_loss,
            frequencies=np.array([250, 500, 1000, 2000, 4000, 6000]),
        )
        nalr_fir, _ = enhancer.build(audiogram)
        reference_24hz = enhancer.apply(nalr_fir, reference_24hz)
        reference_24hz = reference_24hz[nfir : nfir + nsamp]

    # Cochlear model
    # Middle ear
    reference_mid = middle_ear(reference_24hz, freq_sample)
    processed_mid = middle_ear(processed_24hz, freq_sample)

    # Initialize storage
    # Reference and processed envelopes and BM motion
    reference_db = np.zeros((nchan, nsamp))
    processed_db = np.zeros((nchan, nsamp))

    # Reference and processed average spectral values
    reference_average = np.zeros(nchan)
    processed_average = np.zeros(nchan)
    reference_control_average = np.zeros(nchan)
    processed_control_average = np.zeros(nchan)

    # Filter bandwidths adjusted for intensity
    reference_bandwidth = np.zeros(nchan)
    processed_bandwidth = np.zeros(nchan)

    reference_b = np.zeros((nchan, nsamp))
    processed_b = np.zeros((nchan, nsamp))

    # Loop over each filter in the auditory filter bank
    for n in range(nchan):
        # Control signal envelopes for the reference and processed signals
        reference_control, _, processed_control, _ = gammatone_basilar_membrane(
            reference_mid,
            bandwidth_1[n],
            processed_mid,
            bandwidth_1[n],
            freq_sample,
            _center_freq_control[n],
        )

        # Adjust the auditory filter bandwidths for the average signal level
        reference_bandwidth[n] = bandwidth_adjust(
            reference_control, bandwidth_min_x[n], bandwidth_1[n], level1
        )
        processed_bandwidth[n] = bandwidth_adjust(
            processed_control, bandwidth_min_y[n], bandwidth_1[n], level1
        )

        # Envelopes and BM motion of the reference and processed signals
        xenv, xbm, yenv, ybm = gammatone_basilar_membrane(
            reference_mid,
            reference_bandwidth[n],
            processed_mid,
            processed_bandwidth[n],
            freq_sample,
            _center_freq[n],
        )

        # RMS levels of the ref and output envelopes for linear metric
        reference_average[n] = np.sqrt(np.mean(xenv**2))
        processed_average[n] = np.sqrt(np.mean(yenv**2))
        reference_control_average[n] = np.sqrt(np.mean(reference_control**2))
        processed_control_average[n] = np.sqrt(np.mean(processed_control**2))

        # Cochlear compression for the signal envelopes and BM motion
        reference_cochlear_compression, reference_b[n] = env_compress_basilar_membrane(
            xenv,
            xbm,
            reference_control,
            attn_ohc_x[n],
            low_knee_x[n],
            compression_ratio_x[n],
            freq_sample,
            level1,
        )
        processed_cochlear_compression, processed_b[n] = env_compress_basilar_membrane(
            yenv,
            ybm,
            processed_control,
            attn_ohc_y[n],
            low_knee_y[n],
            compression_ratio_y[n],
            freq_sample,
            level1,
        )

        # Correct for the delay between the reference and output
        processed_cochlear_compression = envelope_align(
            reference_cochlear_compression, processed_cochlear_compression
        )  # Align processed envelope to reference
        processed_b[n] = envelope_align(
            reference_b[n], processed_b[n]
        )  # Align processed BM motion to reference

        # Convert the compressed envelopes and BM vibration envelopes to dB SPL
        reference_cochlear_compression, reference_b[n] = envelope_sl(
            reference_cochlear_compression, reference_b[n], attn_ihc_x[n], level1
        )
        processed_cochlear_compression, processed_b[n] = envelope_sl(
            processed_cochlear_compression, processed_b[n], attn_ihc_y[n], level1
        )

        # Apply the IHC rapid and short-term adaptation
        delta = 2  # Amount of overshoot
        reference_db[n], reference_b[n] = inner_hair_cell_adaptation(
            reference_cochlear_compression, reference_b[n], delta, freq_sample
        )
        processed_db[n], processed_b[n] = inner_hair_cell_adaptation(
            processed_cochlear_compression, processed_b[n], delta, freq_sample
        )

    # Additive noise level to give the auditory threshold
    ihc_threshold = -10  # Additive noise level, dB re: auditory threshold
    reference_basilar_membrane = basilar_membrane_add_noise(
        reference_b, ihc_threshold, level1
    )
    processed_basilar_membrane = basilar_membrane_add_noise(
        processed_b, ihc_threshold, level1
    )

    # Correct for the gammatone filterbank interchannel group delay.
    if m_delay > 0:
        reference_db = group_delay_compensate(
            reference_db, reference_bandwidth, _center_freq, freq_sample
        )
        processed_db = group_delay_compensate(
            processed_db, reference_bandwidth, _center_freq, freq_sample
        )
        reference_basilar_membrane = group_delay_compensate(
            reference_basilar_membrane, reference_bandwidth, _center_freq, freq_sample
        )
        processed_basilar_membrane = group_delay_compensate(
            processed_basilar_membrane, reference_bandwidth, _center_freq, freq_sample
        )

    # Convert average gammatone outputs to dB SPL
    reference_sl = convert_rms_to_sl(
        reference_average,
        reference_control_average,
        attn_ohc_x,
        low_knee_x,
        compression_ratio_x,
        attn_ihc_x,
        level1,
    )
    processed_sl = convert_rms_to_sl(
        processed_average,
        processed_control_average,
        attn_ohc_y,
        low_knee_y,
        compression_ratio_y,
        attn_ihc_y,
        level1,
    )

    return (
        reference_db,
        reference_basilar_membrane,
        processed_db,
        processed_basilar_membrane,
        reference_sl,
        processed_sl,
        freq_sample,
    )


def center_frequency(
    nchan: int,
    shift: float | None = None,
    low_freq: int = 80,
    high_freq: int = 8000,
    ear_q: float = 9.26449,
    min_bw: float = 24.7,
) -> ndarray:
    """
    Compute the Equivalent Rectangular Bandwidth_[1] frequency spacing for the
    gammatone filter bank. The equation comes from Malcolm Slaney[2].

    Arguments:
        nchan (int): number of filters in the filter bank
        low_freq (int): Low Frequency level.
        high_freq (int): High Frequency level.
        shift (): optional frequency shift of the filter bank specified as a fractional
            shift in distance along the BM. A positive shift is an increase in frequency
            (basal shift), and negative is a decrease in frequency (apical shift). The
            total length of the BM is normalized to 1. The frequency-to-distance map is
            from D.D. Greenwood[3].
        ear_q (float):
        min_bw (float):

    Returns:


    References:
    .. [1] Moore BCJ, Glasberg BR (1983) Suggested formulae for calculating
           auditory-filter bandwidths and excitation patterns. J Acoustical
           Soc America 74:750-753. Available at
           <https://doi.org/10.1121/1.389861>
    .. [2] Slaney M (1993) An Efficient Implemtnation of the Patterson-
           Holdsworth Auditory Filter Bank. Available at:
           <https://asset-pdf.scinapse.io/prod/396690109/396690109.pdf>.
    .. [3] Greenwood DD (1990) A cochlear frequency-position function for
           several species--29 years later. J Acoust Soc Am 87(6):2592-
           2605. Available at
           <https://doi.o10.1121/1.399052>

    Updates:
    James M. Kates, 25 January 2007.
    Frequency shift added 22 August 2008.
    Lower and upper frequencies fixed at 80 and 8000 Hz, 19 June 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    if shift is not None:
        k = 1
        A = 165.4  # pylint: disable=invalid-name
        a = 2.1  # shift specified as a fraction of the total length
        # Locations of the low and high frequencies on the BM between 0 and 1
        x_low = (1 / a) * np.log10(k + (low_freq / A))
        x_high = (1 / a) * np.log10(k + (high_freq / A))
        # Shift the locations
        x_low = x_low * (1 + shift)
        x_high = x_high * (1 + shift)
        # Compute the new frequency range
        low_freq = A * (10 ** (a * x_low) - k)
        high_freq = A * (10 ** (a * x_high) - k)

    # All of the following expressions are derived in Apple TR #35,
    # "An Efficient Implementation of the Patterson-Holdsworth Cochlear
    # Filter Bank" by Malcolm Slaney.
    # https://engineering.purdue.edu/~malcolm/apple/tr35/PattersonsEar.pdf
    _center_freq = -(ear_q * min_bw) + np.exp(
        np.arange(1, nchan)
        * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw))
        / (nchan - 1)
    ) * (high_freq + ear_q * min_bw)
    _center_freq = np.insert(
        _center_freq, 0, high_freq
    )  # Last center frequency is set to highFreq
    _center_freq = np.flip(_center_freq)
    return _center_freq


def loss_parameters(
    hearing_loss: ndarray,
    center_freq: ndarray,
    audiometric_freq: ndarray | None = None,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Apportion the hearing loss to the outer hair cells (OHC) and the inner
    hair cells (IHC) and to increase the bandwidth of the cochlear filters
    in proportion to the OHC fraction of the total loss.

    Arguments:
        hearing_loss (np.ndarray): hearing loss at the 6 audiometric frequencies
        center_freq (np.ndarray): array containing the center frequencies of the
            gammatone filters arranged from low to high
        audiometric_freq (list):

    Returns:
        attenuated_ohc (): attenuation in dB for the OHC gammatone filters
        bandwidth (): OHC filter bandwidth expressed in terms of normal
        low_knee (): Lower kneepoint for the low-level linear amplification
        compression_ratio (): Ranges from 1.4:1 at 150 Hz to 3.5:1 at 8 kHz for normal
            hearing. Reduced in proportion to the OHC loss to 1:1.
        attenuated_ihc ():	attenuation in dB for the input to the IHC synapse

    Updates:
    James M. Kates, 25 January 2007.
    Version for loss in dB and match of OHC loss to CR, 9 March 2007.
    Low-frequency extent changed to 80 Hz, 27 Oct 2011.
    Lower kneepoint set to 30 dB, 19 June 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Audiometric frequencies in Hz
    if audiometric_freq is None:
        audiometric_freq = np.array([250, 500, 1000, 2000, 4000, 6000])

    # Interpolation to give the loss at the gammatone center frequencies
    # Use linear interpolation in dB. The interpolation assumes that
    # cfreq[1] < aud[1] and cfreq[nfilt] > aud[6]
    nfilt = len(center_freq)
    f_v = np.insert(
        audiometric_freq, [0, len(audiometric_freq)], [center_freq[0], center_freq[-1]]
    )

    # Interpolated gain in dB
    loss = np.interp(
        center_freq,
        f_v,
        np.insert(
            hearing_loss, [0, len(hearing_loss)], [hearing_loss[0], hearing_loss[-1]]
        ),
    )
    loss = np.maximum(loss, 0)
    # Make sure there are no negative losses

    # Compression ratio changes linearly with ERB rate from 1.25:1 in the 80-Hz
    # frequency band to 3.5:1 in the 8-kHz frequency band
    compression_ratio = 1.25 + 2.25 * np.arange(nfilt) / (nfilt - 1)

    # Maximum OHC sensitivity loss depends on the compression ratio. The compression
    # I/O curves assume linear below 30 and above 100 dB SPL in normal ears.
    max_ohc = 70 * (
        1 - (1 / compression_ratio)
    )  # HC loss that results in 1:1 compression
    theoretical_ohc = 1.25 * max_ohc  # Loss threshold for adjusting the OHC parameters

    # Apportion the loss in dB to the outer and inner hair cells based on the data of
    # Moore et al (1999), JASA 106, 2761-2778.

    # Reduce the CR towards 1:1 in proportion to the OHC loss.
    attenuated_ohc = 0.8 * np.copy(loss)
    attenuated_ihc = 0.2 * np.copy(loss)

    attenuated_ohc[loss >= theoretical_ohc] = (
        0.8 * theoretical_ohc[loss >= theoretical_ohc]
    )
    attenuated_ihc[loss >= theoretical_ohc] = 0.2 * theoretical_ohc[
        loss >= theoretical_ohc
    ] + (loss[loss >= theoretical_ohc] - theoretical_ohc[loss >= theoretical_ohc])

    # Adjust the OHC bandwidth in proportion to the OHC loss
    bandwidth = np.ones(nfilt)
    bandwidth = bandwidth + (attenuated_ohc / 50.0) + 2.0 * (attenuated_ohc / 50.0) ** 6

    # Compute the compression lower kneepoint and compression ratio
    low_knee = attenuated_ohc + 30
    upamp = 30 + (70 / compression_ratio)  # Output level for an input of 100 dB SPL

    compression_ratio = (100 - low_knee) / (
        upamp + attenuated_ohc - low_knee
    )  # OHC loss Compression ratio

    return attenuated_ohc, bandwidth, low_knee, compression_ratio, attenuated_ihc


def resample_24khz(
    reference_signal: ndarray, reference_freq: float, freq_sample_hz: float = 24000.0
) -> tuple[ndarray, float]:
    """
    Resample the input signal at 24 kHz. The input sampling rate is
    rounded to the nearest kHz to compute the sampling rate conversion
    ratio.

    Arguments:
    reference_signal (np.ndarray): input signal
    reference_freq (int): sampling rate for the input in Hz
    freq_sample_hz (int): Frequency sample in Hz

    Returns:
    reference_signal_24         signal resampled at kHz (default 24Khz)
    freq_sample_hz     output sampling rate in Hz

    Updates
    James M. Kates, 20 June 2011.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Sampling rate information
    sample_rate_target_khz = np.round(
        freq_sample_hz / 1000
    )  # output rate to nearest kHz
    reference_freq_khz = np.round(reference_freq / 1000)

    # Resample the signal
    if reference_freq_khz == sample_rate_target_khz:
        # No resampling performed if the rates match
        return reference_signal, freq_sample_hz

    if reference_freq_khz < sample_rate_target_khz:
        # Resample for the input rate lower than the output
        resample_signal = resample_poly(
            reference_signal, sample_rate_target_khz, reference_freq_khz
        )

        # Match the RMS level of the resampled signal to that of the input
        reference_rms = np.sqrt(np.mean(reference_signal**2))
        resample_rms = np.sqrt(np.mean(resample_signal**2))
        resample_signal = (reference_rms / resample_rms) * resample_signal

        return resample_signal, freq_sample_hz

    # Resample for the input rate higher than the output
    resample_signal = resample_poly(
        reference_signal, sample_rate_target_khz, reference_freq_khz
    )

    # Reduce the input signal bandwidth to 21 kHz (-10.5 to +10.5 kHz)
    # The power equalization is designed to match the signal intensities
    # over the frequency range spanned by the gammatone filter bank.
    # Chebyshev Type 2 LP
    order = 7
    attenuation = 30  # sidelobe attenuation in dB
    reference_freq_cut = 21 / reference_freq_khz
    reference_b, reference_a = cheby2(order, attenuation, reference_freq_cut)
    reference_filter = lfilter(reference_b, reference_a, reference_signal, axis=0)

    # Reduce the resampled signal bandwidth to 21 kHz (-10.5 to +10.5 kHz)
    resample_rate_cut = 21 / sample_rate_target_khz
    target_b, target_a = cheby2(order, attenuation, resample_rate_cut)
    target_filter = lfilter(target_b, target_a, resample_signal, axis=0)

    # Compute the input and output RMS levels within the 21 kHz bandwidth and
    # match the output to the input
    reference_rms = np.sqrt(np.mean(reference_filter**2))
    resample_rms = np.sqrt(np.mean(target_filter**2))
    resample_signal = (reference_rms / resample_rms) * resample_signal

    return resample_signal, freq_sample_hz


def input_align(reference: ndarray, processed: ndarray) -> tuple[ndarray, ndarray]:
    """
    Approximate temporal alignment of the reference and processed output
    signals. Leading and trailing zeros are then pruned.

    The function assumes that the two sequences have the same sampling rate:
    call eb_Resamp24kHz for each sequence first, then call this function to
    align the signals.

    Arguments:
    reference (np.ndarray): input reference sequence
    processed (np.ndarray): hearing-aid output sequence

    Returns:
    reference (np.ndarray): pruned and shifted reference
    processed (np.ndarray): pruned and shifted hearing-aid output

    Updates:
    James M. Kates, 12 July 2011.
    Match the length of the processed output to the reference for the
    purposes of computing the cross-covariance
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Match the length of the processed output to the reference for the purposes
    # of computing the cross-covariance
    reference_n = len(reference)
    processed_n = len(processed)
    min_sample_length = min(reference_n, processed_n)

    # Determine the delay of the output relative to the reference
    reference_processed_correlation = correlate(
        reference[:min_sample_length] - np.mean(reference[:min_sample_length]),
        processed[:min_sample_length] - np.mean(processed[:min_sample_length]),
        "full",
    )  # Matlab code uses xcov thus the subtraction of mean
    index = np.argmax(np.abs(reference_processed_correlation))
    delay = min_sample_length - index - 1

    # Back up 2 msec to allow for dispersion
    fsamp = 24000.0  # Cochlear model input sampling rate in Hz
    delay = np.rint(delay - 2 * fsamp / 1000.0).astype(int)  # Back up 2 ms

    # Align the output with the reference allowing for the dispersion
    if delay > 0:
        # Output delayed relative to the reference
        processed = np.concatenate((processed[delay:processed_n], np.zeros(delay)))
    else:
        # Output advanced relative to the reference
        processed = np.concatenate((np.zeros(-delay), processed[: processed_n + delay]))

    # Find the start and end of the noiseless reference sequence
    reference_abs = np.abs(reference)
    reference_max = np.max(reference_abs)
    reference_threshold = 0.001 * reference_max  # Zero detection threshold

    above_threshold = np.where(reference_abs > reference_threshold)[0]
    reference_n_above_threshold = above_threshold[0]
    reference_n_below_threshold = above_threshold[-1]

    # Prune the sequences to remove the leading and trailing zeros
    reference_n_below_threshold = min(reference_n_below_threshold, processed_n)

    return (
        reference[reference_n_above_threshold : reference_n_below_threshold + 1],
        processed[reference_n_above_threshold : reference_n_below_threshold + 1],
    )


def middle_ear(reference: ndarray, freq_sample: float) -> ndarray:
    """
    Design the middle ear filters and process the input through the
    cascade of filters. The middle ear model is a 2-pole HP filter
    at 350 Hz in series with a 1-pole LP filter at 5000 Hz. The
    result is a rough approximation to the equal-loudness contour
    at threshold.

    Arguments:
    reference (np.ndarray):	input signal
    freq_sample (float): sampling rate in Hz

    Returns:
    xout (): filtered output

    Updates:
    James M. Kates, 18 January 2007.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Design the 1-pole Butterworth LP using the bilinear transformation
    butterworth_low_pass, low_pass = butter(1, 5000 / (0.5 * freq_sample))

    # LP filter the input
    y = lfilter(butterworth_low_pass, low_pass, reference)

    # Design the 2-pole Butterworth HP using the bilinear transformation
    butterworth_high_pass, high_pass = butter(2, 350 / (0.5 * freq_sample), "high")

    # HP filter the signal
    return lfilter(butterworth_high_pass, high_pass, y)


def gammatone_basilar_membrane(
    reference: ndarray,
    reference_bandwidth: float,
    processed: ndarray,
    processed_bandwidth: float,
    freq_sample: float,
    center_freq: float,
    ear_q: float = 9.26449,
    min_bandwidth: float = 24.7,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    4th-order gammatone auditory filter. This implementation is based on the c program
    published on-line by Ning Ma, U. Sheffield, UK[1]_ that gives an implementation of
    the Martin Cooke filters[2]_: an impulse-invariant transformation of the gammatone
    filter. The signal is demodulated down to baseband using a complex exponential,
    and then passed through a cascade of four one-pole low-pass filters.

    This version filters two signals that have the same sampling rate and the same
    gammatone filter center frequencies. The lengths of the two signals should match;
    if they don't, the signals are truncated to the shorter of the two lengths.

    Arguments:
        reference (): first sequence to be filtered
        reference_bandwidth: bandwidth for x relative to that of a normal ear
        processed (): second sequence to be filtered
        processed_bandwidth (): bandwidth for x relative to that of a normal ear
        freq_sample (): sampling rate in Hz
        center_frequency (int): filter center frequency in Hz
        ear_q: (float): ???
        min_bandwidth (float): ???

    Returns:
        reference_envelope (): filter envelope output (modulated down to baseband)
            1st signal
        reference_basilar_membrane (): Basilar Membrane for the first signal
        processed_envelope (): filter envelope output (modulated down to baseband)
            2nd signal
        processed_basilar_membrane (): Basilar Membrane for the second signal

    References:
    .. [1] Ma N, Green P, Barker J, Coy A (2007) Exploiting correlogram
           structure for robust speech recognition with multiple speech
           sources. Speech Communication, 49 (12): 874-891. Available at
           <https://doi.org/10.1016/j.specom.2007.05.003>
           <https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/gammatone/>
    .. [2] Cooke, M. (1993) Modelling auditory processing and organisation.
           Cambridge University Press

    Updates:
    James M. Kates, 8 Jan 2007.
    Vectorized version for efficient MATLAB execution, 4 February 2007.
    Cosine and sine generation, 29 June 2011.
    Output sine and cosine sequences, 19 June 2012.
    Cosine/sine loop speed increased, 9 August 2013.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Filter Equivalent Rectangular Bandwidth from Moore and Glasberg (1983)
    # doi: 10.1121/1.389861
    erb = min_bandwidth + (center_freq / ear_q)

    # Check the lengths of the two signals and trim to shortest
    min_sample = min(len(reference), len(processed))
    x = reference[:min_sample]
    y = processed[:min_sample]

    # Filter the first signal
    # Initialize the filter coefficients
    tpt = 2 * np.pi / freq_sample
    tpt_bw = reference_bandwidth * tpt * erb * 1.019
    a = np.exp(-tpt_bw)
    a_1 = 4.0 * a
    a_2 = -6.0 * a * a
    a_3 = 4.0 * a * a * a
    a_4 = -a * a * a * a
    a_5 = 4.0 * a * a
    gain = 2.0 * (1 - a_1 - a_2 - a_3 - a_4) / (1 + a_1 + a_5)

    # Initialize the complex demodulation
    npts = len(x)
    sincf, coscf = gammatone_bandwidth_demodulation(
        npts, tpt, center_freq, np.zeros(npts), np.zeros(npts)
    )

    # Filter the real and imaginary parts of the signal
    ureal = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], x * coscf)
    uimag = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], x * sincf)
    assert isinstance(ureal, np.ndarray)  # lfilter can return different types
    assert isinstance(uimag, np.ndarray)

    # Extract the BM velocity and the envelope
    reference_basilar_membrane = gain * (ureal * coscf + uimag * sincf)
    reference_envelope = gain * np.sqrt(ureal * ureal + uimag * uimag)

    # Filter the second signal using the existing cosine and sine sequences
    tpt_bw = processed_bandwidth * tpt * erb * 1.019
    a = np.exp(-tpt_bw)
    a_1 = 4.0 * a
    a_2 = -6.0 * a * a
    a_3 = 4.0 * a * a * a
    a_4 = -a * a * a * a
    a_5 = 4.0 * a * a
    gain = 2.0 * (1 - a_1 - a_2 - a_3 - a_4) / (1 + a_1 + a_5)

    # Filter the real and imaginary parts of the signal
    ureal = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], y * coscf)
    uimag = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], y * sincf)
    assert isinstance(ureal, np.ndarray)
    assert isinstance(uimag, np.ndarray)

    # Extract the BM velocity and the envelope
    processed_basilar_membrane = gain * (ureal * coscf + uimag * sincf)
    processed_envelope = gain * np.sqrt(ureal * ureal + uimag * uimag)

    return (
        reference_envelope,
        reference_basilar_membrane,
        processed_envelope,
        processed_basilar_membrane,
    )


@njit
def gammatone_bandwidth_demodulation(
    npts, tpt, center_freq, center_freq_cos, center_freq_sin
):
    """Gamma tone bandwidth demodulation

    Arguments:
        npts (): ???
        tpt (): ???
        center_freq (): ???
        center_freq_cos (): ???
        sincf (): ???

    Returns:
        sincf (): ???
        coscf (): ???
    """
    cos_n = np.cos(tpt * center_freq)
    sin_n = np.sin(tpt * center_freq)
    cold = 1.0
    sold = 0.0
    center_freq_cos[0] = cold
    center_freq_sin[0] = sold
    for n in range(1, npts):
        arg = cold * cos_n + sold * sin_n
        sold = sold * cos_n - cold * sin_n
        cold = arg
        center_freq_cos[n] = cold
        center_freq_sin[n] = sold

    return center_freq_sin, center_freq_cos


def bandwidth_adjust(
    control: ndarray,
    bandwidth_min: float,
    bandwidth_max: float,
    level1: float,
) -> float:
    """
    Compute the increase in auditory filter bandwidth in response to high signal levels.

    Arguments:
        control (): envelope output in the control filter band
        bandwidth_min (): auditory filter bandwidth computed for the loss (or NH)
        bandwidth_max (): auditory filter bandwidth at maximum OHC damage
        level1 ():     RMS=1 corresponds to Level1 dB SPL

    Returns:
        bandwidth (): filter bandwidth increased for high signal levels

    Updates:
    James M. Kates, 21 June 2011.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Compute the control signal level
    control_rms = np.sqrt(np.mean(control**2))
    control_db = 20 * np.log10(control_rms) + level1

    # Adjust the auditory filter bandwidth
    if control_db < 50:
        # No BW adjustment for a signal below 50 dB SPL
        return bandwidth_min
    if control_db > 100:
        # Maximum BW if signal is above 100 dB SPL
        return bandwidth_max
    return bandwidth_min + ((control_db - 50) / 50) * (bandwidth_max - bandwidth_min)


def env_compress_basilar_membrane(
    envsig: ndarray,
    bm: ndarray,  # pylint: disable=invalid-name
    control: ndarray,
    attn_ohc: float,
    threshold_low: float,
    compression_ratio: float,
    fsamp: float,
    level1: float,
    small: float = 1e-30,
    threshold_high: int = 100,
) -> tuple[ndarray, ndarray]:
    """
    Compute the cochlear compression in one auditory filter band. The gain is linear
    below the lower threshold, compressive with a compression ratio of CR:1 between the
    lower and upper thresholds, and reverts to linear above the upper threshold. The
    compressor assumes that auditory threshold is 0 dB SPL.

    Arguments:
        envsig (): analytic signal envelope (magnitude) returned by the
                gammatone filter bank
        bm (): BM motion output by the filter bank
        control (): analytic control envelope returned by the wide control
                path filter bank
        attn_ohc (): OHC attenuation at the input to the compressor
        threshold_Low (): kneepoint for the low-level linear amplification
        compression_ratio (): compression ratio
        fsamp (): sampling rate in Hz
        level1 (): dB reference level: a signal having an RMS value of 1 is
                assigned to Level1 dB SPL.
        small (): ???
        threshold_high: kneepoint for the high-level linear amplification

    Returns:
        compressed_signal (): compressed version of the signal envelope
        compressed_basilar_membrane (): compressed version of the BM motion

    Updates:
    James M. Kates, 19 January 2007.
    LP filter added 15 Feb 2007 (Ref: Zhang et al., 2001)
    Version to compress the envelope, 20 Feb 2007.
    Change in the OHC I/O function, 9 March 2007.
    Two-tone suppression added 22 August 2008.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Convert the control envelope to dB SPL
    logenv = np.maximum(control, small)
    logenv = level1 + 20 * np.log10(logenv)
    logenv = np.minimum(
        logenv, threshold_high
    )  # Clip signal levels above the upper threshold
    logenv = np.maximum(logenv, threshold_low)  # Clip signal at the lower threshold

    # Compute the compression gain in dB
    gain = -attn_ohc - (logenv - threshold_low) * (1 - (1 / compression_ratio))

    # Convert the gain to linear and apply a LP filter to give a 0.2 ms delay
    gain = 10 ** (gain / 20)
    flp = 800
    b, a = butter(1, flp / (0.5 * fsamp))
    gain = lfilter(b, a, gain)

    # Apply the gain to the signals
    compressed_signal = gain * envsig
    compressed_basilar_membrane = gain * bm

    return compressed_signal, compressed_basilar_membrane


def envelope_align(
    reference: ndarray, output: ndarray, freq_sample: int = 24000, corr_range: int = 100
) -> ndarray:
    """
    Align the envelope of the processed signal to that of the reference signal.

    Arguments:
        reference (): envelope or BM motion of the reference signal
        output (): envelope or BM motion of the output signal
        freq_sample (int): Frequency sample rate in Hz
        corr_range (int): range in msec for the correlation

    Returns:
        y (): shifted output envelope to match the input

    Updates:
    James M. Kates, 28 October 2011.
    Absolute value of the cross-correlation peak removed, 22 June 2012.
    Cross-correlation range reduced, 13 August 2013.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # The MATLAB code limits the range of lags to search (to 100 ms) to save computation
    # time - no such option exists in numpy, but the code below limits the delay to the
    # same range as in Matlab, for consistent results
    lags = np.rint(0.001 * corr_range * freq_sample).astype(int)  # Range in samples
    npts = len(reference)
    lags = min(lags, npts)

    ref_out_correlation = correlate(reference, output, "full")
    location = np.argmax(
        ref_out_correlation[npts - lags : npts + lags]
    )  # Limit the range in which
    delay = lags - location - 1

    # Time shift the output sequence
    if delay > 0:
        # Output delayed relative to the reference
        return np.concatenate((output[delay:npts], np.zeros(delay)))
    return np.concatenate((np.zeros(-delay), output[: npts + delay]))


def envelope_sl(
    reference: ndarray,
    basilar_membrane: ndarray,
    attenuated_ihc: float,
    level1: float,
    small: float = 1e-30,
) -> tuple[ndarray, ndarray]:
    """
    Convert the compressed envelope returned by cochlear_envcomp to dB SL.

    Arguments:
        reference (): linear envelope after compression
        basilar_membrane (): linear Basilar Membrane vibration after compression
        attenuated_ihc (): IHC attenuation at the input to the synapse
        level1 (): level in dB SPL corresponding to 1 RMS
        small (float): ???

    Returns:
        _reference (): reference envelope in dB SL
        _basilar_membrane (): Basilar Membrane vibration with envelope converted to
            dB SL

    Updates:
    James M. Kates, 20 Feb 07.
    IHC attenuation added 9 March 2007.
    Basilar membrane vibration conversion added 2 October 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Convert the envelope to dB SL
    _reference = level1 - attenuated_ihc + 20 * np.log10(reference + small)
    _reference = np.maximum(_reference, 0)

    # Convert the linear BM motion to have a dB SL envelope
    gain = (_reference + small) / (reference + small)
    _basilar_membrane = gain * basilar_membrane

    return _reference, _basilar_membrane


@njit
def inner_hair_cell_adaptation(
    reference_db, reference_basilar_membrane, delta, freq_sample
):
    """
    Provide inner hair cell (IHC) adaptation. The adaptation is based on an
    equivalent RC circuit model, and the derivatives are mapped into
    1st-order backward differences. Rapid and short-term adaptation are
    provided. The input is the signal envelope in dB SL, with IHC attenuation
    already applied to the envelope. The outputs are the envelope in dB SL
    with adaptation providing overshoot of the long-term output level, and
    the BM motion is multiplied by a gain vs. time function that reproduces
    the adaptation. IHC attenuation and additive noise for the equivalent
    auditory threshold are provided by a subsequent call to eb_BMatten.

    Arguments:
        reference_db (np.ndarray): signal envelope in one frequency band in dB SL
             contains OHC compression and IHC attenuation
        reference_basilar_membrane (): basilar membrane vibration with OHC compression
            but no IHC attenuation
        delta (): overshoot factor = delta x steady-state
        freq_sample (int): sampling rate in Hz

    Returns:
        output_db (): envelope in dB SL with IHC adaptation
        output_basilar_membrane (): Basilar Membrane multiplied by the IHC adaptation
            gain function

    Updates:
    James M. Kates, 1 October 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Test the amount of overshoot
    dsmall = 1.0001
    delta = max(delta, dsmall)

    # Initialize adaptation time constants
    tau1 = 2  # Rapid adaptation in msec
    tau2 = 60  # Short-term adaptation in msec
    tau1 = 0.001 * tau1  # Convert to seconds
    tau2 = 0.001 * tau2

    # Equivalent circuit parameters
    freq_sample_inverse = 1 / freq_sample
    r_1 = 1 / delta
    r_2 = 0.5 * (1 - r_1)
    r_3 = r_2
    c_1 = tau1 * (r_1 + r_2) / (r_1 * r_2)
    c_2 = tau2 / ((r_1 + r_2) * r_3)

    # Intermediate values used for the voltage update matrix inversion
    a11 = r_1 + r_2 + r_1 * r_2 * (c_1 / freq_sample_inverse)
    a12 = -r_1
    a21 = -r_3
    a22 = r_2 + r_3 + r_2 * r_3 * (c_2 / freq_sample_inverse)
    denom = 1 / ((a11 * a22) - (a21 * a12))

    # Additional intermediate values
    r_1_inv = 1 / r_1
    product_r1_r2_c1 = r_1 * r_2 * (c_1 / freq_sample_inverse)
    product_r2_r3_c2 = r_2 * r_3 * (c_2 / freq_sample_inverse)

    # Initialize the outputs and state of the equivalent circuit
    nsamp = len(reference_db)
    gain = np.ones_like(
        reference_db
    )  # Gain vector to apply to the BM motion, default is 1
    output_db = np.zeros_like(reference_db)
    v_1 = 0
    v_2 = 0
    small = 1e-30

    # Loop to process the envelope signal
    # The gain asymptote is 1 for an input envelope of 0 dB SPL
    for n in range(nsamp):
        v_0 = reference_db[n]
        b_1 = v_0 * r_2 + product_r1_r2_c1 * v_1
        b_2 = product_r2_r3_c2 * v_2
        v_1 = denom * (a22 * b_1 - a12 * b_2)
        v_2 = denom * (-a21 * b_1 + a11 * b_2)
        out = (v_0 - v_1) * r_1_inv
        output_db[n] = out

    output_db = np.maximum(output_db, 0)
    gain = (output_db + small) / (reference_db + small)

    output_basilar_membrane = gain * reference_basilar_membrane

    return output_db, output_basilar_membrane


def basilar_membrane_add_noise(
    reference: ndarray, threshold: int, level1: float
) -> ndarray:
    """
    Apply the IHC attenuation to the BM motion and to add a low-level Gaussian noise to
    give the auditory threshold.

    Arguments:
        reference (): BM motion to be attenuated
        threshold (): additive noise level in dB re:auditory threshold
        level1 (): an input having RMS=1 corresponds to Level1 dB SPL

    Returns:
        Attenuated signal with threshold noise added

    Updates:
        James M. Kates, 19 June 2012.
        Just additive noise, 2 Oct 2012.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    gain = 10 ** ((threshold - level1) / 20)  # Linear gain for the noise

    # rng = np.random.default_rng()
    noise = gain * np.random.standard_normal(
        reference.shape
    )  # Gaussian RMS=1, then attenuated
    return reference + noise


def group_delay_compensate(
    reference: ndarray,
    bandwidths: ndarray,
    center_freq: ndarray,
    freq_sample: float,
    ear_q: float = 9.26449,
    min_bandwidth: float = 24.7,
) -> ndarray:
    """
    Compensate for the group delay of the gammatone filter bank. The group
    delay is computed for each filter at its center frequency. The firing
    rate output of the IHC model is then adjusted so that all outputs have
    the same group delay.

    Arguments:
        xenv (np.ndarray): matrix of signal envelopes or BM motion
        bandwidths (): gammatone filter bandwidths adjusted for loss
        center_freq (): center frequencies of the bands
        freq_sample (): sampling rate for the input signal in Hz (e.g. 24,000 Hz)
        ear_q (float):
        min_bandwidth (float) :

    Returns:
        processed (): envelopes or BM motion compensated for the group delay.

    Updates:
        James M. Kates, 28 October 2011.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Processing parameters
    nchan = len(bandwidths)

    # Filter ERB from Moore and Glasberg (1983)
    erb = min_bandwidth + (center_freq / ear_q)

    # Initialize the gammatone filter coefficients
    tpt = 2 * np.pi / freq_sample
    tpt_bandwidth = tpt * 1.019 * bandwidths * erb
    a = np.exp(-tpt_bandwidth)
    a_1 = 4.0 * a
    a_2 = -6.0 * a * a
    a_3 = 4.0 * a * a * a
    a_4 = -a * a * a * a
    a_5 = 4.0 * a * a

    # Compute the group delay in samples at fsamp for each filter
    _group_delay = np.zeros(nchan)
    for n in range(nchan):
        _, _group_delay[n] = group_delay(
            ([1, a_1[n], a_5[n]], [1, -a_1[n], -a_2[n], -a_3[n], -a_4[n]]), 1
        )
    _group_delay = np.rint(_group_delay).astype(int)  # convert to integer samples

    # Compute the delay correlation
    group_delay_min = np.min(_group_delay)
    _group_delay = (
        _group_delay - group_delay_min
    )  # Remove the minimum delay from all the over values
    group_delay_max = np.max(_group_delay)
    correct = (
        group_delay_max - _group_delay
    )  # Samples delay needed to add to give alignment

    # Add delay correction to each frequency band
    processed = np.zeros(reference.shape)
    for n in range(nchan):
        ref = reference[n]
        npts = len(ref)
        processed[n] = np.concatenate((np.zeros(correct[n]), ref[: npts - correct[n]]))

    return processed


def convert_rms_to_sl(
    reference: ndarray,
    control: ndarray,
    attenuated_ohc: ndarray | float,
    threshold_low: ndarray | int,
    compression_ratio: ndarray | int,
    attenuated_ihc: ndarray | float,
    level1: float,
    threshold_high: int = 100,
    small: float = 1e-30,
) -> ndarray:
    """
    Covert the Root Mean Square average output of the gammatone filter bank
    into dB SL. The gain is linear below the lower threshold, compressive
    with a compression ratio of CR:1 between the lower and upper thresholds,
    and reverts to linear above the upper threshold. The compressor
    assumes that auditory threshold is 0 dB SPL.

    Arguments:
        reference (): analytic signal envelope (magnitude) returned by the
        gammatone filter bank, RMS average level
        control (): control signal envelope
        attenuated_ohc (): OHC attenuation at the input to the compressor
        threshold_low (): kneepoint for the low-level linear amplification
        compression_ratio (): compression ratio
        attenuated_ihc (): IHC attenuation at the input to the synapse
        level1 (): dB reference level: a signal having an RMS value of 1 is
                assigned to Level1 dB SPL.
        threshold_high (int):
        small (float):

    Returns:
        reference_db (): compressed output in dB above the impaired threshold

    Updates:
        James M. Kates, 6 August 2007.
        Version for two-tone suppression, 29 August 2008.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Initialize the compression parameters
    threshold_high = 100  # Upper compression threshold

    # Convert the control to dB SPL
    small = 1e-30
    control_db_spl = np.maximum(control, small)
    control_db_spl = level1 + 20 * np.log10(control_db_spl)
    control_db_spl = np.minimum(control_db_spl, threshold_high)
    control_db_spl = np.maximum(control_db_spl, threshold_low)

    # Compute compression gain in dB
    gain = -attenuated_ohc - (control_db_spl - threshold_low) * (
        1 - (1 / compression_ratio)
    )

    # Convert the signal envelope to dB SPL
    control_db_spl = np.maximum(reference, small)
    control_db_spl = level1 + 20 * np.log10(control_db_spl)
    control_db_spl = np.maximum(control_db_spl, 0)
    reference_db = control_db_spl + gain - attenuated_ihc
    reference_db = np.maximum(reference_db, 0)

    return reference_db


def env_smooth(envelopes: np.ndarray, segment_size: int, sample_rate: float) -> ndarray:
    """
    Function to smooth the envelope returned by the cochlear model. The
    envelope is divided into segments having a 50% overlap. Each segment is
    windowed, summed, and divided by the window sum to produce the average.
    A raised cosine window is used. The envelope sub-sampling frequency is
    2*(1000/segsize).

    Arguments:
        envelopes (np.ndarray): matrix of envelopes in each of the auditory bands
        segment_size: averaging segment size in msec
        freq_sample (int): input envelope sampling rate in Hz

    Returns:
        smooth: matrix of subsampled windowed averages in each band

    Updates:
        James M. Kates, 26 January 2007.
        Final half segment added 27 August 2012.
        Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Compute the window
    n_samples = int(
        np.around(segment_size * (0.001 * sample_rate))
    )  # Segment size in samples
    test = n_samples - 2 * np.floor(n_samples / 2)  # 0=even, 1=odd
    if test > 0:
        # Force window length to be even
        n_samples = n_samples + 1
    window = np.hanning(n_samples)  # Raised cosine von Hann window
    wsum = np.sum(window)  # Sum for normalization

    #  The first segment has a half window
    nhalf = int(n_samples / 2)
    halfwindow = window[nhalf:n_samples]
    halfsum = np.sum(halfwindow)

    # Number of segments and assign the matrix storage
    n_channels = np.size(envelopes, 0)
    npts = np.size(envelopes, 1)
    nseg = int(
        1 + np.floor(npts / n_samples) + np.floor((npts - n_samples / 2) / n_samples)
    )
    smooth = np.zeros((n_channels, nseg))

    #  Loop to compute the envelope in each frequency band
    for k in range(n_channels):
        # Extract the envelope in the frequency band
        r = envelopes[k, :]  # pylint: disable=invalid-name

        # The first (half) windowed segment
        nstart = 0
        smooth[k, 0] = np.sum(r[nstart:nhalf] * halfwindow.conj().transpose()) / halfsum

        # Loop over the remaining full segments, 50% overlap
        for n in range(1, nseg - 1):
            nstart = int(nstart + nhalf)
            nstop = int(nstart + n_samples)
            smooth[k, n] = sum(r[nstart:nstop] * window.conj().transpose()) / wsum

        # The last (half) windowed segment
        nstart = nstart + nhalf
        nstop = nstart + nhalf
        smooth[k, nseg - 1] = (
            np.sum(r[nstart:nstop] * window[:nhalf].conj().transpose()) / halfsum
        )

    return smooth


def mel_cepstrum_correlation(
    reference: ndarray,
    distorted: ndarray,
    threshold: float,
    addnoise: float,
) -> tuple[float, ndarray]:
    """
    Compute the cross-correlations between the input signal time-frequency
    envelope and the distortion time-frequency envelope.

    For each time interval, the log spectrum is fitted with a set of
    half-cosine basis functions. The spectrum weighted by the basis
    functions corresponds to Mel Cepstral Coefficients computed in the
    frequency domain. The amplitude-normalized cross-covariance between
    the time-varying basis functions for the input and output signals is
    then computed.

    Arguments:
        reference (): subsampled input signal envelope in dB SL in each critical band
        distorted (): subsampled distorted output signal envelope
        threshold (): threshold in dB SPL to include segment in calculation
        addnoise (): additive Gaussian noise to ensure 0 cross-corr at low levels

    Returns:
        average_cepstral_correlation : average cepstral correlation 2-6, input vs output
        individual_cepstral_correlations : individual cepstral correlations,
            input vs output

    Updates:
        James M. Kates, 24 October 2006.
        Difference signal removed for cochlear model, 31 January 2007.
        Absolute value added 13 May 2011.
        Changed to loudness criterion for silence threshsold, 28 August 2012.
        Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Processing parameters
    nbands = reference.shape[0]

    # Mel cepstrum basis functions (mel cepstrum because of auditory bands)
    nbasis = 6  # Number of cepstral coefficients to be used
    freq = np.arange(nbasis)
    k = np.arange(nbands)
    basis = np.cos(np.outer(k, freq) * np.pi / float(nbands - 1))
    mel_cepstral = basis / np.linalg.norm(basis, axis=0)

    # Find the segments that lie sufficiently above the quiescent rate
    reference_linear = 10 ** (
        reference / 20
    )  # Convert envelope dB to linear (specific loudness)
    xsum = np.sum(reference_linear, 0) / nbands  # Proportional to loudness in sones
    xsum = 20 * np.log10(xsum)  # Convert back to dB (loudness in phons)
    index = np.where(xsum > threshold)[0]  # Identify those segments above threshold
    nsamp = index.shape[0]  # Number of segments above threshold

    # Exit if not enough segments above zero
    average_cepstral_correlation = 0.0
    individual_cepstral_correlations = np.zeros(nbasis)
    if nsamp <= 1:
        logger.warning(
            "Function MelCepstrumCorrelation: Signal below threshold, outputs set to 0."
        )
        return average_cepstral_correlation, individual_cepstral_correlations

    # Remove the silent intervals
    ref = reference[:, index]
    proc = distorted[:, index]

    # Add the low-level noise to the envelopes
    ref = ref + addnoise * np.random.standard_normal(ref.shape)
    proc = proc + addnoise * np.random.standard_normal(proc.shape)

    # Compute the mel cepstrum coefficients using only those segments
    # above threshold

    reference_cep = np.dot(mel_cepstral.T, ref)
    processed_cep = np.dot(mel_cepstral.T, proc)

    # Remove the average value from the cepstral coefficients. The
    # cross-correlation thus becomes a cross-covariance, and there
    # is no effect of the absolute signal level in dB.
    reference_cep -= np.mean(reference_cep, axis=1, keepdims=True)
    processed_cep -= np.mean(processed_cep, axis=1, keepdims=True)

    # Normalized cross-correlations between the time-varying cepstral coeff
    # individual_cepstral_correlations = np.zeros(nbasis)  # Input vs output
    small = 1.0e-30
    xsum = np.sum(reference_cep**2, axis=1)
    ysum = np.sum(processed_cep**2, axis=1)
    mask = (xsum < small) | (ysum < small)
    individual_cepstral_correlations = np.zeros(nbasis)
    individual_cepstral_correlations[~mask] = np.abs(
        np.sum(reference_cep[~mask] * processed_cep[~mask], axis=1)
        / np.sqrt(xsum[~mask] * ysum[~mask])
    )

    # Figure of merit is the average of the cepstral correlations, ignoring
    # the first (average spectrum level).
    average_cepstral_correlation = np.sum(
        individual_cepstral_correlations[1:nbasis]
    ) / (nbasis - 1)
    return average_cepstral_correlation, individual_cepstral_correlations


def melcor9(
    reference: ndarray,
    distorted: ndarray,
    threshold: float,
    add_noise: float,
    segment_size: int,
    n_cepstral_coef: int = 6,
) -> tuple[float, float, float, ndarray]:
    """
    Compute the cross-correlations between the input signal
    time-frequency envelope and the distortion time-frequency envelope. For
    each time interval, the log spectrum is fitted with a set of half-cosine
    basis functions. The spectrum weighted by the basis functions corresponds
    to mel cepstral coefficients computed in the frequency domain. The
    amplitude-normalized cross-covariance between the time-varying basis
    functions for the input and output signals is then computed for each of
    the 8 modulation frequencies.

    Arguments:
        reference (): subsampled input signal envelope in dB SL in each critical band
        distorted (): subsampled distorted output signal envelope
        threshold (): threshold in dB SPL to include segment in calculation
        add_noise (): additive Gaussian noise to ensure 0 cross-corr at low levels
        segment_size (): segment size in ms used for the envelope LP filter (8 msec)
        n_cepstral_coef (int): Number of cepstral coefficients

    Returns:
        mel_cepstral_average (): average of the modulation correlations across analysis
            frequency bands and modulation frequency bands, basis functions 2 -6
        mel_cepstral_low (): average over the four lower mod freq bands, 0 - 20 Hz
        mel_cepstral_high (): average over the four higher mod freq bands, 20 - 125 Hz
        mel_cepstral_modulation (): vector of cross-correlations by modulation
            frequency, averaged over analysis frequency band

    Updates:
        James M. Kates, 24 October 2006.
        Difference signal removed for cochlear model, 31 January 2007.
        Absolute value added 13 May 2011.
        Changed to loudness criterion for silence threshold, 28 August 2012.
        Version using envelope modulation filters, 15 July 2014.
        Modulation frequency vector output added 27 August 2014.
        Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Processing parameters
    nbands = reference.shape[0]

    # Mel cepstrum basis functions (mel cepstrum because of auditory bands)
    freq = np.arange(n_cepstral_coef)
    k = np.arange(nbands)
    basis = np.cos(np.outer(k, freq) * np.pi / (nbands - 1))
    cepm = basis / np.linalg.norm(basis, axis=0, keepdims=True)

    # Find the segments that lie sufficiently above the quiescent rate
    # Convert envelope dB to linear (specific loudness)
    reference_linear = 10 ** (reference / 20)

    # Proportional to loudness in sones
    reference_sum = np.sum(reference_linear, 0) / nbands

    # Convert back to dB (loudness in phons)
    reference_sum = 20 * np.log10(reference_sum)

    # Identify those segments above threshold
    index = np.where(reference_sum > threshold)[0]

    segments_above_threshold = index.shape[0]  # Number of segments above threshold

    # Modulation filter bands, segment size is 8 msec
    edge = [4.0, 8.0, 12.5, 20.0, 32.0, 50.0, 80.0]  # 8 bands covering 0 to 125 Hz
    n_modulation_filter_bands = 1 + len(edge)  # Number of modulation filter bands

    # Exit if not enough segments above zero
    mel_cepstral_average = 0.0
    mel_cepstral_low = 0.0
    mel_cepstral_high = 0.0
    mel_cepstral_modulation = np.zeros(n_modulation_filter_bands)
    if segments_above_threshold <= 1:
        logger.warning("Function melcor9: Signal below threshold, outputs set to 0.")
        return (
            mel_cepstral_average,
            mel_cepstral_low,
            mel_cepstral_high,
            mel_cepstral_modulation,
        )

    # Remove the silent intervals
    _reference = reference[:, index]
    _distorted = distorted[:, index]

    # Add the low-level noise to the envelopes
    _reference += add_noise * np.random.standard_normal(_reference.shape)
    _distorted += add_noise * np.random.standard_normal(_distorted.shape)

    # Compute the mel cepstrum coefficients using only those segments
    # above threshold
    reference_cep = np.dot(cepm.T, _reference[:, :segments_above_threshold])
    distorted_cep = np.dot(cepm.T, _distorted[:, :segments_above_threshold])

    reference_cep -= np.mean(reference_cep, axis=1, keepdims=True)
    distorted_cep -= np.mean(distorted_cep, axis=1, keepdims=True)

    # Envelope sampling parameters
    sampling_freq = 1000.0 / (0.5 * segment_size)  # Envelope sampling frequency in Hz
    nyquist_freq = 0.5 * sampling_freq  # Envelope Nyquist frequency

    # Design the linear-phase envelope modulation filters
    n_fir = np.around(
        128 * (nyquist_freq / 125)
    )  # Adjust filter length to sampling rate
    n_fir = int(2 * np.floor(n_fir / 2))  # Force an even filter length
    b = np.zeros((n_modulation_filter_bands, n_fir + 1))

    # LP filter 0-4 Hz
    b[0, :] = firwin(
        n_fir + 1, edge[0] / nyquist_freq, window="hann", pass_zero="lowpass"
    )
    # HP 80-125 Hz
    b[n_modulation_filter_bands - 1, :] = firwin(
        n_fir + 1,
        edge[n_modulation_filter_bands - 2] / nyquist_freq,
        window="hann",
        pass_zero="highpass",
    )
    # Bandpass filter
    for m in range(1, n_modulation_filter_bands - 1):
        b[m, :] = firwin(
            n_fir + 1,
            [edge[m - 1] / nyquist_freq, edge[m] / nyquist_freq],
            window="hann",
            pass_zero="bandpass",
        )

    mel_cepstral_cross_covar = melcor9_crosscovmatrix(
        b,
        n_modulation_filter_bands,
        n_cepstral_coef,
        segments_above_threshold,
        n_fir,
        reference_cep,
        distorted_cep,
    )

    mel_cepstral_average = np.sum(mel_cepstral_cross_covar[:, 1:], axis=(0, 1))
    mel_cepstral_average /= n_modulation_filter_bands * (n_cepstral_coef - 1)

    mel_cepstral_low = np.sum(mel_cepstral_cross_covar[:4, 1:])
    mel_cepstral_low /= 4 * (n_cepstral_coef - 1)

    mel_cepstral_high = np.sum(mel_cepstral_cross_covar[4:8, 1:])
    mel_cepstral_high /= 4 * (n_cepstral_coef - 1)

    mel_cepstral_modulation = np.mean(mel_cepstral_cross_covar[:, 1:], axis=1)

    return (
        mel_cepstral_average,
        mel_cepstral_low,
        mel_cepstral_high,
        mel_cepstral_modulation,
    )


def melcor9_crosscovmatrix(
    b: ndarray,
    nmod: int,
    nbasis: int,
    nsamp: int,
    nfir: int,
    reference_cep: ndarray,
    processed_cep: ndarray,
) -> ndarray:
    """Compute the cross-covariance matrix.

    Arguments:
        b (): ???
        nmod (): ???
        nbasis (): ???
        nsamp (): ???
        nfir (): ???
        xcep (): ???
        ycep (): ???

    Returns:
        cross_covariance_matrix ():
    """
    small = 1.0e-30
    nfir2 = nfir / 2
    # Convolve the input and output envelopes with the modulation filters
    reference = np.zeros((nmod, nbasis, nsamp))
    processed = np.zeros((nmod, nbasis, nsamp))
    for m in range(nmod):
        for j in range(nbasis):
            # Convolve and remove transients
            c = convolve(b[m], reference_cep[j, :], mode="full")
            reference[m, j, :] = c[int(nfir2) : int(nfir2 + nsamp)]
            c = convolve(b[m], processed_cep[j, :], mode="full")
            processed[m, j, :] = c[int(nfir2) : int(nfir2 + nsamp)]

    # Compute the cross-covariance matrix
    cross_covariance_matrix = np.zeros((nmod, nbasis))

    for m in range(nmod):
        # Input freq band j, modulation freq m
        x_j = reference[m]
        x_j -= np.mean(x_j, axis=1, keepdims=True)
        reference_sum = np.sum(x_j**2, axis=1)

        # Processed signal band
        y_j = processed[m]
        y_j -= np.mean(y_j, axis=1, keepdims=True)
        processed_sum = np.sum(y_j**2, axis=1)

        xy = np.sum(x_j * y_j, axis=1)
        mask = (reference_sum < small) | (processed_sum < small)
        cross_covariance_matrix[m, ~mask] = np.abs(xy[~mask]) / np.sqrt(
            reference_sum[~mask] * processed_sum[~mask]
        )

    return cross_covariance_matrix


def spectrum_diff(
    reference_sl: ndarray, processed_sl: ndarray
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Compute changes in the long-term spectrum and spectral slope.

    The metric is based on the spectral distortion metric of Moore and Tan[1]_
    (JAES, Vol 52, pp 900-914). The log envelopes in dB SL are converted to
    linear to approximate specific loudness. The outputs are the sum of the
    absolute differences, the standard deviation of the differences, and the
    maximum absolute difference. The same three outputs are provided for the
    normalized spectral difference and for the slope. The output is
    calibrated so that a processed signal having 0 amplitude produces a
    value of 1 for the spectrum difference.

    Abs diff: weight all deviations uniformly
    Std diff: weight larger deviations more than smaller deviations
    Max diff: only weight the largest deviation

    Arguments:
        reference_sl (np.ndarray): reference signal spectrum in dB SL
        processed_sl (np.ndarray): degraded signal spectrum in dB SL

    Returns:
        dloud (np.array) : [sum abs diff, std dev diff, max diff] spectra
        dnorm (np.array) : [sum abs diff, std dev diff, max diff] norm spectra
        dslope (np.array) : [sum abs diff, std dev diff, max diff] slope

    References:
    .. [1] Moore BCJ, Tan, CT (2004) Development and Validation of a Method
           for Predicting the Perceived Naturalness of Sounds Subjected to
           Spectral Distortion J Audio Eng Soc 52(9):900-914. Available at.
           <http://www.aes.org/e-lib/browse.cfm?elib=13018>.

    Updates:
        James M. Kates, 28 June 2012.
        Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Convert the dB SL to linear magnitude values. Because of the auditory
    # filter bank, the OHC compression, and auditory threshold, the linear
    # values are closely related to specific loudness.
    nbands = reference_sl.shape[0]
    reference_linear_magnitude = 10 ** (reference_sl / 20)
    processed_linear_magnitude = 10 ** (processed_sl / 20)

    # Normalize the level of the reference and degraded signals to have the
    # same loudness. Thus overall level is ignored while differences in
    # spectral shape are measured.
    reference_sum = np.sum(reference_linear_magnitude)
    reference_linear_magnitude /= (
        reference_sum  # Loudness sum = 1 (arbitrary amplitude, proportional to sones)
    )
    processed_sum = np.sum(processed_linear_magnitude)
    processed_linear_magnitude /= processed_sum

    # Compute the spectrum difference
    dloud = np.zeros(3)
    diff_spectrum = (
        reference_linear_magnitude - processed_linear_magnitude
    )  # Difference in specific loudness in each band
    dloud[0] = np.sum(np.abs(diff_spectrum))
    dloud[1] = nbands * np.std(diff_spectrum)  # Biased std: second moment
    dloud[2] = np.max(np.abs(diff_spectrum))

    # Compute the normalized spectrum difference
    dnorm = np.zeros(3)
    diff_normalised_spectrum = (
        reference_linear_magnitude - processed_linear_magnitude
    ) / (
        reference_linear_magnitude + processed_linear_magnitude
    )  # Relative difference in specific loudness
    dnorm[0] = np.sum(np.abs(diff_normalised_spectrum))
    dnorm[1] = nbands * np.std(diff_normalised_spectrum)
    dnorm[2] = np.max(np.abs(diff_normalised_spectrum))

    # Compute the slope difference
    dslope = np.zeros(3)
    reference_slope = (
        reference_linear_magnitude[1:nbands]
        - reference_linear_magnitude[0 : nbands - 1]
    )
    processed_slope = (
        processed_linear_magnitude[1:nbands]
        - processed_linear_magnitude[0 : nbands - 1]
    )
    diff_slope = reference_slope - processed_slope  # Slope difference
    dslope[0] = np.sum(np.abs(diff_slope))
    dslope[1] = nbands * np.std(diff_slope)
    dslope[2] = np.max(np.abs(diff_slope))

    return dloud, dnorm, dslope


def bm_covary(
    reference_basilar_membrane: ndarray,
    processed_basilar_membrane: ndarray,
    segment_size: int,
    sample_rate: float,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Compute the cross-covariance (normalized cross-correlation) between  the reference
    and processed signals in each auditory band. The signals are divided into segments
    having 50% overlap.

    Arguments:
        reference_basilar_membrane (): Basilar Membrane movement, reference signal
        processed_basilar_membrane (): Basilar Membrane movement, processed signal
        segment_size (): signal segment size, msec
        freq_sample (int): sampling rate in Hz

    Returns:
        signal_cross_covariance (np.array) : [nchan,nseg] of cross-covariance values
        reference_mean_square (np.array) : [nchan,nseg] of MS input signal energy values
        processed_mean_square (np.array) : [nchan,nseg] of MS processed signal energy
            values

    Updates:
        James M. Kates, 28 August 2012.
        Output amplitude adjustment added, 30 october 2012.
        Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Initialize parameters
    small = 1.0e-30

    # Lag for computing the cross-covariance
    lagsize = 1.0  # Lag (+/-) in msec
    maxlag = np.around(lagsize * (0.001 * sample_rate))  # Lag in samples

    # Compute the segment size in samples
    nwin = int(np.around(segment_size * (0.001 * sample_rate)))

    nwin += nwin % 2 == 1  # Force window length to be even
    window = np.hanning(nwin).conj().transpose()  # Raised cosine von Hann window

    # compute inverted Window autocorrelation
    win_corr = correlate(window, window, "full")
    start_sample = int(len(window) - 1 - maxlag)
    end_sample = int(maxlag + len(window))
    if start_sample < 0:
        raise ValueError("segment size too small")
    win_corr = 1 / win_corr[start_sample:end_sample]
    win_sum2 = 1.0 / np.sum(window**2)  # Window power, inverted

    # The first segment has a half window
    nhalf = int(nwin / 2)
    half_window = window[nhalf:nwin]
    half_corr = correlate(half_window, half_window, "full")
    start_sample = int(len(half_window) - 1 - maxlag)
    end_sample = int(maxlag + len(half_window))
    if start_sample < 0:
        raise ValueError("segment size too small")
    half_corr = 1 / half_corr[start_sample:end_sample]
    halfsum2 = 1.0 / np.sum(half_window**2)  # MS sum normalization, first segment

    # Number of segments
    nchan = reference_basilar_membrane.shape[0]
    npts = reference_basilar_membrane.shape[1]
    nseg = int(1 + np.floor(npts / nwin) + np.floor((npts - nwin / 2) / nwin))

    reference_mean_square = np.zeros((nchan, nseg))
    processed_mean_square = np.zeros((nchan, nseg))
    signal_cross_covariance = np.zeros((nchan, nseg))

    # Loop to compute the signal mean-squared level in each band for each
    # segment and to compute the cross-corvariances.
    for k in range(nchan):
        # Extract the BM motion in the frequency band
        x = reference_basilar_membrane[k, :]
        y = processed_basilar_membrane[k, :]

        # The first (half) windowed segment
        nstart = 0
        reference_seg = x[nstart:nhalf] * half_window  # Window the reference
        processed_seg = y[nstart:nhalf] * half_window  # Window the processed signal
        reference_seg = reference_seg - np.mean(reference_seg)  # Make 0-mean
        processed_seg = processed_seg - np.mean(processed_seg)

        # Normalize signal MS value by the window
        ref_mean_square = np.sum(reference_seg**2) * halfsum2

        proc_mean_squared = np.sum(processed_seg**2) * halfsum2
        correlation = correlate(reference_seg, processed_seg, "full")
        correlation = correlation[
            int(len(reference_seg) - 1 - maxlag) : int(maxlag + len(reference_seg))
        ]
        unbiased_cross_correlation = np.max(np.abs(correlation * half_corr))
        if (ref_mean_square > small) and (proc_mean_squared > small):
            # Normalize cross-covariance
            signal_cross_covariance[k, 0] = unbiased_cross_correlation / np.sqrt(
                ref_mean_square * proc_mean_squared
            )
        else:
            signal_cross_covariance[k, 0] = 0.0

        # Save the reference MS level
        reference_mean_square[k, 0] = ref_mean_square
        processed_mean_square[k, 0] = proc_mean_squared

        # Loop over the remaining full segments, 50% overlap
        for n in range(1, nseg - 1):
            nstart = nstart + nhalf
            nstop = nstart + nwin
            reference_seg = x[nstart:nstop] * window  # Window the reference
            processed_seg = y[nstart:nstop] * window  # Window the processed signal
            reference_seg = reference_seg - np.mean(reference_seg)  # Make 0-mean
            processed_seg = processed_seg - np.mean(processed_seg)

            # Normalize signal MS value by the window
            ref_mean_square = np.sum(reference_seg**2) * win_sum2
            proc_mean_squared = np.sum(processed_seg**2) * win_sum2
            correlation = correlate(reference_seg, processed_seg, "full")
            correlation = correlation[
                int(len(reference_seg) - 1 - maxlag) : int(maxlag + len(reference_seg))
            ]
            unbiased_cross_correlation = np.max(np.abs(correlation * win_corr))
            if (ref_mean_square > small) and (proc_mean_squared > small):
                # Normalize cross-covariance
                signal_cross_covariance[k, n] = unbiased_cross_correlation / np.sqrt(
                    ref_mean_square * proc_mean_squared
                )
            else:
                signal_cross_covariance[k, n] = 0.0

            reference_mean_square[k, n] = ref_mean_square
            processed_mean_square[k, n] = proc_mean_squared

        # The last (half) windowed segment
        nstart = nstart + nhalf
        nstop = nstart + nhalf
        reference_seg = x[nstart:nstop] * window[0:nhalf]  # Window the reference
        processed_seg = y[nstart:nstop] * window[0:nhalf]  # Window the processed signal
        reference_seg = reference_seg - np.mean(reference_seg)  # Make 0-mean
        processed_seg = processed_seg - np.mean(processed_seg)
        # Normalize signal MS value by the window
        ref_mean_square = np.sum(reference_seg**2) * halfsum2
        proc_mean_squared = np.sum(processed_seg**2) * halfsum2

        correlation = np.correlate(reference_seg, processed_seg, "full")
        correlation = correlation[
            int(len(reference_seg) - 1 - maxlag) : int(maxlag + len(reference_seg))
        ]

        unbiased_cross_correlation = np.max(np.abs(correlation * half_corr))
        if (ref_mean_square > small) and (proc_mean_squared > small):
            # Normalized cross-covariance
            signal_cross_covariance[k, nseg - 1] = unbiased_cross_correlation / np.sqrt(
                ref_mean_square * proc_mean_squared
            )
        else:
            signal_cross_covariance[k, nseg - 1] = 0.0

        # Save the reference and processed MS level
        reference_mean_square[k, nseg - 1] = ref_mean_square
        processed_mean_square[k, nseg - 1] = proc_mean_squared

    # Limit the cross-covariance to lie between 0 and 1
    signal_cross_covariance = np.clip(signal_cross_covariance, 0, 1)

    # Adjust the BM magnitude to correspond to the envelope in dB SL
    reference_mean_square *= 2.0
    processed_mean_square *= 2.0

    return signal_cross_covariance, reference_mean_square, processed_mean_square


def ave_covary2(
    signal_cross_covariance: np.ndarray,
    reference_signal_mean_square: np.ndarray,
    threshold_db: float,
    lp_filter_order: ndarray | None = None,
    freq_cutoff: ndarray | None = None,
) -> tuple[float, ndarray]:
    """
    Compute the average cross-covariance between the reference and processed
    signals in each auditory band.

    The silent time-frequency tiles are removed from consideration. The
    cross-covariance is computed for each segment in each frequency band. The
    values are weighted by 1 for inclusion or 0 if the tile is below
    threshold. The sum of the covariance values across time and frequency are
    then divided by the total number of tiles above threshold. The calculation
    is a modification of Tan et al.[1]_ . The cross-covariance is also output
    with a frequency weighting that reflects the loss of IHC synchronization at high
    frequencies Johnson[2]_.

    Arguments:
        signal_cross_covariance (np.array): [nchan,nseg] of cross-covariance values
        reference_signal_mean_square (np.array): [nchan,nseg] of reference signal MS
            values
        threshold_db (): threshold in dB SL to include segment ave over freq in
            average
        lp_filter (list): LP filter order
        freq_cutoff (list): Cutoff frequencies in Hz

    Returns:
        average_covariance (): cross-covariance in segments averaged over time and
            frequency
        ihc_sync_covariance (): cross-covariance array, 6 different weightings for loss
            of IHC synchronization at high frequencies:
              LP Filter Order     Cutoff Freq, kHz
                1              1.5
                3              2.0
                5              2.5, 3.0, 3.5, 4.0

    References:

    .. [1] Tan CT, Moore, BCJ, Zacharov N, Mattila VV (2004) Predicting the Perceived
           Quality of Nonlinearly Distorted Music and Speech Signals. J Audio Eng Soc
           52(9):900-914. Available at.
           <http://www.aes.org/e-lib/browse.cfm?elib=13013>.

    .. [2] Johnson DH (1980) The relationship between spike rate and synchrony in
           responses of auditory‐nerve fibers to single tones J Acoustical Soc of Am
           68:1115 Available at.
           <https://doi.org/10.1121/1.384982>

    Updates:
        James M. Kates, 28 August 2012.
        Adjusted for BM vibration in dB SL, 30 October 2012.
        Threshold for including time-freq tile modified, 30 January 2013.
        Version for different sync loss, 15 February 2013.
        Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Array dimensions
    n_channels = signal_cross_covariance.shape[0]

    # Initialize the LP filter for loss of IHC synchronization
    # Center frequencies in Hz on an ERB scale
    _center_freq = center_frequency(n_channels)
    # Default LP filter order
    if lp_filter_order is None:
        lp_filter_order = np.array([1, 3, 5, 5, 5, 5])
    # Default cutoff frequencies in Hz
    if freq_cutoff is None:
        freq_cutoff = 1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    fc2p = (
        np.atleast_2d(freq_cutoff ** (2 * lp_filter_order)).repeat(n_channels, axis=0).T
    )
    freq2p = _center_freq ** (
        2 * np.atleast_2d(lp_filter_order).repeat(n_channels, axis=0).T
    )
    fsync = np.sqrt(fc2p / (fc2p + freq2p))

    # Find the segments that lie sufficiently above the threshold.
    # Convert squared amplitude to dB envelope
    signal_rms = np.sqrt(reference_signal_mean_square)
    # Linear amplitude (specific loudness)
    signal_linear_amplitude = 10 ** (signal_rms / 20)
    # Intensity averaged over frequency bands
    reference_mean = np.sum(signal_linear_amplitude, 0) / n_channels
    # Convert back to dB (loudness in phons)
    reference_mean = 20 * np.log10(reference_mean)
    # Identify those segments above threshold
    index = np.argwhere(reference_mean > threshold_db).T
    if index.size != 1:
        index = index.squeeze()
    nseg = index.shape[0]  # Number of segments above threshold

    # Exit if not enough segments above zero
    if nseg <= 1:
        logger.warning(
            "Function AveCovary2: Ave signal below threshold, outputs set to 0."
        )
        average_covariance = 0
        # syncov = 0
        ihc_sync_covariance = np.zeros(6)
        return average_covariance, ihc_sync_covariance

    # Remove the silent segments
    signal_cross_covariance = signal_cross_covariance[:, index]
    signal_rms = signal_rms[:, index]

    # Compute the time-frequency weights. The weight=1 if a segment in a
    # frequency band is above threshold, and weight=0 if below threshold.
    weight = np.zeros((n_channels, nseg))  # No IHC synchronization roll-off
    weight[signal_rms > threshold_db] = 1

    # The wsync tensor should be constructed as follows:
    #
    # wsync = np.zeros((6, n_channels, nseg))
    # for k in range(n_channels):
    #    for n in range(nseg):
    #        # Thresh in dB SL for including time-freq tile
    #        if signal_rms[k, n] > threshold_db:
    #            wsync[:, k, n] = fsync[:, k]
    #
    # This can be written is an efficient vectorsized form as follows:
    wsync = np.zeros((6, n_channels, nseg))
    mask = signal_rms > threshold_db
    fsync3d = np.repeat(fsync[..., None], nseg, axis=2)
    wsync[:, mask] = fsync3d[:, mask]

    # Sum the weighted covariance values
    # Sum of weighted time-freq tiles
    csum = np.sum(np.sum(weight * signal_cross_covariance))

    wsum = np.sum(np.sum(weight))  # Total number of tiles above threshold

    tiles_above_threshold = np.zeros(6)

    # Sum of weighted time-freq tiles
    sum_weighted_time_freq = np.sum(wsync * signal_cross_covariance, axis=(1, 2))

    tiles_above_threshold = np.sum(wsync, axis=(1, 2))

    # Exit if not enough segments above zero
    if wsum < 1:
        average_covariance = 0
        logger.warning(
            "Function AveCovary2: Signal tiles below threshold, outputs set to 0."
        )
    else:
        average_covariance = csum / wsum
    ihc_sync_covariance = sum_weighted_time_freq / tiles_above_threshold

    return average_covariance, ihc_sync_covariance
