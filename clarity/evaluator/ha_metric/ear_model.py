"""Ear model for hearing aid HASPI, HASQI, HAAQI metrics."""
from __future__ import annotations

# pylint: disable=import-error
import logging
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from scipy.signal import correlate, group_delay, lfilter, resample_poly

from clarity.enhancer.nalr import NALR
from clarity.evaluator.ha_metric.gammatone_filter import GammatoneFilter

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class EarModel:
    """Ear model for hearing aid metrics.

    Class implements a cochlear model that includes the middle ear,
    auditory filter bank, Outer Hair Cell (OHC) dynamic-range compression,
    and Inner Hair Cell (IHC) attenuation.

    """

    # Basilar Membrane filter coefficients
    COMPRESS_BASILAR_MEMBRANE_COEFS = {
        "24000": {
            "b": [0.09510798340249643, 0.09510798340249643],
            "a": [1.0, -0.8097840331950071],
        }
    }
    # Middle ear filter coefficients
    MIDDLE_EAR_COEF = {
        "24000": {
            "butterworth_low_pass": [0.4341737512063021, 0.4341737512063021],
            "low_pass": [1.0, -0.13165249758739583],
            "butterworth_high_pass": [
                0.9372603902698923,
                -1.8745207805397845,
                0.9372603902698923,
            ],
            "high_pass": [1.0, -1.8705806407352794, 0.8784609203442912],
        }
    }
    # Resample filter coefficients
    RESAMPLE_COEFS = {
        "22050": {
            "a": [
                1.0,
                6.563229198721187,
                18.505433817865256,
                29.05506150301662,
                27.433674675654423,
                15.576261643609874,
                4.923968595144289,
                0.6685242529240554,
            ],
            "b": [
                0.8176333242499695,
                5.694267965735418,
                17.024770717918447,
                28.32640483556404,
                28.32640483556404,
                17.024770717918447,
                5.694267965735419,
                0.8176333242499697,
            ],
        },
        "24000": {
            "a": [
                1.0,
                5.657986938256279,
                14.00815896651005,
                19.634707135261287,
                16.803741671162324,
                8.771318394792921,
                2.5835900814553923,
                0.3310596846351593,
            ],
            "b": [
                0.5753778624919913,
                3.8728648973844546,
                11.32098778566558,
                18.626050890494696,
                18.626050890494696,
                11.320987785665578,
                3.872864897384454,
                0.5753778624919911,
            ],
        },
        "44100": {
            "a": [
                1.0,
                -0.07081207237077872,
                1.2647594875422048,
                0.2132405823253818,
                0.4820212559269799,
                0.13421541556794442,
                0.06248563152819375,
                0.010693174482029118,
            ],
            "b": [
                0.10526806659004136,
                0.2673828276910548,
                0.5089236138475818,
                0.6667272293722993,
                0.6667272293722992,
                0.5089236138475817,
                0.2673828276910549,
                0.1052680665900414,
            ],
        },
    }

    def __init__(
        self,
        equalisation: int,
        target_freq: float = 24000.0,
        nchan: int = 32,
        m_delay: int = 1,
        small: float = 1e-30,
        ear_q: float = 9.26449,
    ):
        """
        Constructor takes the reference and processed signals that are to be
        compared. The reference is at the reference intensity (e.g. 65 dB SPL
        or with NAL-R amplification) and has no other processing. The processed
        signal is the hearing-aid output, and is assumed to have the same or
        greater group delay compared to the reference.

        Arguments:
        equalisation (int): purpose for the calculation:
             0=intelligibility: reference is normal hearing and must not
               include NAL-R EQ
             1=quality: reference does not include NAL-R EQ
             2=quality: reference already has NAL-R EQ applied
        target_freq (int): sampling rate for resampling the signals, Hz.
            Both, reference and processed signals are resampled to this rate.
            Default is 24000 Hz.
        nchan (int): auditory frequency bands
        m_delay (int): Compensate for the gammatone group delay.
        small (float): small number to avoid division by zero
        ear_q (float): quality factor of the gammatone filter

        """
        self.equalisation = equalisation
        self.target_freq = target_freq
        self.nchan = nchan
        self.m_delay = m_delay
        self.small = small
        self.ear_q = ear_q

    def compute(
        self,
        reference: ndarray,
        reference_freq: float,
        processed: ndarray,
        processed_freq: float,
        hearing_loss: ndarray,
        level1: float,
        shift: float | None = None,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, float]:
        """
        Apply the model to the signals.

        The method returns the envelopes of the signals after OHC compression
        and IHC loss attenuation.

        Arguments:
            reference (ndarray): reference signal: should be adjusted to 65 dB SPL
            (equalisation=0 or 1) or to 65 dB SPL plus NAL-R gain (equalisation=2)
            reference_freq (int): sampling rate for the reference signal, Hz
            processed (ndarray): processed signal (e.g. hearing-aid output) includes
                HA gain
            processed_freq (int): sampling rate for the processed signal, Hz
            hearing_loss (ndarray): audiogram giving the hearing loss in dB at 6
                audiometric frequencies: [250, 500, 1000, 2000, 4000, 6000] Hz
            level1 (float): level of the reference signal in dB SPL
            shift (float): optional frequency shift of the filter bank specified
                as a fractional shift in distance along the BM. A positive shift is an
                increase in frequency (basal shift), and negative is a decrease in
                frequency (apical shift). The total length of the BM is norm to 1.
                The frequency-to-distance map is from D.D. Greenwood[3].

        Returns:
            reference_db (ndarray): envelope for the reference in each band
            reference_basilar_membrane (): BM motion for the reference in each band
            processed_db (ndarray): envelope for the processed signal in each band
            processed_basilar_membrane (): BM motion for the processed signal in
                each band
            reference_sl (ndarray): compressed RMS average reference in
                each band converted to dB SL
            processed_sl (ndarray): compressed RMS average output in each
                band converted to dB SL
            freq_sample (float): sampling rate in Hz for the model outputs
        """
        # Center frequencies on an ERB scale
        _center_freq = self.center_frequencies()

        # Cochlear model parameters for the processed signal
        (
            attn_ohc_y,
            bandwidth_min_y,
            low_knee_y,
            compression_ratio_y,
            attn_ihc_y,
        ) = self.loss_parameters(hearing_loss, _center_freq)

        # The cochlear model parameters for the reference are the same as for
        # the hearing loss if calculating quality, but are for normal hearing
        # if calculating intelligibility.
        attn_ohc_x = attn_ohc_y.copy()
        bandwidth_min_x = bandwidth_min_y.copy()
        low_knee_x = low_knee_y.copy()
        compression_ratio_x = compression_ratio_y.copy()
        attn_ihc_x = attn_ihc_y.copy()

        if self.equalisation == 0:
            [
                attn_ohc_x,
                bandwidth_min_x,
                low_knee_x,
                compression_ratio_x,
                attn_ihc_x,
            ] = self.loss_parameters(np.zeros(len(hearing_loss)), _center_freq)

        # Compute center frequencies for the control
        _center_freq_control = self.center_frequencies(shift=shift)
        # Maximum BW for the control
        _, bandwidth_1, _, _, _ = self.loss_parameters(
            np.full(6, 100), _center_freq_control
        )

        reference_24hz, _ = self.resample(reference, reference_freq, self.target_freq)
        processed_24hz, freq_sample = self.resample(
            processed, processed_freq, self.target_freq
        )

        # Check file sizes
        min_signal_length = min(len(reference_24hz), len(processed_24hz))
        reference_24hz = reference_24hz[:min_signal_length]
        processed_24hz = processed_24hz[:min_signal_length]

        reference_24hz, processed_24hz = self.input_align(
            reference_24hz, processed_24hz, freq_sample
        )
        nsamp = len(reference_24hz)

        # For HASQI, here add NAL-R equalization if the quality reference doesn't
        # already have it.
        if self.equalisation == 1:
            nfir = 140  # Length in samples of the FIR NAL-R EQ filter (24-kHz rate)
            enhancer = NALR(nfir, freq_sample)
            aud = np.array([250, 500, 1000, 2000, 4000, 6000])
            nalr_fir, _ = enhancer.build(hearing_loss, aud)
            reference_24hz = enhancer.apply(nalr_fir, reference_24hz)
            reference_24hz = reference_24hz[nfir : nfir + nsamp]

        # Cochlear model
        # Middle ear
        reference_mid = self.middle_ear(reference_24hz, freq_sample)
        processed_mid = self.middle_ear(processed_24hz, freq_sample)

        # Check the lengths of the two signals and trim to shortest
        min_number_sample = min(len(reference_mid), len(processed_mid))
        reference_mid = reference_mid[:min_number_sample]
        processed_mid = processed_mid[:min_number_sample]

        # Initialize storage
        # Reference and processed envelopes and BM motion
        reference_db = np.zeros((self.nchan, nsamp))
        processed_db = np.zeros((self.nchan, nsamp))

        # Reference and processed average spectral values
        reference_average = np.zeros(self.nchan)
        processed_average = np.zeros(self.nchan)
        reference_control_average = np.zeros(self.nchan)
        processed_control_average = np.zeros(self.nchan)

        # Filter bandwidths adjusted for intensity
        reference_bandwidth = np.zeros(self.nchan)
        processed_bandwidth = np.zeros(self.nchan)

        reference_b = np.zeros((self.nchan, nsamp))
        processed_b = np.zeros((self.nchan, nsamp))

        gammatone_filter = GammatoneFilter(freq_sample=freq_sample)

        for n in range(self.nchan):
            reference_control, _ = gammatone_filter.compute(
                reference_mid, bandwidth_1[n], _center_freq_control[n]
            )
            processed_control, _ = gammatone_filter.compute(
                processed_mid, bandwidth_1[n], _center_freq_control[n]
            )

            # Adjust the auditory filter bandwidths for the average signal level
            reference_bandwidth[n] = self.bandwidth_adjust(
                reference_control, bandwidth_min_x[n], bandwidth_1[n], level1
            )
            processed_bandwidth[n] = self.bandwidth_adjust(
                processed_control, bandwidth_min_y[n], bandwidth_1[n], level1
            )

            # Envelopes and BM motion of the reference and processed signals
            xenv, xbm = gammatone_filter.compute(
                reference_mid, reference_bandwidth[n], _center_freq[n]
            )
            yenv, ybm = gammatone_filter.compute(
                processed_mid, processed_bandwidth[n], _center_freq[n]
            )

            # RMS levels of the ref and output envelopes for linear metric
            reference_average[n] = np.sqrt(np.mean(xenv**2))
            processed_average[n] = np.sqrt(np.mean(yenv**2))
            reference_control_average[n] = np.sqrt(np.mean(reference_control**2))
            processed_control_average[n] = np.sqrt(np.mean(processed_control**2))

            # Cochlear compression for the signal envelopes and BM motion
            (
                reference_cochlear_compression,
                reference_b[n],
            ) = self.env_compress_basilar_membrane(
                xenv,
                xbm,
                reference_control,
                attn_ohc_x[n],
                low_knee_x[n],
                compression_ratio_x[n],
                fsamp=freq_sample,
                level1=level1,
            )
            (
                processed_cochlear_compression,
                processed_b[n],
            ) = self.env_compress_basilar_membrane(
                yenv,
                ybm,
                processed_control,
                attn_ohc_y[n],
                low_knee_y[n],
                compression_ratio_y[n],
                fsamp=freq_sample,
                level1=level1,
            )

            # Correct for the delay between the reference and output
            # Align processed envelope to reference
            processed_cochlear_compression = self.envelope_align(
                reference_cochlear_compression,
                processed_cochlear_compression,
                freq_sample,
            )
            # Align processed BM motion to reference
            processed_b[n] = self.envelope_align(
                reference_b[n], processed_b[n], freq_sample
            )

            # Convert the compressed envelopes and BM vibration envelopes to dB SPL
            reference_cochlear_compression, reference_b[n] = self.envelope_sl(
                reference_cochlear_compression, reference_b[n], attn_ihc_x[n], level1
            )
            processed_cochlear_compression, processed_b[n] = self.envelope_sl(
                processed_cochlear_compression, processed_b[n], attn_ihc_y[n], level1
            )

            # Apply the IHC rapid and short-term adaptation
            delta = 2  # Amount of overshoot
            reference_db[n], reference_b[n] = self.inner_hair_cell_adaptation(
                reference_cochlear_compression, reference_b[n], delta, freq_sample
            )
            processed_db[n], processed_b[n] = self.inner_hair_cell_adaptation(
                processed_cochlear_compression, processed_b[n], delta, freq_sample
            )

        # Additive noise level to give the auditory threshold
        ihc_threshold = -10  # Additive noise level, dB re: auditory threshold
        reference_basilar_membrane = self.basilar_membrane_add_noise(
            reference_b, ihc_threshold, level1
        )
        processed_basilar_membrane = self.basilar_membrane_add_noise(
            processed_b, ihc_threshold, level1
        )

        # Correct for the gammatone filterbank interchannel group delay.
        if self.m_delay > 0:
            reference_db = self.group_delay_compensate(
                reference_db, reference_bandwidth, _center_freq, freq_sample
            )
            processed_db = self.group_delay_compensate(
                processed_db, reference_bandwidth, _center_freq, freq_sample
            )
            reference_basilar_membrane = self.group_delay_compensate(
                reference_basilar_membrane,
                reference_bandwidth,
                _center_freq,
                freq_sample,
            )
            processed_basilar_membrane = self.group_delay_compensate(
                processed_basilar_membrane,
                reference_bandwidth,
                _center_freq,
                freq_sample,
            )

        # Convert average gammatone outputs to dB SPL
        reference_sl = self.convert_rms_to_sl(
            reference_average,
            reference_control_average,
            attn_ohc_x,
            low_knee_x,
            compression_ratio_x,
            attn_ihc_x,
            level1,
        )
        processed_sl = self.convert_rms_to_sl(
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

    def center_frequencies(
        self,
        low_freq: int = 80,
        high_freq: int = 8000,
        min_bw: float = 24.7,
        shift: float | None = None,
    ):
        """
        Compute the Equivalent Rectangular Bandwidth_[1] frequency spacing for the
        gammatone filterbank. The equation comes from Malcolm Slaney[1].

        Arguments:
            low_freq (int): Low Frequency level.
            high_freq (int): High Frequency level.
            min_bw (float):
            shift (float): Basal shift of the basilar membrane length


        Returns:
            center_freq (torch.FloatTensor): Center frequencies of the
                gammatone filterbank.

        References:
        [1] Slaney M (1993) An Efficient Implemtnation of the Patterson-
               Holdsworth Auditory Filter Bank. Available at:
               <https://asset-pdf.scinapse.io/prod/396690109/396690109.pdf>.
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
            low_freq = A * (np.power(10, (a * x_low)) - k)
            high_freq = A * (np.power(10, (a * x_high)) - k)

        # All of the following expressions are derived in Apple TR #35,
        # "An Efficient Implementation of the Patterson-Holdsworth Cochlear
        # Filter Bank" by Malcolm Slaney.
        # https://engineering.purdue.edu/~malcolm/apple/tr35/PattersonsEar.pdf
        _center_freq = -(self.ear_q * min_bw) + np.exp(
            np.arange(1, self.nchan)
            * (
                -np.log(high_freq + self.ear_q * min_bw)
                + np.log(low_freq + self.ear_q * min_bw)
            )
            / (self.nchan - 1)
        ) * (high_freq + self.ear_q * min_bw)

        # Last center frequency is set to highFreq
        _center_freq = np.concatenate((np.array([high_freq]), _center_freq))
        _center_freq = np.flip(_center_freq)
        return _center_freq

    @staticmethod
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
            hearing_loss (ndarray): hearing loss at the 6 audiometric frequencies
            center_freq (ndarray): array containing the center frequencies of the
                gammatone filters arranged from low to high
            audiometric_freq (ndarray): array containing the audiometric frequencies

        Returns:
            attenuated_ohc (ndarray): attenuation in dB for the OHC gammatone filters
            bandwidth (ndarray): OHC filter bandwidth expressed in terms of normal
            low_knee (ndarray): Lower kneepoint for the low-level linear amplification
            compression_ratio (ndarray): Ranges from 1.4:1 at 150 Hz to 3.5:1
                at 8 kHz for normal hearing. Reduced in proportion to the OHC
                loss to 1:1.
            attenuated_ihc (ndarray): attenuation in dB for the input to the IHC synapse
        """
        # Audiometric frequencies in Hz
        if audiometric_freq is None:
            audiometric_freq = np.array([250, 500, 1000, 2000, 4000, 6000])

        # Interpolation to give the loss at the gammatone center frequencies
        # Use linear interpolation in dB. The interpolation assumes that
        # cfreq[0] < aud[0] and cfreq[nfilt -1] > aud[5]
        nfilt = len(center_freq)
        f_v = np.zeros(len(audiometric_freq) + 2)
        f_v[1:-1] = audiometric_freq
        f_v[0] = audiometric_freq[0]
        f_v[-1] = audiometric_freq[-1]

        fv_interp = np.zeros(len(audiometric_freq) + 2)
        fv_interp[1:-1] = hearing_loss
        fv_interp[0] = hearing_loss[0]
        fv_interp[-1] = hearing_loss[-1]

        # Interpolated gain in dB
        loss = np.interp(
            center_freq,
            f_v,
            fv_interp,
        )

        # Make sure there are no negative losses
        loss = np.maximum(loss, 0)

        # Compression ratio changes linearly with ERB rate from 1.25:1 in the 80-Hz
        # frequency band to 3.5:1 in the 8-kHz frequency band
        compression_ratio = 1.25 + 2.25 * np.arange(nfilt) / (nfilt - 1)

        # Maximum OHC sensitivity loss depends on the compression ratio. The compression
        # I/O curves assume linear below 30 and above 100 dB SPL in normal ears.

        # HC loss that results in 1:1 compression
        max_ohc = 70 * (1 - (1 / compression_ratio))

        # Loss threshold for adjusting the OHC parameters
        theoretical_ohc = 1.25 * max_ohc

        # Apportion the loss in dB to the outer and inner hair cells based
        # on the data of Moore et al (1999), JASA 106, 2761-2778.

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
        bandwidth = (
            bandwidth + (attenuated_ohc / 50.0) + 2.0 * (attenuated_ohc / 50.0) ** 6
        )

        # Compute the compression lower kneepoint and compression ratio
        low_knee = attenuated_ohc + 30

        # Output level for an input of 100 dB SPL
        upamp = 30 + (70 / compression_ratio)

        # OHC loss Compression ratio
        compression_ratio = (100 - low_knee) / (upamp + attenuated_ohc - low_knee)

        return attenuated_ohc, bandwidth, low_knee, compression_ratio, attenuated_ihc

    def resample(
        self,
        reference_signal: ndarray,
        reference_sample_rate: float,
        target_sample_rate: float = 24000.0,
    ) -> tuple[ndarray, float]:
        """
        Resample the input signal at `target_sample_rate`.
        The input sampling rate is rounded to the nearest kHz
        to compute the sampling rate conversion ratio.

        Arguments:
        reference_signal (np.ndarray): input signal
        reference_sample_rate (int): sampling rate for the input in Hz
        freq_sample_hz (int): Frequency sample in Hz

        Returns:
        reference_signal_24         signal resampled at kHz (default 24Khz)
        freq_sample_hz     output sampling rate in Hz

        Updates
        James M. Kates, 20 June 2011.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """

        # Sampling rate information
        target_freq_khz = np.round(target_sample_rate / 1000)
        # output rate to nearest kHz
        reference_freq_khz = np.round(reference_sample_rate / 1000)

        # Resample the signal
        if reference_freq_khz == target_freq_khz:
            # No resampling performed if the rates match
            return reference_signal, target_sample_rate

        # Resample for the input to output sample rate
        resample_signal = resample_poly(
            reference_signal, target_freq_khz, reference_freq_khz
        )

        if reference_freq_khz < target_freq_khz:
            # Match the RMS level of the resampled signal to that of the input
            reference_rms = np.sqrt(np.mean(reference_signal**2))
            resample_rms = np.sqrt(np.mean(resample_signal**2))

        else:
            # Reduce the input signal bandwidth to 21 kHz (-10.5 to +10.5 kHz)
            # The power equalization is designed to match the signal intensities
            # over the frequency range spanned by the gammatone filter bank.
            # Chebyshev Type 2 LP
            coef_ref = self.RESAMPLE_COEFS[str(int(reference_sample_rate))]
            coef_target = self.RESAMPLE_COEFS[str(int(target_sample_rate))]

            reference_filter = lfilter(
                coef_ref["b"],
                coef_ref["a"],
                reference_signal,
                axis=0,
            )
            target_filter = lfilter(
                coef_target["b"], coef_target["a"], resample_signal, axis=0
            )

            reference_rms = np.sqrt(np.mean(reference_filter**2))
            resample_rms = np.sqrt(np.mean(target_filter**2))

        resample_signal = (reference_rms / resample_rms) * resample_signal

        return resample_signal, target_sample_rate

    @staticmethod
    def input_align(
        reference: ndarray, processed: ndarray, fsamp: float = 24000.0
    ) -> tuple[ndarray, ndarray]:
        """
        Approximate temporal alignment of the reference and processed output
        signals. Leading and trailing zeros are then pruned.

        The method assumes that the two sequences have the same sampling rate.

        Arguments:
            reference (np.ndarray): input reference sequence
            processed (np.ndarray): hearing-aid output sequence
            fsamp (float): Cochlear model input sampling rate in Hz

        Returns:
            reference (np.ndarray): pruned and shifted reference
            processed (np.ndarray): pruned and shifted hearing-aid output
        """

        # Match the length of the processed output to the reference for the purposes
        # of computing the cross-covariance
        reference_n = len(reference)
        processed_n = len(processed)
        min_sample_length = min(reference_n, processed_n)

        # Determine the delay of the output relative to the reference
        # Matlab code uses xcov thus the subtraction of mean
        reference_processed_correlation = correlate(
            reference[:min_sample_length] - np.mean(reference[:min_sample_length]),
            processed[:min_sample_length] - np.mean(processed[:min_sample_length]),
            "full",
        )
        index = np.argmax(np.abs(reference_processed_correlation))
        delay = min_sample_length - index - 1

        # Back up 2 msec to allow for dispersion
        delay = np.rint(delay - 2 * fsamp / 1000.0).astype(int)  # Back up 2 ms

        # Align the output with the reference allowing for the dispersion
        processed_out = np.zeros(processed_n)
        if delay > 0:
            # Output delayed relative to the reference
            processed_out[: processed_n - delay] = processed[delay:processed_n]
        else:
            processed_out[abs(delay) - 1 :] = processed[: processed_n - abs(delay)]

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
            processed_out[
                reference_n_above_threshold : reference_n_below_threshold + 1
            ],
        )

    def middle_ear(self, reference: ndarray, freq_sample: float) -> ndarray:
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
            xout (ndarray): filtered output
        """

        # Design the 1-pole Butterworth LP using the bilinear transformation
        coef = self.MIDDLE_EAR_COEF[str(int(freq_sample))]

        # LP filter the input
        y = lfilter(coef["butterworth_low_pass"], coef["low_pass"], reference)
        # HP filter the signal
        return lfilter(coef["butterworth_high_pass"], coef["high_pass"], y)

    @staticmethod
    def bandwidth_adjust(
        control: ndarray,
        bandwidth_min: float,
        bandwidth_max: float,
        level1: float,
    ) -> float:
        """
        Compute the increase in auditory filter bandwidth in
        response to high signal levels.

        Arguments:
            control (): envelope output in the control filter band
            bandwidth_min (): auditory filter bandwidth computed for
                the loss (or NH)
            bandwidth_max (): auditory filter bandwidth at
                maximum OHC damage
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
        return bandwidth_min + ((control_db - 50) / 50) * (
            bandwidth_max - bandwidth_min
        )

    def env_compress_basilar_membrane(
        self,
        envsig: ndarray,
        bm: ndarray,  # pylint: disable=invalid-name
        control: ndarray,
        attn_ohc: float,
        threshold_low: float,
        compression_ratio: float,
        fsamp: float,
        level1: float,
        threshold_high: int = 100,
    ) -> tuple[ndarray, ndarray]:
        """
        Compute the cochlear compression in one auditory filter band.
        The gain is linear below the lower threshold, compressive with
        a compression ratio of CR:1 between the lower and upper thresholds,
        and reverts to linear above the upper threshold. The compressor
        assumes that auditory threshold is 0 dB SPL.

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
        logenv = np.maximum(control, self.small)
        logenv = level1 + 20 * np.log10(logenv)
        logenv = np.minimum(
            logenv, threshold_high
        )  # Clip signal levels above the upper threshold
        logenv = np.maximum(logenv, threshold_low)  # Clip signal at the lower threshold

        # Compute the compression gain in dB
        gain = -attn_ohc - (logenv - threshold_low) * (1 - (1 / compression_ratio))

        # Convert the gain to linear and apply a LP filter to give a 0.2 ms delay
        gain = 10 ** (gain / 20)
        coefs = self.COMPRESS_BASILAR_MEMBRANE_COEFS[str(int(fsamp))]

        gain = lfilter(coefs["b"], coefs["a"], gain)

        # Apply the gain to the signals
        compressed_signal = gain * envsig
        compressed_basilar_membrane = gain * bm

        return compressed_signal, compressed_basilar_membrane

    @staticmethod
    def envelope_align(
        reference: ndarray,
        output: ndarray,
        freq_sample: float,
        corr_range: int = 100,
    ) -> ndarray:
        """
        Align the envelope of the processed signal to that of the reference signal.

        Arguments:
            reference (): envelope or BM motion of the reference signal
            output (): envelope or BM motion of the output signal
            freq_sample (float): Frequency sample rate in Hz
            corr_range (int): range in msec for the correlation

        Returns:
            y (): shifted output envelope to match the input

        Updates:
        James M. Kates, 28 October 2011.
        Absolute value of the cross-correlation peak removed, 22 June 2012.
        Cross-correlation range reduced, 13 August 2013.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """

        # The MATLAB code limits the range of lags to search (to 100 ms)
        # to save computation time - no such option exists in numpy,
        # but the code below limits the delay to the same range as in
        # Matlab, for consistent results
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
        self,
        reference: ndarray,
        basilar_membrane: ndarray,
        attenuated_ihc: float,
        level1: float,
    ) -> tuple[ndarray, ndarray]:
        """
        Convert the compressed envelope returned by cochlear_envcomp to dB SL.

        Arguments:
            reference (): linear envelope after compression
            basilar_membrane (): linear Basilar Membrane vibration after compression
            attenuated_ihc (): IHC attenuation at the input to the synapse
            level1 (): level in dB SPL corresponding to 1 RMS

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
        _reference = level1 - attenuated_ihc + 20 * np.log10(reference + self.small)
        _reference = np.maximum(_reference, 0)

        # Convert the linear BM motion to have a dB SL envelope
        gain = (_reference + self.small) / (reference + self.small)
        _basilar_membrane = gain * basilar_membrane

        return _reference, _basilar_membrane

    # Method needs to be static in order to be used with numba
    @staticmethod
    @njit
    def inner_hair_cell_adaptation(
        reference_db: ndarray,
        reference_basilar_membrane: ndarray,
        delta: float,
        freq_sample: float,
        small: float = 1e-30,
    ) -> tuple[ndarray, ndarray]:
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
            reference_basilar_membrane (ndarray): basilar membrane vibration with
                OHC compression but no IHC attenuation
            delta (float): overshoot factor = delta x steady-state
            freq_sample (int): sampling rate in Hz
            small (float): small number to avoid log of zero

        Returns:
            output_db (ndarray): envelope in dB SL with IHC adaptation
            output_basilar_membrane (ndarray): Basilar Membrane multiplied
                by the IHC adaptation gain function

        Updates:
        James M. Kates, 1 October 2012.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """
        # Test the amount of overshoot
        delta = np.maximum(delta, 1.0001)

        # Initialize adaptation time constants
        tau1 = 2e-3  # Rapid adaptation in seconds
        tau2 = 60e-3  # Short-term adaptation in seconds

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
        gain = np.ones_like(reference_db)
        output_db = np.zeros_like(reference_db)
        v_1 = 0
        v_2 = 0

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

    @staticmethod
    def basilar_membrane_add_noise(
        reference: ndarray, threshold: int, level1: float
    ) -> ndarray:
        """
        Apply the IHC attenuation to the BM motion and to add a
        low-level Gaussian noise to give the auditory threshold.

        Arguments:
            reference (ndarray): BM motion to be attenuated
            threshold (int): additive noise level in dB re:auditory threshold
            level1 (float): an input having RMS=1 corresponds to Level1 dB SPL

        Returns:
            Attenuated signal with threshold noise added

        Updates:
            James M. Kates, 19 June 2012.
            Just additive noise, 2 Oct 2012.
            Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """
        # Linear gain for the noise
        gain = 10 ** ((threshold - level1) / 20)

        # Gaussian RMS=1, then attenuated
        noise = gain * np.random.standard_normal(reference.shape)
        return reference + noise

    def group_delay_compensate(
        self,
        reference: ndarray,
        bandwidths: ndarray,
        center_freq: ndarray,
        freq_sample: float,
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
        erb = min_bandwidth + (center_freq / self.ear_q)

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
            processed[n] = np.concatenate(
                (np.zeros(correct[n]), ref[: npts - correct[n]])
            )

        return processed

    def convert_rms_to_sl(
        self,
        reference: ndarray,
        control: ndarray,
        attenuated_ohc: ndarray | float,
        threshold_low: ndarray | int,
        compression_ratio: ndarray | int,
        attenuated_ihc: ndarray | float,
        level1: float,
        threshold_high: int = 100,
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

        Returns:
            reference_db (): compressed output in dB above the impaired threshold

        Updates:
            James M. Kates, 6 August 2007.
            Version for two-tone suppression, 29 August 2008.
            Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """

        control_db_spl = np.maximum(control, self.small)
        control_db_spl = level1 + 20 * np.log10(control_db_spl)
        control_db_spl = np.minimum(control_db_spl, threshold_high)
        control_db_spl = np.maximum(control_db_spl, threshold_low)

        # Compute compression gain in dB
        gain = -attenuated_ohc - (control_db_spl - threshold_low) * (
            1 - (1 / compression_ratio)
        )

        # Convert the signal envelope to dB SPL
        control_db_spl = np.maximum(reference, self.small)
        control_db_spl = level1 + 20 * np.log10(control_db_spl)
        control_db_spl = np.maximum(control_db_spl, 0)
        reference_db = control_db_spl + gain - attenuated_ihc
        reference_db = np.maximum(reference_db, 0)

        return reference_db
