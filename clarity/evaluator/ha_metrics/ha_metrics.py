"""Abstract class for hearing aid metrics."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import cheby2, convolve, correlate, lfilter, resample_poly

from clarity.enhancer.nalr import NALR


class ha_metrics:
    """Abstract class for hearing aids metrics"""

    def __init__(
        self,
        reference: np.ndarray,
        reference_freq: int,
        processed: np.ndarray,
        processed_freq: int,
        hearing_loss: np.ndarray,
        equalisation: int,
        level1: int = 65,
    ):
        """Constructor.

        Args:
            reference (np.ndarray): reference signal
            reference_freq (int): sampling rate of reference signal
            processed (np.ndarray): processed signal
            processed_freq (int): sampling rate of processed signal
            hearing_loss (np.ndarray): hearing loss
            level1 (int, optional): level in dB SPL corresponding to RMS=1. Defaults to 65.
        """
        self.reference = reference
        self.reference_freq = reference_freq
        self.processed = processed
        self.processed_freq = processed_freq
        self.hearing_loss = hearing_loss
        self.equalisation = equalisation
        self.level1 = level1

    @lru_cache(maxsize=None)
    def ear_model(self, nchan, shift: float | None = None) -> np.ndarray:
        """
        Method that implements a cochlear model that includes the middle ear,
        auditory filter bank, Outer Hair Cell (OHC) dynamic-range compression,
        and Inner Hair Cell (IHC) attenuation.

        Args:
            nchan (int): number of auditory frequency bands
            shift (float): Basal shift of the basilar membrane length. Defaults to None.

        Returns:
            np.ndarray: ear model
        """

        # OHC and IHC parameters for the hearing loss
        # Auditory filter center frequencies span 80 to 8000 Hz.
        _center_freq = self.center_frequency(nchan)

        # Cochlear model parameters for the processed signal
        (
            attn_ohc_y,
            bandwidth_min_y,
            low_knee_y,
            compression_ratio_y,
            attn_ihc_y,
        ) = self.loss_parameters(self.hearing_loss, _center_freq)

        # The cochlear model parameters for the reference are the same as for the hearing
        # loss if calculating quality, but are for normal hearing if calculating
        # intelligibility.
        if self.equalisation == 0:
            [
                attn_ohc_x,
                bandwidth_min_x,
                low_knee_x,
                compression_ratio_x,
                attn_ihc_x,
            ] = self.loss_parameters(np.zeros(len(self.hearing_loss)), _center_freq)
        else:
            attn_ohc_x = attn_ohc_y.copy()
            bandwidth_min_x = bandwidth_min_y.copy()
            low_knee_x = low_knee_y.copy()
            compression_ratio_x = compression_ratio_y.copy()
            attn_ihc_x = attn_ihc_y.copy()

        # Parameters for the control filter bank
        hl_max = np.full(6, 100)
        # Compute center frequencies for the control
        _center_freq_control = self.center_frequency(nchan, shift)

        # Maximum BW for the control
        _, bandwidth_1, _, _, _ = self.loss_parameters(hl_max, _center_freq_control)

        # Input signal adjustments
        # Convert the signals to 24 kHz sampling rate.
        # Using 24 kHz guarantees that all of the cochlear filters have the same shape
        # independent of the incoming signal sampling rates

        reference_24hz, _ = self.resample_24khz(self.reference, self.reference_freq)
        processed_24hz, freq_sample = self.resample_24khz(
            self.processed, self.processed_freq
        )

        # Bulk broadband signal alignment
        reference_24hz, processed_24hz = self.input_align(
            reference_24hz, processed_24hz
        )
        nsamp = len(reference_24hz)

        # For HASQI, here add NAL-R equalization if the quality reference doesn't
        # already have it.
        if self.equalisation == 1:
            nfir = 140  # Length in samples of the FIR NAL-R EQ filter (24-kHz rate)
            enhancer = NALR(nfir, freq_sample)
            aud = [250, 500, 1000, 2000, 4000, 6000]
            nalr_fir, _ = enhancer.build(self.hearing_loss, aud)
            reference_24hz = convolve(
                reference_24hz, nalr_fir
            )  # Apply the NAL-R filter
            reference_24hz = reference_24hz[nfir : nfir + nsamp]

        return self.hearing_loss

    @staticmethod
    @lru_cache(maxsize=None)
    def center_frequency(
        nchan: int,
        shift: float | None = None,
        low_freq: int = 80,
        high_freq: int = 8000,
        ear_q: float = 9.26449,
        min_bw: float = 24.7,
    ) -> np.ndarray:
        """
        Compute the Equivalent Rectangular Bandwidth_[1] frequency spacing for the
        gammatone filter bank. The equation comes from Malcolm Slaney[2].

        Arguments:
            nchan (int): number of filters in the filter bank
            shift (float | None): optional frequency shift of the filter bank specified as a fractional
                shift in distance along the BM. A positive shift is an increase in frequency
                (basal shift), and negative is a decrease in frequency (apical shift). The
                total length of the BM is normalized to 1. The frequency-to-distance map is
                from D.D. Greenwood[3].
            low_freq (int): Low Frequency level.
            high_freq (int): High Frequency level.
            ear_q (float):
            min_bw (float):

        Returns:
            np.ndarray: center frequencies

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

    @staticmethod
    @lru_cache(maxsize=None)
    def loss_parameters(
        hearing_loss, center_freq: np.ndarray, audiometric_freq: list | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apportion the hearing loss to the outer hair cells (OHC) and the inner
        hair cells (IHC) and to increase the bandwidth of the cochlear filters
        in proportion to the OHC fraction of the total loss.

        Arguments:
            hearing_loss (np.ndarray): hearing loss at the 6 audiometric frequencies
            center_freq (np.ndarray): array containing the center frequencies of the
                gammatone filters arranged from low to high
            audiometric_freq (list | None): list of audiometric frequencies

        Returns:
            attenuated_ohc (np.ndarray): attenuation in dB for the OHC gammatone filters
            bandwidth (np.ndarray): OHC filter bandwidth expressed in terms of normal
            low_knee (np.ndarray): Lower kneepoint for the low-level linear amplification
            compression_ratio (np.ndarray): Ranges from 1.4:1 at 150 Hz to 3.5:1 at 8 kHz for normal
                hearing. Reduced in proportion to the OHC loss to 1:1.
            attenuated_ihc (np.ndarray):	attenuation in dB for the input to the IHC synapse

        Updates:
        James M. Kates, 25 January 2007.
        Version for loss in dB and match of OHC loss to CR, 9 March 2007.
        Low-frequency extent changed to 80 Hz, 27 Oct 2011.
        Lower kneepoint set to 30 dB, 19 June 2012.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """
        # Audiometric frequencies in Hz
        if audiometric_freq is None:
            audiometric_freq = [250, 500, 1000, 2000, 4000, 6000]

        # Interpolation to give the loss at the gammatone center frequencies
        # Use linear interpolation in dB. The interpolation assumes that
        # cfreq[1] < aud[1] and cfreq[nfilt] > aud[6]
        nfilt = len(center_freq)
        f_v = np.insert(
            audiometric_freq,
            [0, len(audiometric_freq)],
            [center_freq[0], center_freq[-1]],
        )

        # Interpolated gain in dB
        loss = np.interp(
            center_freq,
            f_v,
            np.insert(
                hearing_loss,
                [0, len(hearing_loss)],
                [hearing_loss[0], hearing_loss[-1]],
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
        theoretical_ohc = (
            1.25 * max_ohc
        )  # Loss threshold for adjusting the OHC parameters

        # Apportion the loss in dB to the outer and inner hair cells based on the data of
        # Moore et al (1999), JASA 106, 2761-2778.

        # Reduce the CR towards 1:1 in proportion to the OHC loss.
        attenuated_ohc = 0.8 * np.copy(loss)
        attnenuated_ihc = 0.2 * np.copy(loss)

        attenuated_ohc[loss >= theoretical_ohc] = (
            0.8 * theoretical_ohc[loss >= theoretical_ohc]
        )
        attnenuated_ihc[loss >= theoretical_ohc] = 0.2 * theoretical_ohc[
            loss >= theoretical_ohc
        ] + (loss[loss >= theoretical_ohc] - theoretical_ohc[loss >= theoretical_ohc])

        # Adjust the OHC bandwidth in proportion to the OHC loss
        bandwidth = np.ones(nfilt)
        bandwidth = (
            bandwidth + (attenuated_ohc / 50.0) + 2.0 * (attenuated_ohc / 50.0) ** 6
        )

        # Compute the compression lower kneepoint and compression ratio
        low_knee = attenuated_ohc + 30
        upamp = 30 + (70 / compression_ratio)  # Output level for an input of 100 dB SPL

        compression_ratio = (100 - low_knee) / (
            upamp + attenuated_ohc - low_knee
        )  # OHC loss Compression ratio

        return attenuated_ohc, bandwidth, low_knee, compression_ratio, attnenuated_ihc

    @staticmethod
    @lru_cache(maxsize=None)
    def resample_24khz(
        signal: np.ndarray, sample_frequency: int, freq_sample_hz: int = 24000
    ) -> tuple[np.ndarray, int]:
        """
        Resample the input signal at 24 kHz. The input sampling rate is
        rounded to the nearest kHz to compute the sampling rate conversion
        ratio.

        Args:
            signal (np.ndarray): input signal
            sample_frequency (int): sampling rate for the input in Hz
            freq_sample_hz (int): target frequency sample in Hz

        Returns:
            resample_signal (np.ndarray): signal resampled at kHz (default 24Khz)
            freq_sample_hz (int): target frequency sample in Hz

        Updates
        James M. Kates, 20 June 2011.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """

        # Sampling rate information
        sample_rate_target_khz = round(
            freq_sample_hz / 1000
        )  # output rate to nearest kHz
        reference_freq_khz = round(sample_frequency / 1000)

        # Resample the signal
        if reference_freq_khz == sample_rate_target_khz:
            # No resampling performed if the rates match
            return signal, freq_sample_hz

        if reference_freq_khz < sample_rate_target_khz:
            # Resample for the input rate lower than the output
            resample_signal = resample_poly(
                signal, sample_rate_target_khz, reference_freq_khz
            )

            # Match the RMS level of the resampled signal to that of the input
            reference_rms = np.sqrt(np.mean(signal**2))
            resample_rms = np.sqrt(np.mean(resample_signal**2))
            resample_signal = (reference_rms / resample_rms) * resample_signal

            return resample_signal, freq_sample_hz

        # Resample for the input rate higher than the output
        resample_signal = resample_poly(
            signal, sample_rate_target_khz, reference_freq_khz
        )

        # Reduce the input signal bandwidth to 21 kHz (-10.5 to +10.5 kHz)
        # The power equalization is designed to match the signal intensities
        # over the frequency range spanned by the gammatone filter bank.
        # Chebyshev Type 2 LP
        order = 7
        attenuation = 30  # sidelobe attenuation in dB
        reference_freq_cut = 21 / reference_freq_khz
        reference_b, reference_a = cheby2(order, attenuation, reference_freq_cut)
        reference_filter = lfilter(reference_b, reference_a, signal, axis=0)

        # Reduce the resampled signal bandwisth to 21 kHz (-10.5 to +10.5 kHz)
        resample_freq_cut = 21 / sample_rate_target_khz
        target_b, target_a = cheby2(order, attenuation, resample_freq_cut)
        target_filter = lfilter(target_b, target_a, resample_signal, axis=0)

        # Compute the input and output RMS levels within the 21 kHz bandwidth and
        # match the output to the input
        reference_rms = np.sqrt(np.mean(reference_filter**2))
        resample_rms = np.sqrt(np.mean(target_filter**2))
        resample_signal = (reference_rms / resample_rms) * resample_signal

        return resample_signal, freq_sample_hz

    @staticmethod
    def input_align(
        reference: np.ndarray, processed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Approximate temporal alignment of the reference and processed output
        signals. Leading and trailing zeros are then pruned.

        The function assumes that the two sequences have the same sampling rate:
        call eb_Resamp24kHz for each sequence first, then call this function to
        align the signals.

        Argus:
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
        processed_n = len(processed)
        min_sample_length = min(len(reference), processed_n)

        # Determine the delay of the output relative to the reference
        reference_processed_correlation = correlate(
            reference[:min_sample_length] - np.mean(reference[:min_sample_length]),
            processed[:min_sample_length] - np.mean(processed[:min_sample_length]),
            "full",
        )  # Matlab code uses xcov thus the subtraction of mean
        index = np.argmax(np.abs(reference_processed_correlation))
        delay = min_sample_length - index - 1

        # Back up 2 msec to allow for dispersion
        fsamp = 24000  # Cochlear model input sampling rate in Hz
        delay = round(delay - 2 * fsamp / 1000)  # Back up 2 ms

        # Align the output with the reference allowing for the dispersion
        if delay > 0:
            # Output delayed relative to the reference
            processed = np.concatenate((processed[delay:processed_n], np.zeros(delay)))
        else:
            # Output advanced relative to the reference
            processed = np.concatenate(
                (np.zeros(-delay), processed[: processed_n + delay])
            )

        # Find the start and end of the noiseless reference sequence
        reference_abs = np.abs(reference)
        reference_threshold = 0.001 * np.max(reference_abs)  # Zero detection threshold

        above_threshold = np.where(reference_abs > reference_threshold)[0]
        reference_n_above_threshold = above_threshold[0]
        reference_n_below_threshold = above_threshold[-1]

        # Prune the sequences to remove the leading and trailing zeros
        reference_n_below_threshold = min(reference_n_below_threshold, processed_n)

        return (
            reference[reference_n_above_threshold : reference_n_below_threshold + 1],
            processed[reference_n_above_threshold : reference_n_below_threshold + 1],
        )
