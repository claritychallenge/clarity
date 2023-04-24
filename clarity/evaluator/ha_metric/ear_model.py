"""Ear model for hearing aid HASPI, HASQI, HAAQI metrics."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import lfilter, group_delay, resample_poly

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
        "1000": {
            "a": [
                7.304486430753721e-06,
                -3.652243189942665e-05,
                6.57403771137576e-05,
                -3.652243164508467e-05,
                -3.6522431645084686e-05,
                6.57403771137576e-05,
                -3.652243189942664e-05,
                7.304486430753717e-06,
            ],
            "b": [
                1.0,
                -6.999701654296145,
                20.998209970175072,
                -34.99552503642888,
                34.9940335298877,
                -20.995525258398366,
                6.998210147750661,
                -0.9997016986900424,
            ],
        },
        "4000": {
            "a": [
                1.8263259094457514e-06,
                -9.13162954325422e-06,
                1.6436933173088156e-05,
                -9.131629539279679e-06,
                -9.131629539279682e-06,
                1.6436933173088152e-05,
                -9.131629543254222e-06,
                1.8263259094457518e-06,
            ],
            "b": [
                1.0,
                -6.999925413554031,
                20.999552484099084,
                -34.99888121718487,
                34.998508298829314,
                -20.99888123105902,
                6.999552495198395,
                -0.9999254163288592,
            ],
        },
        "8000": {
            "a": [
                9.131799818887095e-07,
                -4.565899908946721e-06,
                8.218619835507907e-06,
                -4.5658999084498955e-06,
                -4.565899908449898e-06,
                8.218619835507905e-06,
                -4.565899908946719e-06,
                9.131799818887092e-07,
            ],
            "b": [
                1.0,
                -6.999962706775348,
                20.99977624134581,
                -34.99944060509882,
                34.99925414244415,
                -20.999440608567404,
                6.999776244120671,
                -0.999962707469063,
            ],
        },
        "12000": {
            "a": [
                6.087904384925872e-07,
                -3.043952192315727e-06,
                5.479113945991657e-06,
                -3.043952192168517e-06,
                -3.043952192168517e-06,
                5.479113945991657e-06,
                -3.043952192315727e-06,
                6.087904384925871e-07,
            ],
            "b": [
                1.0,
                -6.999975137849862,
                20.999850827407492,
                -34.99962706928954,
                34.999502760080446,
                -20.99962707083113,
                6.99985082864077,
                -0.9999751381581813,
            ],
        },
        "16000": {
            "a": [
                4.5659424784400506e-07,
                -2.282971239157921e-06,
                4.109348230409734e-06,
                -2.282971239095819e-06,
                -2.282971239095817e-06,
                4.109348230409735e-06,
                -2.282971239157922e-06,
                4.5659424784400506e-07,
            ],
            "b": [
                1.0,
                -6.999981353387257,
                20.999888120496976,
                -34.99972030167602,
                34.99962706947945,
                -20.999720302543164,
                6.999888121190696,
                -0.9999813535606874,
            ],
        },
        "22050": {
            "a": [
                3.3131640152343904e-07,
                -1.6565820075934677e-06,
                2.9818476136397693e-06,
                -1.6565820075697404e-06,
                -1.6565820075697404e-06,
                2.9818476136397684e-06,
                -1.6565820075934683e-06,
                3.313164015234392e-07,
            ],
            "b": [
                1.0,
                -6.999986469577973,
                20.999918817559152,
                -34.99979704412617,
                34.99972939247261,
                -20.99979704458275,
                6.9999188179244145,
                -0.9999864696692883,
            ],
        },
        "24000": {
            "a": [
                3.0439711121772204e-07,
                -1.521985556070209e-06,
                2.739574000904294e-06,
                -1.5219855560518085e-06,
                -1.5219855560518085e-06,
                2.739574000904294e-06,
                -1.5219855560702088e-06,
                3.04397111217722e-07,
            ],
            "b": [
                1.0,
                -6.9999875689247455,
                20.999925413625554,
                -34.99981353425659,
                34.99975137926572,
                -20.99981353464199,
                6.999925413933876,
                -0.9999875690018258,
            ],
        },
        "44100": {
            "a": [
                1.6565876111669459e-07,
                -8.28293805580507e-07,
                1.4909288500413535e-06,
                -8.282938055775411e-07,
                -8.282938055775413e-07,
                1.4909288500413532e-06,
                -8.282938055805069e-07,
                1.6565876111669456e-07,
            ],
            "b": [
                1.0,
                -6.9999932347889295,
                20.999959408756414,
                -34.999898521948104,
                34.99986469600691,
                -20.999898522062253,
                6.99995940884773,
                -0.99999323481176,
            ],
        },
    }

    def __init__(
        self,
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
    ):
        """
        Constructor takes the reference and processed signals that are to be
        compared. The reference is at the reference intensity (e.g. 65 dB SPL
        or with NAL-R amplification) and has no other processing. The processed
        signal is the hearing-aid output, and is assumed to have the same or
        greater group delay compared to the reference.

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

        """
        self.reference = reference
        self.reference_freq = reference_freq
        self.processed = processed
        self.processed_freq = processed_freq
        self.hearing_loss = hearing_loss
        self.itype = itype
        self.level1 = level1
        self.nchan = nchan
        self.m_delay = m_delay
        self.shift = shift

    def apply(self):
        """
        Apply the model to the signals.

        The method returns the envelopes of the signals after OHC compression
        and IHC loss attenuation.


        Returns:
        reference_db (): envelope for the reference in each band
        reference_basilar_membrane (): BM motion for the reference in each band
        processed_db (): envelope for the processed signal in each band
        processed_basilar_membrane (): BM motion for the processed signal in each band
        reference_sl (): compressed RMS average reference in each band converted
            to dB SL
        processed_sl (): compressed RMS average output in each band converted to dB SL
        freq_sample (): sampling rate in Hz for the model outputs
        """
        # Center frequencies on an ERB scale
        cfreq = self.center_frequencies()

    def center_frequencies(
        self,
        low_freq: int = 80,
        high_freq: int = 8000,
        ear_q: float = 9.26449,
        min_bw: float = 24.7,
    ):
        """
        Compute the Equivalent Rectangular Bandwidth_[1] frequency spacing for the
        gammatone filterbank. The equation comes from Malcolm Slaney[2].

        Arguments:
            low_freq (int): Low Frequency level.
            high_freq (int): High Frequency level.
            ear_q (float):
            min_bw (float):


        Returns:
            center_freq (torch.FloatTensor): Center frequencies of the gammatone filterbank.

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
        """
        if self.shift is not None:
            k = 1
            A = 165.4  # pylint: disable=invalid-name
            a = 2.1  # shift specified as a fraction of the total length
            # Locations of the low and high frequencies on the BM between 0 and 1
            x_low = (1 / a) * np.log10(k + (low_freq / A))
            x_high = (1 / a) * np.log10(k + (high_freq / A))
            # Shift the locations
            x_low = x_low * (1 + self.shift)
            x_high = x_high * (1 + self.shift)
            # Compute the new frequency range
            lowFreq = A * (np.power(10, (a * x_low)) - k)
            highFreq = A * (np.power(10, (a * x_high)) - k)

        # All of the following expressions are derived in Apple TR #35,
        # "An Efficient Implementation of the Patterson-Holdsworth Cochlear
        # Filter Bank" by Malcolm Slaney.
        # https://engineering.purdue.edu/~malcolm/apple/tr35/PattersonsEar.pdf
        _center_freq = -(ear_q * min_bw) + np.exp(
            np.arange(1, self.nchan)
            * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw))
            / (self.nchan - 1)
        ) * (high_freq + ear_q * min_bw)

        # Last center frequency is set to highFreq
        _center_freq = np.concatenate((np.array([high_freq]), _center_freq))
        _center_freq = np.flip(_center_freq)
        return _center_freq

    def loss_parameters(
        self,
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
            attenuated_ohc (np.ndarray): attenuation in dB for the OHC gammatone filters
            bandwidth (np.ndarray): OHC filter bandwidth expressed in terms of normal
            low_knee (np.ndarray): Lower kneepoint for the low-level linear amplification
            compression_ratio (np.ndarray): Ranges from 1.4:1 at 150 Hz to 3.5:1 at 8 kHz for
                normal hearing. Reduced in proportion to the OHC loss to 1:1.
            attenuated_ihc (np.ndarray): attenuation in dB for the input to the IHC synapse
        """
        # Audiometric frequencies in Hz
        if audiometric_freq is None:
            audiometric_freq = np.array([250, 500, 1000, 2000, 4000, 6000])

        # Interpolation to give the loss at the gammatone center frequencies
        # Use linear interpolation in dB. The interpolation assumes that
        # cfreq[0] < aud[0] and cfreq[nfilt -1] > aud[5]
        nfilt = len(center_freq)
        f_v = np.zeros(nfilt + 2)
        f_v[1:-1] = center_freq
        f_v[0] = audiometric_freq[0]
        f_v[-1] = audiometric_freq[-1]

        fv_interp = np.zeros(len(hearing_loss) + 2)
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
            reference_b, reference_a = self.RESAMPLE_COEFS[reference_sample_rate]
            target_b, target_a = self.RESAMPLE_COEFS[target_sample_rate]

            reference_filter = lfilter(
                reference_b, reference_a, reference_signal, axis=0
            )
            target_filter = lfilter(target_b, target_a, resample_signal, axis=0)

            reference_rms = np.sqrt(np.mean(reference_filter**2))
            resample_rms = np.sqrt(np.mean(target_filter**2))

        resample_signal = (reference_rms / resample_rms) * resample_signal

        return resample_signal, target_sample_rate
