"""Ear model for the hearing aid model."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final, Tuple

import numpy as np
from numba import njit
from scipy.signal import butter, correlate, lfilter

from clarity.utils.audiogram import Audiogram

if TYPE_CHECKING:
    from numpy import ndarray


class Ear:
    """Representation of Ear for the hearing aid model.

    The model assumes signals at 24000 Hz, please ensure that the input
    signals are resampled to 24000 Hz before using the model.

    """

    BANDWITH_1: Final = np.array(
        1.280963780608,
        1.34463367600863,
        1.40430675025603,
        1.46192770111434,
        1.51915683759555,
        1.57734190779408,
        1.63752299850873,
        1.70045880652767,
        1.76666378924627,
        1.83644871838567,
        1.90996001038767,
        1.98721534334793,
        2.06813449183959,
        2.15256517637044,
        2.24030420428322,
        2.33111441100562,
        2.42473799259552,
        2.52090681759646,
        2.61935025946742,
        2.71980102514491,
        2.82199938493744,
        2.92569614162676,
        3.03065461604316,
        3.13665187384759,
        3.24347937385736,
        3.35094318140336,
        3.45886386002423,
        3.56707613033287,
        3.67542836520979,
        3.78378197475444,
        3.89201072193444,
        4,
    )

    COMPRESS_BASILAR_MEMBRANE_COEFS: Final = {
        "24000": {
            "b": [0.09510798340249643, 0.09510798340249643],
            "a": [1.0, -0.8097840331950071],
        }
    }
    GROUP_DELAY_COEFS: Final = [
        0,
        50,
        92,
        127,
        157,
        183,
        205,
        225,
        242,
        256,
        267,
        275,
        283,
        291,
        299,
        305,
        311,
        316,
        320,
        325,
        329,
        332,
        335,
        338,
        340,
        341,
        342,
        344,
        344,
        345,
        346,
        347,
    ]

    # Middle ear filter coefficients
    MIDDLE_EAR_COEF: Final = {
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

    def __init__(
        self,
        itype: int,
        nchan: int = 32,
        m_delay: int = 1,
        shift: float | None = None,
    ):
        """
        Constructor for the Ear model.
        Args:
            itype (int): purpose for the calculation:
                 0=intelligibility: reference is normal hearing and must not
                   include NAL-R EQ
                 1=quality: reference does not include NAL-R EQ
                 2=quality: reference already has NAL-R EQ applied
            level1:   level calibration: signal RMS=1 corresponds to Level1 dB SPL
            nchan (int): auditory frequency bands. Default: 32
            m_delay (int): Compensate for the gammatone group delay. Default: 1
            shift (float): Basal shift of the basilar membrane length. Default: None
        """
        self.sample_rate = 24000
        self.itype = itype
        self.level1 = None
        self.nchan = nchan
        self.m_delay = m_delay
        self.shift = shift
        self.audiogram = None
        self._center_freq = self.center_frequency(self.nchan)
        self.small = 1e-30

    def set_audiogram(self, audiogram: Audiogram) -> None:
        """Set the audiogram for the ear model.
        Args:
            audiogram (Audiogram): Audiogram object.
        """

        self.audiogram = audiogram

    def process(
        self, signal: ndarray, sample_rate: float, level1: float = 65.0
    ) -> tuple[ndarray | Any, Any, Any]:
        """Process the input signal and return the HAAQI score.
        Args:
            signal (np.ndarray): Input signal.
            sample_rate (float): Sampling rate of the input signal.
            level1 (float): Level calibration: signal RMS=1 corresponds to Level1 dB SPL.
        Returns:
            float: HAAQI score.
        """

        self.level1 = level1
        if self.audiogram is None:
            logging.error(
                "Error: Audiogram is not set. "
                "Please set the audiogram before processing."
            )
            raise ValueError("Audiogram is not set.")
        if sample_rate != self.sample_rate:
            logging.error(
                "Error: only a sampling frequency of 24000 Hz can be used by HAAQI, "
                "HASQI and HASPI."
            )
            raise ValueError("Invalid sampling frequency, valid value is 24000")

        if self.shift in [None, 0]:
            _center_freq_control = self._center_freq.copy()
        else:
            _center_freq_control = self.center_frequency(shift=self.shift)

        nsamp = len(signal)

        if self.itype == 0:
            hearing_loss_x = np.zeros(len(self.audiogram.frequencies))
        else:
            hearing_loss_x = self.audiogram.levels.copy()
        [
            attn_ohc_x,
            bandwidth_min_x,
            low_knee_x,
            compression_ratio_x,
            attn_ihc_x,
        ] = self.loss_parameters(hearing_loss_x, self._center_freq)

        if self.itype == 1:
            pass

        signal_mid = self.middle_ear(signal)
        signal_db = np.zeros((self.nchan, nsamp))
        signal_average = np.zeros(self.nchan)
        signal_control_average = np.zeros(self.nchan)
        signal_bandwidth = np.zeros(self.nchan)
        signal_b = np.zeros((self.nchan, nsamp))

        for n in range(self.nchan):
            signal_control, _ = self.gammatone_basilar_membrane(
                signal_mid, self.BANDWITH_1[n], _center_freq_control
            )
            signal_bandwidth[n] = self.bandwidth_adjust(
                signal_control, bandwidth_min_x[n], self.BANDWITH_1[n], self.level1
            )
            envelope, basilar_membrane = self.gammatone_basilar_membrane(
                signal_mid,
                signal_bandwidth[n],
                self._center_freq[n],
            )
            signal_average[n] = np.sqrt(np.mean(envelope**2))
            signal_control_average[n] = np.sqrt(np.mean(signal_control**2))

            (
                signal_cochlear_compression,
                signal_b[n],
            ) = self.env_compress_basilar_membrane(
                envelope,
                basilar_membrane,
                signal_control,
                attn_ohc_x[n],
                low_knee_x[n],
                compression_ratio_x[n],
            )
            signal_cochlear_compression, signal_b[n] = self.envelope_sl(
                signal_cochlear_compression, signal_b[n], attn_ihc_x[n], self.level1
            )
            # Apply the IHC rapid and short-term adaptation
            delta = 2  # Amount of overshoot

            signal_db[n], signal_b[n] = self.inner_hair_cell_adaptation(
                signal_cochlear_compression, signal_b[n], delta, self.sample_rate
            )
        # Additive noise level to give the auditory threshold
        ihc_threshold = -10  # Additive noise level, dB re: auditory threshold
        signal_basilar_membrane = self.basilar_membrane_add_noise(
            signal_b, ihc_threshold
        )

        if self.m_delay > 0:
            signal_db = self.group_delay_compensate(signal_db)

            signal_basilar_membrane = self.group_delay_compensate(
                signal_basilar_membrane
            )

        # Convert average gammatone outputs to dB SPL
        signal_sl = self.convert_rms_to_sl(
            signal_average,
            signal_control_average,
            attn_ohc_x,
            low_knee_x,
            compression_ratio_x,
            attn_ihc_x,
        )

        return signal_db, signal_basilar_membrane, signal_sl

    def center_frequency(
        self,
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
            center_freq (np.ndarray): Center frequencies of the gammatone filter bank.
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
            np.arange(1, self.nchan)
            * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw))
            / (self.nchan - 1)
        ) * (high_freq + ear_q * min_bw)
        _center_freq = np.insert(
            _center_freq, 0, high_freq
        )  # Last center frequency is set to highFreq
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
        f_v = np.zeros(len(audiometric_freq) + 2)
        f_v[0] = center_freq[0]
        f_v[1:-1] = audiometric_freq
        f_v[-1] = center_freq[-1]

        # Interpolated gain in dB
        loss_temp = np.zeros(len(hearing_loss) + 2)
        loss_temp[0] = hearing_loss[0]
        loss_temp[1:-1] = hearing_loss
        loss_temp[-1] = hearing_loss[-1]
        loss = np.interp(
            center_freq,
            f_v,
            loss_temp,
        )
        # Make sure there are no negative losses
        loss[loss < 0] = 0

        # Compression ratio changes linearly with ERB rate from 1.25:1 in the 80-Hz
        # frequency band to 3.5:1 in the 8-kHz frequency band
        compression_ratio = 1.25 + 2.25 * np.arange(nfilt) / (nfilt - 1)

        # Maximum OHC sensitivity loss depends on the compression ratio. The compression
        # I/O curves assume linear below 30 and above 100 dB SPL in normal ears.
        # HC loss that results in 1:1 compression
        max_ohc = 70 * (1 - (1 / compression_ratio))
        theoretical_ohc = (
            1.25 * max_ohc
        )  # Loss threshold for adjusting the OHC parameters

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
        upamp = 30 + (70 / compression_ratio)  # Output level for an input of 100 dB SPL

        compression_ratio = (100 - low_knee) / (
            upamp + attenuated_ohc - low_knee
        )  # OHC loss Compression ratio

        return attenuated_ohc, bandwidth, low_knee, compression_ratio, attenuated_ihc

    def middle_ear(self, signal: ndarray) -> ndarray:
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
        """
        # LP filter the input
        signal = lfilter(
            self.MIDDLE_EAR_COEF["24000"]["butterworth_low_pass"],
            self.MIDDLE_EAR_COEF["24000"]["low_pass"],
            signal,
        )

        # HP filter the signal
        return lfilter(
            self.MIDDLE_EAR_COEF["24000"]["butterworth_high_pass"],
            self.MIDDLE_EAR_COEF["24000"]["high_pass"],
            signal,
        )

    def gammatone_basilar_membrane(
        self,
        signal: ndarray,
        bandwidth: float,
        center_freq: float,
        ear_q: float = 9.26449,
        min_bandwidth: float = 24.7,
    ) -> tuple[ndarray, ndarray]:
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
            signal (): first sequence to be filtered
            bandwidth: bandwidth for x relative to that of a normal ear
            center_freq (int): filter center frequency in Hz
            ear_q: (float): ???
            min_bandwidth (float): ???

        Returns:
            reference_envelope (): filter envelope output (modulated down to baseband)
                1st signal
            reference_basilar_membrane (): Basilar Membrane for the first signal
            processed_envelope (): filter envelope output (modulated down to baseband)
                2nd signal
            processed_basilar_membrane (): Basilar Membrane for the second signal
        """
        # Filter Equivalent Rectangular Bandwidth from Moore and Glasberg (1983)
        # doi: 10.1121/1.389861
        erb = min_bandwidth + (center_freq / ear_q)

        # Filter the first signal
        # Initialize the filter coefficients
        tpt = 2 * np.pi / self.sample_rate
        tpt_bw = bandwidth * tpt * erb * 1.019
        a = np.exp(-tpt_bw)
        a_1 = 4.0 * a
        a_2 = -6.0 * a * a
        a_3 = 4.0 * a * a * a
        a_4 = -a * a * a * a
        a_5 = 4.0 * a * a
        gain = 2.0 * (1 - a_1 - a_2 - a_3 - a_4) / (1 + a_1 + a_5)

        # Initialize the complex demodulation
        npts = len(signal)
        sincf, coscf = self.gammatone_bandwidth_demodulation(npts, tpt, center_freq)

        # Filter the real and imaginary parts of the signal
        ureal = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], signal * coscf)
        uimag = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], signal * sincf)
        assert isinstance(ureal, np.ndarray)
        assert isinstance(uimag, np.ndarray)

        # Extract the BM velocity and the envelope
        reference_basilar_membrane = gain * (ureal * coscf + uimag * sincf)
        reference_envelope = gain * np.sqrt(ureal * ureal + uimag * uimag)

        return (
            reference_envelope,
            reference_basilar_membrane,
        )

    def bandwidth_adjust(
        self,
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
        logenv = self.level1 + 20 * np.log10(logenv)
        # Clip signal levels above the upper threshold
        logenv = np.minimum(logenv, threshold_high)
        # Clip signal at the lower threshold
        logenv = np.maximum(logenv, threshold_low)

        # Compute the compression gain in dB
        gain = -attn_ohc - (logenv - threshold_low) * (1 - (1 / compression_ratio))

        # Convert the gain to linear and apply a LP filter to give a 0.2 ms delay
        gain = 10 ** (gain / 20)

        # a and b were computed as follows using fsamp = 24000 Hz:
        # flp = 800
        # b, a = butter(1, flp / (0.5 * fsamp))
        gain = lfilter(
            self.COMPRESS_BASILAR_MEMBRANE_COEFS["24000"]["b"],
            self.COMPRESS_BASILAR_MEMBRANE_COEFS["24000"]["a"],
            gain,
        )

        # Apply the gain to the signals
        compressed_signal = gain * envsig
        compressed_basilar_membrane = gain * bm

        return compressed_signal, compressed_basilar_membrane

    def envelope_sl(
        self,
        envelope: ndarray,
        basilar_membrane: ndarray,
        attenuated_ihc: float,
        level1: float,
    ) -> tuple[ndarray, ndarray]:
        """
        Convert the compressed envelope returned by cochlear_envcomp to dB SL.

        Arguments:
            signal (): linear envelope after compression
            basilar_membrane (): linear Basilar Membrane vibration after compression
            attenuated_ihc (): IHC attenuation at the input to the synapse
            level1 (): level in dB SPL corresponding to 1 RMS

        Returns:
            _reference (): reference envelope in dB SL
            _basilar_membrane (): Basilar Membrane vibration with envelope converted to
                dB SL
        """
        # Convert the envelope to dB SL
        _envelope = level1 - attenuated_ihc + 20 * np.log10(envelope + self.small)
        _envelope = np.maximum(_envelope, 0)

        # Convert the linear BM motion to have a dB SL envelope
        gain = (_envelope + self.small) / (envelope + self.small)

        return _envelope, gain * basilar_membrane

    @staticmethod
    @njit
    def inner_hair_cell_adaptation(
        signal_db, basilar_membrane, delta, freq_sample, small
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
            signal_db (np.ndarray): signal envelope in one frequency band in dB SL
                 contains OHC compression and IHC attenuation
            basilar_membrane (): basilar membrane vibration with OHC compression
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
        nsamp = len(signal_db)

        output_db = np.zeros_like(signal_db)
        v_1 = 0
        v_2 = 0

        # Loop to process the envelope signal
        # The gain asymptote is 1 for an input envelope of 0 dB SPL
        for n in range(nsamp):
            v_0 = signal_db[n]
            b_1 = v_0 * r_2 + product_r1_r2_c1 * v_1
            b_2 = product_r2_r3_c2 * v_2
            v_1 = denom * (a22 * b_1 - a12 * b_2)
            v_2 = denom * (-a21 * b_1 + a11 * b_2)
            out = (v_0 - v_1) * r_1_inv
            output_db[n] = out

        output_db = np.maximum(output_db, 0)
        gain = (output_db + small) / (signal_db + small)

        return output_db, gain * basilar_membrane

    def basilar_membrane_add_noise(self, signal: ndarray, threshold: int) -> ndarray:
        """
        Apply the IHC attenuation to the BM motion and to add a low-level Gaussian noise to
        give the auditory threshold.

        Arguments:
            reference (): BM motion to be attenuated
            threshold (): additive noise level in dB re:auditory threshold
            level1 (): an input having RMS=1 corresponds to Level1 dB SPL

        Returns:
            Attenuated signal with threshold noise added
        """
        gain = 10 ** ((threshold - self.level1) / 20)  # Linear gain for the noise
        noise = gain * np.random.standard_normal(signal.shape)
        return signal + noise

    def group_delay_compensate(
        self,
        input_signal: ndarray,
    ) -> ndarray:
        """
        Compensate for the group delay of the gammatone filter bank. The group
        delay is computed for each filter at its center frequency. The firing
        rate output of the IHC model is then adjusted so that all outputs have
        the same group delay.

        Arguments:
            xenv (np.ndarray): matrix of signal envelopes or BM motion

        Returns:
            processed (): envelopes or BM motion compensated for the group delay.
        """
        # Add delay correction to each frequency band
        processed = np.zeros_like(input_signal)
        npts = len(input_signal.shape[1])
        for n in range(self.nchan):
            ref = input_signal[n]
            processed[n] = np.concatenate(
                (
                    np.zeros(self.GROUP_DELAY_COEFS[n]),
                    ref[: npts - self.GROUP_DELAY_COEFS[n]],
                )
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
        """
        control_db_spl = np.maximum(control, self.small)
        control_db_spl = self.level1 + 20 * np.log10(control_db_spl)
        control_db_spl = np.minimum(control_db_spl, 100)
        control_db_spl = np.maximum(control_db_spl, threshold_low)

        # Compute compression gain in dB
        gain = -attenuated_ohc - (control_db_spl - threshold_low) * (
            1 - (1 / compression_ratio)
        )

        # Convert the signal envelope to dB SPL
        control_db_spl = np.maximum(reference, small)
        control_db_spl = self.level1 + 20 * np.log10(control_db_spl)
        control_db_spl = np.maximum(control_db_spl, 0)
        reference_db = control_db_spl + gain - attenuated_ihc
        reference_db = np.maximum(reference_db, 0)

        return reference_db

    @staticmethod
    @njit
    def gammatone_bandwidth_demodulation(npts, tpt, center_freq):
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
        center_freq_cos = np.zeros(npts)
        center_freq_sin = np.zeros(npts)

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
