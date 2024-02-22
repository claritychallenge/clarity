"""Ear model for the hearing aid model."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numba import njit
from scipy.signal import butter, correlate, group_delay, lfilter

from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram

if TYPE_CHECKING:
    from numpy import ndarray


class Ear:
    """Representation of Ear for the hearing aid model.

    The model assumes signals at 24000 Hz, please ensure that the input
    signals are resampled to 24000 Hz before using the model.

    The ear model for the enhanced signal depends on the reference signal.
    Therefore, the reference signal must be processed first.

    The process is as follows:
    1. Set the audiogram using the `set_audiogram` method.
    2. Process the reference signal using the `process_reference` method.
    3. Process the enhanced signal using the `process_enhanced` method.

    For everytime you set the audiogram, the reference and enhanced signals
    must be processed again.

    The parameter `signals_same_size` has relevance when the ear model is called
    several times to compute scores for different signals. All signals must have
    the same length. This is to prevent the sincf and coscf from being recomputed
    every time a signal is processed.

    Example:
    Crompute the ear model for the reference and enhanced signals.

    >>> from scipy.io import wavfile
    >>> from clarity.evaluator.ha import Ear
    >>> from clarity.utils.audiogram import Audiogram
    >>> from clarity.utils.signal_processing import resample

    The audiogram is required to compute the ear model. The audiogram
    levels and frequencies can be set as follows:

    >>> audiogram_levels = np.array([30, 40, 40, 65, 70, 65])
    >>> audiogram_frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])
    >>> audiogram = Audiogram(
    ...     levels=audiogram_levels,
    ...     frequencies=audiogram_frequencies,
    ... )

    We set some parameters for the ear model:
    >>> sr = 24000
    >>> level1 = 65
    >>> equalisation = 0
    >>> num_bands = 32
    >>> m_delay = 1

    The reference and enhanced signals can be resampled as follows:
    >>> enhanced, sr_e = wavfile.read("enhanced.wav")
    >>> reference, sr_r = wavfile.read("reference.wav")
    >>> enhanced = resample(enhanced, sr_e, sr)
    >>> reference = resample(reference, sr_r, sr)

    The ear model can be computed as follows:
    >>> ear = Ear(equalisation, num_bands, m_delay)
    >>> ear.set_audiogram(audiogram)
    >>> reference_db, reference_basilar_membrane, reference_sl = ear.process_reference(
    ...     reference, level1
    ... )
    >>> enhanced_db, enhanced_basilar_membrane, enhanced_sl = ear.process_enhanced(
    ...      enhanced
    ... )

    The output ```reference_db```, ```reference_basilar_membrane```,
    ```reference_sl```, ```enhanced_db```, ```enhanced_basilar_membrane```,
    and ```enhanced_sl``` are then used by HAAQI, HASQI, and HASPI metrics.
    """

    SAMPLE_RATE: Final = 24000
    SMALL_VALUE: Final = 1e-30

    def __init__(
        self,
        equalisation: int,
        num_bands: int = 32,
        m_delay: int = 1,
        shift: float | None = None,
        signals_same_size: bool = True,
    ):
        """
        Constructor for the Ear model.

        - Signals must be resampled to 24000 Hz before using the model.
        - Reference and enhanced signals must have the same length

        Args:
            equalisation (int): purpose for the calculation:
                 0=intelligibility: reference is normal hearing and must not
                   include NAL-R EQ
                 1=quality: reference does not include NAL-R EQ
                 2=quality: reference already has NAL-R EQ applied
            num_bands (int): auditory frequency bands. Default: 32
            m_delay (int): Compensate for the gammatone group delay. Default: 1
            shift (float): Basal shift of the basilar membrane length. Default: None
            signals_same_size (bool): If True, it assumes that the Ear model object will
                be called several times to compute scores for different signals. All
                signals must have the same length. Default: True
        """

        self.equalisation = equalisation
        self.num_bands = num_bands
        self.m_delay = m_delay
        self.signals_same_size = signals_same_size

        # Compute signal independent parameters that are reused for all signals
        self.center_freq = self.center_frequency()
        self.center_freq_control = self.center_frequency(shift=shift)

        _, self.bandwidth_1, _, _, _ = self.loss_parameters(
            hearing_loss=np.array([100, 100, 100, 100, 100, 100]),
            center_freq=self.center_freq_control,
            audiometric_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
        )

        flp = 800
        b, a = butter(1, flp / (0.5 * self.SAMPLE_RATE))
        self.compress_basilar_membrane_coef: dict = {
            "b": b,
            "a": a,
        }

        butterworth_low_pass, low_pass = butter(1, 5000 / (0.5 * self.SAMPLE_RATE))
        butterworth_high_pass, high_pass = butter(
            2, 350 / (0.5 * self.SAMPLE_RATE), "high"
        )

        self.middle_ear_coef: dict = {
            "butterworth_low_pass": butterworth_low_pass,
            "low_pass": low_pass,
            "butterworth_high_pass": butterworth_high_pass,
            "high_pass": high_pass,
        }

        # Initialise variables that are set when the audiogram is set
        # or when processing the reference signal.
        self.reference_computed = False
        self.level1: float = 65.0
        self.audiogram: Audiogram | None = None

        # Array variables are initialised as empty arrays instead of None.
        # This to prevent a mypy error:
        # >>> error: Value of type "None" is not indexable  [index]
        self.attn_ohc: ndarray = np.empty(0)
        self.bandwidth_min: ndarray = np.empty(0)
        self.low_knee: ndarray = np.empty(0)
        self.compression_ratio: ndarray = np.empty(0)
        self.attn_ihc: ndarray = np.empty(0)

        # Variables used for alignment
        self.reference_cochlear_compression: ndarray = np.empty(0)
        self.reference_bandwidth: ndarray = np.empty(0)
        self.temp_reference_b: ndarray = np.empty(0)
        self.reference_align: ndarray = np.empty(0)

        # sincf and coscf depends on the length of the signals, so
        # they are only computed the first time a reference signal is processed
        self.sincf: ndarray = np.empty(0)
        self.coscf: ndarray = np.empty(0)

        self.sincf_control: ndarray = np.empty(0)
        self.coscf_control: ndarray = np.empty(0)

        # start and end to remove the leading and trailing zeros
        self.start_signal: int = 0
        self.end_signal: int = -1

    def set_audiogram(self, audiogram: Audiogram) -> None:
        """Set the audiogram for the ear model.

        It sets the `reference_computed` attribute to False.

        Args:
            audiogram (Audiogram): Audiogram object.
        """
        self.audiogram = audiogram

        # Reset the variables if the audiogram is set
        self.reference_computed = False
        self.reference_cochlear_compression = np.empty(0)
        self.temp_reference_b = np.empty(0)

        if not self.signals_same_size:
            # reset the variables if the next reference signal has different length
            self.sincf = np.empty(0)
            self.coscf = np.empty(0)
            self.sincf_control = np.empty(0)
            self.coscf_control = np.empty(0)

    def process_reference(
        self, signal: ndarray, level1: float = 65.0
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Process the reference signal.

        The variables `self.attn_ohc`, `self.bandwidth_min`, `self.low_knee`,
        `self.compression_ratio` and, `self.attn_ihc` depend on the metric is running.
        HASPI reference is normal hearing, and HAAQI & HASQI reference is hearing loss.

        Args:
            signal (np.ndarray): Input signal.
            level1 (float): Level calibration:
                signal RMS=1 corresponds to Level1 dB SPL.

        Returns:
            reference_db (np.ndarray): envelope for the reference signal in each band.
            reference_basilar_membrane (np.ndarray): Basilar membrane motion for the
                reference signal in each band
            reference_sl (np.ndarray): compressed RMS average of the reference signal in
                each band converted to dB SL
        """
        num_samples = len(signal)

        self.start_signal, self.end_signal = self.find_noiseless_boundaries(signal)
        signal = signal[self.start_signal : self.end_signal + 1]

        # Save the reference signal for alignment with the enhanced signal
        self.reference_align = signal

        if not isinstance(self.audiogram, Audiogram):
            logging.error("Audiogram is not set")
            raise ValueError("Audiogram is not set. Please set the audiogram first.")

        if len(self.coscf) == 0 or len(self.sincf) == 0:
            # Precompute the coscf and sincf for the reference signal
            # These are reused by the enhanced signal
            self.sincf = np.zeros((self.num_bands, num_samples))
            self.coscf = np.zeros((self.num_bands, num_samples))

            self.sincf_control = np.zeros((self.num_bands, num_samples))
            self.coscf_control = np.zeros((self.num_bands, num_samples))

            tpt = 2 * np.pi / self.SAMPLE_RATE
            for n in range(self.num_bands):
                self.sincf[n], self.coscf[n] = self.gammatone_bandwidth_demodulation(
                    num_samples, tpt, self.center_freq[n]
                )

                (
                    self.sincf_control[n],
                    self.coscf_control[n],
                ) = self.gammatone_bandwidth_demodulation(
                    num_samples, tpt, self.center_freq_control[n]
                )

        # The cochlear model parameters for the reference signal are the same as for the
        # hearing loss when computing quality (HAAQI and HASQI).
        # But, it is `normal hearing` when computing intelligibility (HASPI).

        self.level1 = level1

        if self.equalisation == 0:
            hearing_loss = np.zeros_like(self.audiogram.levels)
        else:
            hearing_loss = self.audiogram.levels

        [
            self.attn_ohc,
            self.bandwidth_min,
            self.low_knee,
            self.compression_ratio,
            self.attn_ihc,
        ] = self.loss_parameters(
            hearing_loss=hearing_loss,
            center_freq=self.center_freq,
            audiometric_freq=self.audiogram.frequencies,
        )

        # For HAAQI and HASQI, add NAL-R equalization if the quality reference doesn't
        # already have it.
        if self.equalisation == 1:
            # Length in samples of the FIR NAL-R EQ filter (24-kHz rate)
            nfir = 140
            enhancer = NALR(nfir, self.SAMPLE_RATE)
            nalr_fir, _ = enhancer.build(self.audiogram)
            signal = enhancer.apply(nalr_fir, signal)
            signal = signal[nfir : nfir + num_samples]

        # Initialised to store the reference cochlear compression and BM motion
        # for alignment with the enhanced signal
        self.reference_cochlear_compression = np.zeros((self.num_bands, num_samples))
        self.temp_reference_b = np.zeros((self.num_bands, num_samples))
        self.reference_bandwidth = np.zeros(self.num_bands)

        # compute the reference signal
        (
            reference_db,
            reference_basilar_membrane,
            reference_sl,
        ) = self.process_common(
            signal=signal,
        )

        # Set the reference computed to True
        self.reference_computed = True

        return reference_db, reference_basilar_membrane, reference_sl

    def process_enhanced(self, signal: ndarray) -> tuple[ndarray, ndarray, ndarray]:
        """Process the enhanced signal.

        The variables `self.attn_ohc`, `self.bandwidth_min`, `self.low_knee`,
        `self.compression_ratio` and, `self.attn_ihc` depend on the metric is running.
        They are recomputed when running HASPI.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            enhanced_db (np.ndarray): envelope for the enhanced signal in each band.
            enhanced_basilar_membrane (np.ndarray): Basilar membrane motion for the
                enhanced signal in each band
            enhanced_sl (np.ndarray): compressed RMS average of the enhanced signal in
                each band converted to dB SL
        """

        if not self.reference_computed:
            logging.error("Reference signal is not computed")
            raise ValueError(
                "Reference signal is not computed."
                "Please compute the reference signal first."
            )

        if not isinstance(self.audiogram, Audiogram):
            raise ValueError("Set the Audiogram before calling `process_enhanced`.")

        # Remove the leading and trailing zeros according the reference signal
        signal = signal[self.start_signal : self.end_signal + 1]

        signal = self.input_align(signal)

        # The cochlear model parameters for the enhanced signal are the same as for the
        # hearing loss if calculating quality.
        # But are for normal hearing if calculating intelligibility (HASPI).
        if self.equalisation == 0:
            [
                self.attn_ohc,
                self.bandwidth_min,
                self.low_knee,
                self.compression_ratio,
                self.attn_ihc,
            ] = self.loss_parameters(
                hearing_loss=self.audiogram.levels,
                center_freq=self.center_freq,
                audiometric_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
            )

        # Compute the enhanced signal
        (
            enhanced_db,
            enhanced_basilar_membrane,
            enhanced_sl,
        ) = self.process_common(
            signal=signal,
        )

        return enhanced_db, enhanced_basilar_membrane, enhanced_sl

    def process_common(self, signal: ndarray) -> tuple[ndarray | Any, Any, Any]:
        """Run common steps for reference and enhanced signals.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            signal_db (np.ndarray): envelope for the signal in each band.
            signal_basilar_membrane (np.ndarray): Basilar membrane motion for the
                signal in each band
            signal_sl (np.ndarray): compressed RMS average of the signal in
                each band converted to dB SL
        """

        num_samples = len(signal)

        # Middle ear
        signal_mid = self.middle_ear(signal)

        signal_db = np.zeros((self.num_bands, num_samples))
        signal_average = np.zeros(self.num_bands)
        signal_control_average = np.zeros(self.num_bands)

        signal_b = np.zeros((self.num_bands, num_samples))

        for n in range(self.num_bands):
            # control signal
            signal_control, _ = self.gammatone_basilar_membrane(
                signal=signal_mid,
                bandwidth=self.bandwidth_1[n],
                center_freq=self.center_freq_control[n],
                coscf=self.coscf_control[n],
                sincf=self.sincf_control[n],
            )

            # Adjust the auditory filter bandwidths for the average signal level
            signal_bandwidth = self.bandwidth_adjust(
                control=signal_control,
                bandwidth_min=self.bandwidth_min[n],
                bandwidth_max=self.bandwidth_1[n],
            )

            # Envelopes and BM motion of the signal
            envelope, basilar_membrane = self.gammatone_basilar_membrane(
                signal_mid,
                signal_bandwidth,
                self.center_freq[n],
                coscf=self.coscf[n],
                sincf=self.sincf[n],
            )

            # RMS levels of the envelopes for linear metric
            signal_average[n] = np.sqrt(np.mean(envelope**2))
            signal_control_average[n] = np.sqrt(np.mean(signal_control**2))

            # Cochlear compression for the signal envelopes and BM motion
            (
                signal_cochlear_compression,
                signal_b[n],
            ) = self.env_compress_basilar_membrane(
                envelope,
                basilar_membrane,
                signal_control,
                self.attn_ohc[n],
                self.low_knee[n],
                self.compression_ratio[n],
            )

            if self.reference_computed:
                # Processing enhanced signal
                signal_cochlear_compression = self.envelope_align(
                    self.reference_cochlear_compression[n], signal_cochlear_compression
                )
                # Align processed BM motion to reference
                signal_b[n] = self.envelope_align(self.temp_reference_b[n], signal_b[n])
            else:
                # Processing reference signal
                self.reference_cochlear_compression[n] = signal_cochlear_compression
                self.temp_reference_b[n] = signal_b[n]
                self.reference_bandwidth[n] = signal_bandwidth

            # Convert the compressed envelopes and BM vibration envelopes to dB SPL
            signal_cochlear_compression, signal_b[n] = self.envelope_sl(
                envelope=signal_cochlear_compression,
                basilar_membrane=signal_b[n],
                attenuated_ihc=self.attn_ihc[n],
            )

            # Apply the IHC rapid and short-term adaptation
            delta = 2  # Amount of overshoot

            signal_db[n], signal_b[n] = self.inner_hair_cell_adaptation(
                signal_cochlear_compression,
                signal_b[n],
                delta,
                self.SAMPLE_RATE,
                self.SMALL_VALUE,
            )
        # Additive noise level to give the auditory threshold
        ihc_threshold = -10  # Additive noise level, dB re: auditory threshold
        signal_basilar_membrane = self.basilar_membrane_add_noise(
            signal_b, ihc_threshold
        )

        if self.m_delay > 0:
            signal_db = self.group_delay_compensate(
                input_signal=signal_db,
                bandwidths=self.reference_bandwidth,
                center_freq=self.center_freq,
            )

            signal_basilar_membrane = self.group_delay_compensate(
                input_signal=signal_basilar_membrane,
                bandwidths=self.reference_bandwidth,
                center_freq=self.center_freq,
            )

        # Convert average gammatone outputs to dB SPL
        signal_sl = self.convert_rms_to_sl(
            signal_average,
            signal_control_average,
            self.attn_ohc,
            self.low_knee,
            self.compression_ratio,
            self.attn_ihc,
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
            shift (): optional frequency shift of the filter bank specified as a
                fractional shift in distance along the BM. A positive shift is an
                increase in frequency (basal shift), and negative is a decrease
                in frequency (apical shift). The total length of the BM is normalized
                to 1. The frequency-to-distance map is from D.D. Greenwood[3].
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
            np.arange(1, self.num_bands)
            * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw))
            / (self.num_bands - 1)
        ) * (high_freq + ear_q * min_bw)

        # Last center frequency is set to highFreq
        _center_freq = np.insert(_center_freq, 0, high_freq)
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
            compression_ratio (): Ranges from 1.4:1 at 150 Hz to 3.5:1 at 8 kHz for
                normal hearing. Reduced in proportion to the OHC loss to 1:1.
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

    def middle_ear(self, signal: ndarray) -> ndarray:
        """
        Design the middle ear filters and process the input through the
        cascade of filters. The middle ear model is a 2-pole HP filter
        at 350 Hz in series with a 1-pole LP filter at 5000 Hz. The
        result is a rough approximation to the equal-loudness contour
        at threshold.

        Args:
            signal (np.ndarray): signal to be processed

        Returns:
            signal (np.ndarray): signal processed through the middle ear filters
        """
        # LP filter the input
        signal = lfilter(
            self.middle_ear_coef["butterworth_low_pass"],
            self.middle_ear_coef["low_pass"],
            signal,
        )

        # HP filter the signal
        return lfilter(
            self.middle_ear_coef["butterworth_high_pass"],
            self.middle_ear_coef["high_pass"],
            signal,
        )

    def gammatone_basilar_membrane(
        self,
        signal: ndarray,
        bandwidth: float,
        center_freq: float,
        coscf: ndarray,
        sincf: ndarray,
        ear_q: float = 9.26449,
        min_bandwidth: float = 24.7,
    ) -> tuple[ndarray, ndarray]:
        """
        4th-order gammatone auditory filter. This implementation is based on
        the c program published on-line by Ning Ma, U. Sheffield, UK[1]_ that
        gives an implementation of the Martin Cooke filters[2]_:
        an impulse-invariant transformation of the gammatone
        filter. The signal is demodulated down to baseband using a complex exponential,
        and then passed through a cascade of four one-pole low-pass filters.

        This version filters two signals that have the same sampling rate and the same
        gammatone filter center frequencies. The lengths of the two signals should
        match; if they don't, the signals are truncated to the shorter of the two
        lengths.

        Args:
            signal (): first sequence to be filtered
            bandwidth: bandwidth for x relative to that of a normal ear
            center_freq (int): filter center frequency in Hz
            coscf (): cosine for centre frequency
            sincf (): sine for centre frequency
            ear_q: (float): ???
            min_bandwidth (float): ???

        Returns:
            envelope (): envelope of the filtered signal
            basilar_membrane (): BM motion output by the filter bank
        """
        # Filter Equivalent Rectangular Bandwidth from Moore and Glasberg (1983)
        # doi: 10.1121/1.389861
        erb = min_bandwidth + (center_freq / ear_q)

        # Filter the first signal
        # Initialize the filter coefficients
        tpt = 2 * np.pi / self.SAMPLE_RATE
        tpt_bw = bandwidth * tpt * erb * 1.019
        a = np.exp(-tpt_bw)
        a_1 = 4.0 * a
        a_2 = -6.0 * a * a
        a_3 = 4.0 * a * a * a
        a_4 = -a * a * a * a
        a_5 = 4.0 * a * a
        gain = 2.0 * (1 - a_1 - a_2 - a_3 - a_4) / (1 + a_1 + a_5)

        # Initialize the complex demodulation
        # Filter the real and imaginary parts of the signal

        ureal = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], signal * coscf)
        uimag = lfilter([1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], signal * sincf)
        assert isinstance(ureal, np.ndarray)
        assert isinstance(uimag, np.ndarray)

        # Extract the BM velocity and the envelope
        basilar_membrane = gain * (ureal * coscf + uimag * sincf)
        envelope = gain * np.sqrt(ureal * ureal + uimag * uimag)

        return envelope, basilar_membrane

    def bandwidth_adjust(
        self, control: ndarray, bandwidth_min: float, bandwidth_max: float
    ) -> float:
        """
        Compute the increase in auditory filter bandwidth in response to high signal
        levels.

        Args:
            control (): envelope output in the control filter band
            bandwidth_min (): auditory filter bandwidth computed for the loss (or NH)
            bandwidth_max (): auditory filter bandwidth at maximum OHC damage

        Returns:
            bandwidth (): filter bandwidth increased for high signal levels
        """

        # Compute the control signal level
        control_rms = np.sqrt(np.mean(control**2))
        control_db = 20 * np.log10(control_rms) + self.level1

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
        below the lower threshold, compressive with a compression ratio of CR:1 between
        the lower and upper thresholds, and reverts to linear above the upper threshold.
        The compressor assumes that auditory threshold is 0 dB SPL.

        Args:
            envsig (): analytic signal envelope (magnitude) returned by the
                    gammatone filter bank
            bm (): BM motion output by the filter bank
            control (): analytic control envelope returned by the wide control
                    path filter bank
            attn_ohc (): OHC attenuation at the input to the compressor
            threshold_low (): kneepoint for the low-level linear amplification
            compression_ratio (): compression ratio
            threshold_high: kneepoint for the high-level linear amplification

        Returns:
            compressed_signal (): compressed version of the signal envelope
            compressed_basilar_membrane (): compressed version of the BM motion
        """
        # Convert the control envelope to dB SPL
        logenv = np.maximum(control, self.SMALL_VALUE)
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
            self.compress_basilar_membrane_coef["b"],
            self.compress_basilar_membrane_coef["a"],
            gain,
        )

        # Apply the gain to the signals
        compressed_signal = gain * envsig
        compressed_basilar_membrane = gain * bm

        return compressed_signal, compressed_basilar_membrane

    def envelope_sl(
        self, envelope: ndarray, basilar_membrane: ndarray, attenuated_ihc: float
    ) -> tuple[ndarray, ndarray]:
        """
        Convert the compressed envelope returned by cochlear_envcomp to dB SL.

        Args:
            envelope (): linear envelope after compression
            basilar_membrane (): linear Basilar Membrane vibration after compression
            attenuated_ihc (): IHC attenuation at the input to the synapse

        Returns:
            _reference (): reference envelope in dB SL
            _basilar_membrane (): Basilar Membrane vibration with envelope converted to
                dB SL
        """
        # Convert the envelope to dB SL
        _envelope = (
            self.level1 - attenuated_ihc + 20 * np.log10(envelope + self.SMALL_VALUE)
        )
        _envelope = np.maximum(_envelope, 0)

        # Convert the linear BM motion to have a dB SL envelope
        gain = (_envelope + self.SMALL_VALUE) / (envelope + self.SMALL_VALUE)

        return _envelope, gain * basilar_membrane

    def basilar_membrane_add_noise(self, signal: ndarray, threshold: int) -> ndarray:
        """
        Apply the IHC attenuation to the BM motion and to add a low-level Gaussian noise
        to give the auditory threshold.

        Args:
            signal (): BM motion to be attenuated
            threshold (): additive noise level in dB re:auditory threshold

        Returns:
            Attenuated signal with threshold noise added
        """
        # Linear gain for the noise
        gain = 10 ** ((float(threshold) - self.level1) / 20)
        noise = np.random.standard_normal(signal.shape)
        noise = gain * noise
        return signal + noise

    def group_delay_compensate(
        self,
        input_signal: ndarray,
        bandwidths: ndarray,
        center_freq: ndarray,
        ear_q: float = 9.26449,
        min_bandwidth: float = 24.7,
    ) -> ndarray:
        """
        Compensate for the group delay of the gammatone filter bank. The group
        delay is computed for each filter at its center frequency. The firing
        rate output of the IHC model is then adjusted so that all outputs have
        the same group delay.

        Args:
            input_signal (np.ndarray): matrix of signal envelopes or BM motion
            bandwidths (np.ndarray): array of filter bandwidths
            center_freq (np.ndarray): array of filter center frequencies
            ear_q (float): ???
            min_bandwidth (float): ???

        Returns:
            processed (): envelopes or BM motion compensated for the group delay.
        """

        # Filter ERB from Moore and Glasberg (1983)
        erb = min_bandwidth + (center_freq / ear_q)

        # Initialize the gammatone filter coefficients
        tpt = 2 * np.pi / self.SAMPLE_RATE
        tpt_bandwidth = tpt * 1.019 * bandwidths * erb
        a = np.exp(-tpt_bandwidth)
        a_1 = 4.0 * a
        a_2 = -6.0 * a * a
        a_3 = 4.0 * a * a * a
        a_4 = -a * a * a * a
        a_5 = 4.0 * a * a

        # Compute the group delay in samples at fsamp for each filter
        _group_delay = np.zeros(self.num_bands)
        for n in range(self.num_bands):
            _, _group_delay[n] = group_delay(
                ([1, a_1[n], a_5[n]], [1, -a_1[n], -a_2[n], -a_3[n], -a_4[n]]), 1
            )
        _group_delay = np.rint(_group_delay).astype(int)  # convert to integer samples

        # Compute the delay correlation
        group_delay_min = np.min(_group_delay)
        # Remove the minimum delay from all the over values
        _group_delay = _group_delay - group_delay_min
        group_delay_max = np.max(_group_delay)
        # Samples delay needed to add to give alignment
        correct = group_delay_max - _group_delay

        # Add delay correction to each frequency band
        processed = np.zeros_like(input_signal)
        npts = input_signal.shape[1]
        for n in range(self.num_bands):
            ref = input_signal[n]
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
    ) -> ndarray:
        """
        Covert the Root Mean Square average output of the gammatone filter bank
        into dB SL. The gain is linear below the lower threshold, compressive
        with a compression ratio of CR:1 between the lower and upper thresholds,
        and reverts to linear above the upper threshold. The compressor
        assumes that auditory threshold is 0 dB SPL.

        Args:
            reference (): analytic signal envelope (magnitude) returned by the
                gammatone filter bank, RMS average level
            control (): control signal envelope
            attenuated_ohc (): OHC attenuation at the input to the compressor
            threshold_low (): kneepoint for the low-level linear amplification
            compression_ratio (): compression ratio
            attenuated_ihc (): IHC attenuation at the input to the synapse

        Returns:
            reference_db (): compressed output in dB above the impaired threshold
        """
        control_db_spl = np.maximum(control, self.SMALL_VALUE)
        control_db_spl = self.level1 + 20 * np.log10(control_db_spl)
        control_db_spl = np.minimum(control_db_spl, 100)
        control_db_spl = np.maximum(control_db_spl, threshold_low)

        # Compute compression gain in dB
        gain = -attenuated_ohc - (control_db_spl - threshold_low) * (
            1 - (1 / compression_ratio)
        )

        # Convert the signal envelope to dB SPL
        control_db_spl = np.maximum(reference, self.SMALL_VALUE)
        control_db_spl = self.level1 + 20 * np.log10(control_db_spl)
        control_db_spl = np.maximum(control_db_spl, 0)
        reference_db = control_db_spl + gain - attenuated_ihc
        reference_db = np.maximum(reference_db, 0)

        return reference_db

    def envelope_align(
        self,
        reference: ndarray,
        output: ndarray,
        corr_range: int = 100,
    ) -> ndarray:
        """
        Align the envelope of the processed signal to that of the reference signal.

        Args:
            reference (): envelope or BM motion of the reference signal
            output (): envelope or BM motion of the output signal
            corr_range (int): range in msec for the correlation

        Returns:
            y (): shifted output envelope to match the input
        """

        # The MATLAB code limits the range of lags to search (to 100 ms) to save
        # computation time - no such option exists in numpy, but the code below
        # limits the delay to the same range as in Matlab, for consistent results

        # Range in samples
        lags = np.rint(0.001 * corr_range * self.SAMPLE_RATE).astype(int)
        npts = len(reference)
        lags = min(lags, npts)

        ref_out_correlation = correlate(reference, output, "same")

        location = np.argmax(
            np.abs(ref_out_correlation[int(npts / 2) - lags : int(npts / 2) + lags])
        )
        delay = lags - location

        # Time shift the output sequence
        if delay > 0:
            # Output delayed relative to the reference
            return np.concatenate((output[delay:npts], np.zeros(delay)))
        return np.concatenate((np.zeros(-delay), output[: npts + delay]))

    @staticmethod
    @njit
    def find_noiseless_boundaries(signal: ndarray) -> tuple[int, int]:
        """Prune silence from the signal.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            start (int): Start index of the signal.
            end (int): End index of the signal.
        """
        signal_abs = np.abs(signal)
        signal_max = np.max(signal_abs)
        threshold = 0.001 * signal_max  # Zero detection threshold

        above_threshold = np.where(signal_abs > threshold)[0]
        start = above_threshold[0]
        end = min(above_threshold[-1], len(signal))
        return start, end

    @staticmethod
    @njit
    def gammatone_bandwidth_demodulation(npts, tpt, center_freq):
        """Gamma tone bandwidth demodulation

        Arguments:
            npts (): ???
            tpt (): ???
            center_freq (): ???

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

        Args:
            signal_db (np.ndarray): signal envelope in one frequency band in dB SL
                 contains OHC compression and IHC attenuation
            basilar_membrane (): basilar membrane vibration with OHC compression
                but no IHC attenuation
            delta (): overshoot factor = delta x steady-state
            freq_sample (int): sampling rate in Hz
            small (): small value to avoid divide by zero

        Returns:
            output_db (): envelope in dB SL with IHC adaptation
            output_basilar_membrane (): Basilar Membrane multiplied by the IHC
                adaptation gain function

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

    def input_align(self, signal):
        """
        Approximate temporal alignment of the reference and processed output
        signals. Leading and trailing zeros are then pruned.

        Args:
             signal (np.ndarray): hearing-aid output sequence

        Returns:
              signal (np.ndarray): shifted
        """
        signal_length = len(signal)

        reference_processed_correlation = correlate(
            self.reference_align,
            signal,
            mode="same",
        )
        index = np.argmax(np.abs(reference_processed_correlation))
        delay = int(signal_length / 2) - index

        # Back up 2 msec to allow for dispersion
        delay = np.rint(delay - 2 * self.SAMPLE_RATE / 1000.0).astype(int)

        # Align the output with the reference allowing for the dispersion
        if delay > 0:
            # Output delayed relative to the reference
            return np.concatenate((signal[delay:signal_length], np.zeros(delay)))
        # Output advanced relative to the reference
        return np.concatenate((np.zeros(-delay), signal[: signal_length + delay]))
