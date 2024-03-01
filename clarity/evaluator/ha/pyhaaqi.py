"""Implementation PyHAAQI evaluator."""
from __future__ import annotations

import logging
from typing import Final

import numpy as np
from numpy import ndarray
from scipy.signal import convolve, correlate, firwin

from clarity.evaluator.ha.earmodel import Ear
from clarity.utils.audiogram import Audiogram
from clarity.utils.signal_processing import resample

logger = logging.getLogger(__name__)


class HaaqiV1:
    """HAAQI evaluator class.

    This class implements the HAAQI evaluator as described in the paper:
    "The Hearing-Aid Audio Quality Index (HAAQI)"
    https://doi.org/10.1109%2FTASLP.2015.2507858

    The Class is designed to be used with the following workflow:
    The process several signals with the same audiogram and the same reference signal.
    1. Set the audiogram using the `set_audiogram` method.
    2. Set the reference signal using the `set_reference` method.
    3. Process the enhanced signal using the `score` method.

    Then, for each enhanced signal, the `score` method can be called to obtain
    the HAAQI score.

    Or, to process a differen reference and enhanced signal using the same
    audiogram:
    1. Set the audiogram using the `set_audiogram` method.
    2. Process the reference and enhanced signals using the `process` method.

    Then, call the `process` method for each reference and enhanced signal to
    obtain the HAAQI score.

    Example 1:
    For scoring different enhanced signals with the same audiogram and reference:
    >>> from clarity.evaluator.ha import HaaqiV1
    >>> from clarity.utils.audiogram import Audiogram
    >>> from scipy.io import wavfile

    Create an audiogram
    >>> audiogram_levels = np.array([30, 40, 40, 65, 70, 65])
    >>> audiogram_frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])
    >>> audiogram = Audiogram(
    >>>     levels=audiogram_levels,
    >>>     frequencies=audiogram_frequencies,
    >>> )

    Load enhanced and reference signals
    >>> sr_r, reference = wavfile.read("reference.wav")
    >>> sr_e_1, enhanced_1 = wavfile.read("enhanced_1.wav")
    >>> sr_e_2, enhanced_2 = wavfile.read("enhanced_2.wav")

    Create the HAAQI evaluator
    >>> ha = HaaqiV1()
    >>> ha.set_audiogram(audiogram)

    Set the reference signal
    >>> ha.set_reference(reference, sr_r)

    Process the enhanced signals
    >>> score_1, _, _ ,_ = ha.score(enhanced_1, sr_e_1)
    >>> score_2, _, _ ,_ = ha.score(enhanced_2, sr_e_2)

    ------------------------------------------------------------------------------------
    Example 2:
    For scoring different reference and enhanced signals with the same audiogram:

    Create the HAAQI evaluator
    >>> ha = HaaqiV1()
    >>> ha.set_audiogram(audiogram)

    Load enhanced and reference signals
    >>> reference_1, sr_r_1 = wavfile.read("reference_1.wav")
    >>> reference_2, sr_r_2 = wavfile.read("reference_2.wav")
    >>> enhanced_1, sr_e_1 = wavfile.read("enhanced_1.wav")
    >>> enhanced_2, sr_e_2 = wavfile.read("enhanced_2.wav")

    Process the reference and enhanced signals
    >>> score_1 = ha.process(reference_1, sr_r_1, enhanced_1, sr_e_1)
    >>> score_2 = ha.process(reference_2, sr_r_2, enhanced_2, sr_e_2)
    """

    HAAQI_AUDIOGRAM_FREQUENCIES: Final = np.array([250, 500, 1000, 2000, 4000, 6000])
    EAR_SAMPLE_RATE: Final = 24000
    SMALL_VALUE: Final = 1e-30

    def __init__(
        self,
        equalisation: int = 1,
        num_bands: int = 32,
        silence_threshold: float = 2.5,
        add_noise: float = 0.0,
        segment_size: int = 8,
        n_cepstral_coef: int = 6,
        segment_covariance: int = 16,
        ear_model_kwargs: dict | None = None,
    ) -> None:
        """Initialise HAAQI evaluator.

        Args:
            equalisation (int, optional): purpose for the calculation:
                 0=intelligibility: reference is normal hearing and must not
                   include NAL-R EQ
                 1=quality: reference does not include NAL-R EQ
                 2=quality: reference already has NAL-R EQ applied.
                 Defaults to 1.
            num_bands (int, optional): number of auditory bands. Defaults to 32.
            silence_threshold (float, optional): threshold in dB SPL for
                silence. Defaults to 2.5.
            add_noise (float, optional): level of noise to add to the signal.
                Defaults to 0.0.
            segment_size (int, optional): segment size in milliseconds for
                envelope smoothing. Defaults to 8.
            n_cepstral_coef (int, optional): number of cepstral coefficients.
                Defaults to 6.
            segment_covariance (int, optional): segment size in milliseconds for
                covariance calculation. Defaults to 16.
            ear_model_kwargs (dict, optional): keyword arguments for the Ear
                model. Defaults to None.
        """
        if ear_model_kwargs is None:
            ear_model_kwargs = {}

        self.num_bands = num_bands
        self.silence_threshold = silence_threshold
        self.add_noise = add_noise
        self.segment_size = segment_size
        self.n_cepstral_coef = n_cepstral_coef
        self.segment_covariance = segment_covariance

        self.audiogram: Audiogram | None = None
        self.level1: float = 65.0

        earmodel_kwa = {}
        earmodel_kwa["m_delay"] = (
            ear_model_kwargs["m_delay"] if "m_delay" in ear_model_kwargs else 1
        )
        earmodel_kwa["shift"] = (
            ear_model_kwargs["shift"] if "shift" in ear_model_kwargs else None
        )
        earmodel_kwa["signals_same_size"] = (
            ear_model_kwargs["signals_same_size"]
            if "signals_same_size" in ear_model_kwargs
            else True
        )

        self.ear_model = Ear(
            equalisation=equalisation, num_bands=self.num_bands, **earmodel_kwa
        )

        # Set reference variables that will be reused in different methods
        self.reference_smooth: ndarray = np.empty(0)
        self.reference_basilar_membrane: ndarray = np.empty(0)
        self.reference_sl: ndarray = np.empty(0)
        self.reference_computed: ndarray = np.empty(0)
        self.reference_cep: ndarray = np.empty(0)
        self.reference_linear_magnitude: ndarray = np.empty(0)

        # Variables for silence threshold
        self.segments_above_threshold: int = 0
        self.index_above_threshold: ndarray = np.empty(0)

        # Mel cepstrum basis functions
        freq = np.arange(self.n_cepstral_coef)
        k = np.arange(self.num_bands)
        basis = np.cos(np.outer(k, freq) * np.pi / (self.num_bands - 1))
        self.cepm = basis / np.linalg.norm(basis, axis=0, keepdims=True)

        # For melcor9
        # Modulation filter bands, segment size is 8 msec
        # 8 bands covering 0 to 125 Hz
        self.edge = [4.0, 8.0, 12.5, 20.0, 32.0, 50.0, 80.0]
        # Number of modulation filter bands
        self.n_modulation_filter_bands = 1 + len(self.edge)

    def reset_reference(self) -> None:
        """Reset the reference signal variables used by
        the enhanced signal.
        """
        self.ear_model.reset_reference()
        self.reference_smooth: ndarray = np.empty(0)
        self.reference_basilar_membrane: ndarray = np.empty(0)
        self.reference_sl: ndarray = np.empty(0)
        self.reference_computed: ndarray = np.empty(0)
        self.reference_cep: ndarray = np.empty(0)
        self.reference_linear_magnitude: ndarray = np.empty(0)

        self.segments_above_threshold: int = 0
        self.index_above_threshold: ndarray = np.empty(0)

    def set_audiogram(self, audiogram: Audiogram) -> None:
        """Set the audiogram.

        Args:
            audiogram (Audiogram): audiogram to set.
        """
        self.ear_model.set_audiogram(audiogram)
        self.audiogram = audiogram

    def process(
        self,
        reference: ndarray,
        reference_sample_rate: float,
        enhanced: ndarray,
        enhanced_sample_rate: float,
        level1: float = 65.0,
    ) -> float:
        """Process the reference and enhanced signals.

        This method is a wrapper for the set_reference and score methods.
        It assumes that the references need to be recomputed every time.
        Recommended for when processing several different signals with the same
        audiogram.

        Args:
            reference (ndarray): reference signal.
            reference_sample_rate (float): reference signal sample rate.
            enhanced (ndarray): enhanced signal.
            enhanced_sample_rate (float): enhanced signal sample rate.
            level1 (float, optional): level1. Defaults to 65.0.
            keep_reference (bool, optional): keep reference. Defaults to False.

        Returns:
            HAAQI_score: float
            Nonlinear_model: float
            Linear_model: float
            raw_data: ndarray
        """
        self.reset_reference()
        self.set_reference(reference, reference_sample_rate, level1)
        score, _, _, _ = self.score(enhanced, enhanced_sample_rate)
        return score

    def set_reference(
        self, reference: ndarray, sample_rate: float, level1: float = 65.0
    ) -> None:
        """Set the reference signal.

        Method computes the ear model and other variables from the reference
         signal that will be reused in the ```score``` method.

        ```set_audiogram``` must be called before this method.

        Args:
            reference (ndarray): reference signal.
            sample_rate (float): reference signal sample rate.
            level1 (float, optional): level1. Defaults to 65.0.

        Raises:
            ValueError: Audiogram must be set before calling this method.

        Example:
            >>> ha = HAAQI_V1()
            >>> ha.set_audiogram(audiogram)
            >>> ha.set_reference(reference, sr)
        """
        if self.audiogram is None:
            raise ValueError("Audiogram must be set before calling this method.")

        if sample_rate != self.EAR_SAMPLE_RATE:
            logger.warning(
                "Sample rate of the reference signal is different from the "
                "ear model sample rate. Resampling."
            )
            reference = resample(reference, sample_rate, self.EAR_SAMPLE_RATE)

        # Compute Ear model
        (
            reference_db,
            self.reference_basilar_membrane,
            self.reference_sl,
        ) = self.ear_model.process_reference(reference, level1)

        # ***********************
        # Save reference smooth for reuse in melcor9
        self.reference_smooth = self.env_smooth(reference_db)

        self.ref_noise = np.random.standard_normal(self.reference_smooth.shape)
        self.enh_noise = np.random.standard_normal(self.reference_smooth.shape)

        # ***********************
        # Save reference_linear_magnitude for reuse in diff_spectrum
        self.reference_linear_magnitude = 10 ** (self.reference_sl / 20)
        reference_sum = np.sum(self.reference_linear_magnitude)
        # Loudness sum = 1 (arbitrary amplitude, proportional to sones)
        self.reference_linear_magnitude /= reference_sum

        # ***********************
        # Save index_above_threshold and segments_above_threshold
        # for reuse in melcor9

        # Find the segments that lie sufficiently above the quiescent rate
        # Convert envelope dB to linear (specific loudness)
        reference_linear = 10 ** (self.reference_smooth / 20)
        # Proportional to loudness in sones
        reference_sum = np.sum(reference_linear, 0) / self.num_bands
        # Convert back to dB (loudness in phons)
        reference_sum = 20 * np.log10(reference_sum)
        # Identify those segments above threshold
        self.index_above_threshold = np.where(reference_sum > self.silence_threshold)[0]
        # Number of segments above threshold
        self.segments_above_threshold = self.index_above_threshold.shape[0]

    def score(
        self,
        enhanced: ndarray,
        sample_rate: float,
    ) -> tuple[float, float, float, ndarray]:
        """Score the enhanced signal.

        Args:
            enhanced (ndarray): enhanced signal.
            sample_rate (float): enhanced signal sample rate.

        Returns:
            HAAQI_score: float
            Nonlinear_model: float
            Linear_model: float
            raw_data: ndarray

        Example:
            >>> ha = HAAQI_V1()
            >>> ha.set_audiogram(audiogram)
            >>> ha.set_reference(reference, sr)
            >>> score, nonlinear, linear, raw = ha.score(enhanced, sr)
        """
        if sample_rate != self.EAR_SAMPLE_RATE:
            logger.warning(
                "Sample rate of the enhanced signal is different from the "
                "ear model sample rate. Resampling."
            )
            enhanced = resample(enhanced, sample_rate, self.EAR_SAMPLE_RATE)

        (
            enhanced_db,
            enhanced_basilar_membrane,
            enhanced_sl,
        ) = self.ear_model.process_enhanced(enhanced)

        linear_model, d_loud, d_norm = self.linear_model(enhanced_sl)
        (
            nonlinear_model,
            mel_cepstral_high,
            basilar_membrane_sync5,
        ) = self.non_linear_model(enhanced_db, enhanced_basilar_membrane)

        # Combined model
        combined_model = (
            0.336 * nonlinear_model
            + 0.001 * linear_model
            + 0.501 * (nonlinear_model**2)
            + 0.161 * (linear_model**2)
        )  # Polynomial sum

        # Raw data
        raw = [mel_cepstral_high, basilar_membrane_sync5, d_loud, d_norm]

        return combined_model, nonlinear_model, linear_model, np.array(raw)

    def linear_model(self, enhanced_sl: ndarray) -> tuple[float, float, float]:
        """
        Compute the linear model.

        Args:
            enhanced_sl (ndarray): enhanced signal spectrum in dB SL

        Returns:
            linear_model: float
            d_loud: float
            d_norm: float
        """
        dloud_stats, dnorm_stats, _ = self.spectrum_diff(enhanced_sl)

        # Loudness difference std
        d_loud = dloud_stats[1] / 2.5
        d_loud = 1.0 - d_loud
        d_loud = min(d_loud, 1)
        d_loud = max(d_loud, 0)

        # Dnorm:std
        # Slope difference std
        d_norm = dnorm_stats[1] / 25
        d_norm = 1.0 - d_norm
        d_norm = min(d_norm, 1)
        d_norm = max(d_norm, 0)

        return 0.329 * d_loud + 0.671 * d_norm, float(d_loud), float(d_norm)

    def non_linear_model(
        self, enhanced_db: ndarray, enhanced_basilar_membrane: ndarray
    ) -> tuple[float, float, float]:
        """
        Compute the non-linear model.

        Args:
            enhanced_db (ndarray): enhanced signal spectrum in dB SL
            enhanced_basilar_membrane (ndarray): enhanced basilar membrane movement

        Returns:
            nonlinear_model: float
            mel_cepstral_high: float
            basilar_membrane_sync5: float
        """
        signal_cross_covariance, reference_mean_square, _ = self.bm_covary(
            self.reference_basilar_membrane,
            enhanced_basilar_membrane,
        )

        enhanced_smooth = self.env_smooth(enhanced_db)
        _, _, mel_cepstral_high, _ = self.melcor9(enhanced_smooth)

        _, ihc_sync_covariance = self.ave_covary2(
            signal_cross_covariance, reference_mean_square
        )

        # Ave segment coherence with IHC loss of sync
        basilar_membrane_sync5 = ihc_sync_covariance[4]

        # Construct the models
        # Nonlinear model - Combined envelope and TFS
        return (
            0.754 * (mel_cepstral_high**3) + 0.246 * basilar_membrane_sync5,
            mel_cepstral_high,
            basilar_membrane_sync5,
        )

    def env_smooth(self, envelopes: np.ndarray) -> ndarray:
        """
        Function to smooth the envelope returned by the cochlear model. The
        envelope is divided into segments having a 50% overlap. Each segment is
        windowed, summed, and divided by the window sum to produce the average.
        A raised cosine window is used. The envelope sub-sampling frequency is
        2*(1000/segsize).

        Arguments:
            envelopes (np.ndarray): matrix of envelopes in each of the auditory bands
        Returns:
            smooth: matrix of subsampled windowed averages in each band
        """

        # Compute the window
        # Segment size in samples
        if self.segment_size == 8:
            n_samples = 192
            wsum = 95.5
            nhalf = 96
            halfsum = 47.75

            window = np.hanning(n_samples)
            halfwindow = window[nhalf:n_samples]

        else:
            n_samples = int(
                np.around(self.segment_size * (0.001 * self.EAR_SAMPLE_RATE))
            )
            n_samples += n_samples % 2

            # Raised cosine von Hann window
            window = np.hanning(n_samples)
            # Sum for normalization
            wsum = np.sum(window)

            #  The first segment has a half window
            nhalf = int(n_samples / 2)

            halfwindow = window[nhalf:n_samples]
            halfsum = np.sum(halfwindow)

        # Number of segments and assign the matrix storage
        n_channels, npts = envelopes.shape
        nseg = int(
            1
            + np.floor(npts / n_samples)
            + np.floor((npts - n_samples / 2) / n_samples)
        )

        smooth = np.zeros((n_channels, nseg))

        #  Loop to compute the envelope in each frequency band
        for k in range(n_channels):
            # Extract the envelope in the frequency band
            r = envelopes[k, :]  # pylint: disable=invalid-name

            # The first (half) windowed segment
            nstart = 0
            smooth[k, 0] = (
                np.sum(r[nstart:nhalf] * halfwindow.conj().transpose()) / halfsum
            )

            # Loop over the remaining full segments, 50% overlap
            for n in range(1, nseg - 1):
                nstart = int(nstart + nhalf)
                nstop = int(nstart + n_samples)
                smooth[k, n] = (
                    np.sum(r[nstart:nstop] * window.conj().transpose()) / wsum
                )

            # The last (half) windowed segment
            nstart = nstart + nhalf
            nstop = nstart + nhalf
            smooth[k, nseg - 1] = (
                np.sum(r[nstart:nstop] * window[:nhalf].conj().transpose()) / halfsum
            )

        return smooth

    def melcor9(
        self,
        signal: ndarray,
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
            distorted (): subsampled distorted output signal envelope

        Returns:
            mel_cepstral_average (): average of the modulation correlations across
                analysis frequency bands and modulation frequency bands,
                basis functions 2 -6
            mel_cepstral_low (): average over the four lower mod freq bands, 0 - 20 Hz
            mel_cepstral_high (): average over the four higher mod freq bands,
                20 - 125 Hz
            mel_cepstral_modulation (): vector of cross-correlations by modulation
                frequency, averaged over analysis frequency band
        """
        _reference = self.reference_smooth[:, self.index_above_threshold]
        _reference += (
            self.add_noise * self.ref_noise[:, self.index_above_threshold]
        )  # np.random.standard_normal(_reference.shape)
        self.reference_cep = np.dot(
            self.cepm.T, _reference[:, : self.segments_above_threshold]
        )
        self.reference_cep -= np.mean(self.reference_cep, axis=1, keepdims=True)

        if self.segments_above_threshold <= 1:
            logger.warning(
                "Function melcor9: Signal below threshold, outputs set to 0."
            )
            return 0.0, 0.0, 0.0, np.zeros(self.n_modulation_filter_bands)

        # Remove the silent intervals
        signal = signal[:, self.index_above_threshold]

        # Add the low-level noise to the envelopes
        signal += (
            self.add_noise * self.enh_noise[:, self.index_above_threshold]
        )  # np.random.standard_normal(signal.shape)

        # Compute the mel cepstrum coefficients using only those segments
        # above threshold
        distorted_cep = np.dot(self.cepm.T, signal[:, : self.segments_above_threshold])
        distorted_cep -= np.mean(distorted_cep, axis=1, keepdims=True)

        # Envelope sampling parameters
        # Envelope sampling frequency in Hz
        sampling_freq = 1000.0 / (0.5 * self.segment_size)
        # Envelope Nyquist frequency
        nyquist_freq = 0.5 * sampling_freq

        # Design the linear-phase envelope modulation filters
        # Adjust filter length to sampling rate
        n_fir = np.around(128 * (nyquist_freq / 125))
        n_fir = int(2 * np.floor(n_fir / 2))  # Force an even filter length
        b = np.zeros((self.n_modulation_filter_bands, n_fir + 1))

        # LP filter 0-4 Hz
        b[0, :] = firwin(
            n_fir + 1, self.edge[0] / nyquist_freq, window="hann", pass_zero="lowpass"
        )
        # HP 80-125 Hz
        b[self.n_modulation_filter_bands - 1, :] = firwin(
            n_fir + 1,
            self.edge[self.n_modulation_filter_bands - 2] / nyquist_freq,
            window="hann",
            pass_zero="highpass",
        )
        # Bandpass filter
        for m in range(1, self.n_modulation_filter_bands - 1):
            b[m, :] = firwin(
                n_fir + 1,
                [self.edge[m - 1] / nyquist_freq, self.edge[m] / nyquist_freq],
                window="hann",
                pass_zero="bandpass",
            )

        mel_cepstral_cross_covar = self.melcor9_crosscovmatrix(
            b,
            self.n_modulation_filter_bands,
            self.n_cepstral_coef,
            self.segments_above_threshold,
            n_fir,
            self.reference_cep,
            distorted_cep,
        )

        mel_cepstral_average = np.sum(mel_cepstral_cross_covar[:, 1:], axis=(0, 1))
        mel_cepstral_average /= self.n_modulation_filter_bands * (
            self.n_cepstral_coef - 1
        )

        mel_cepstral_low = np.sum(mel_cepstral_cross_covar[:4, 1:])
        mel_cepstral_low /= 4 * (self.n_cepstral_coef - 1)

        mel_cepstral_high = np.sum(mel_cepstral_cross_covar[4:8, 1:])
        mel_cepstral_high /= 4 * (self.n_cepstral_coef - 1)

        mel_cepstral_modulation = np.mean(mel_cepstral_cross_covar[:, 1:], axis=1)

        return (
            mel_cepstral_average,
            mel_cepstral_low,
            mel_cepstral_high,
            mel_cepstral_modulation,
        )

    def melcor9_crosscovmatrix(
        self,
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
            b (ndarray): filter
            nmod (int): number of modulation filter bands
            nbasis (int): number of cepstral coefficients
            nsamp (int): number of samples above threshold
            nfir (int): filter length
            reference_cep (ndarray): mel cepstrum coefficients of the reference signal
            processed_cep (ndarray): mel cepstrum coefficients of the processed signal

        Returns:
            cross_covariance_matrix (ndarray):
        """
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
            mask = (reference_sum < self.SMALL_VALUE) | (
                processed_sum < self.SMALL_VALUE
            )
            cross_covariance_matrix[m, ~mask] = np.abs(xy[~mask]) / np.sqrt(
                reference_sum[~mask] * processed_sum[~mask]
            )

        return cross_covariance_matrix

    def spectrum_diff(self, processed_sl: ndarray) -> tuple[ndarray, ndarray, ndarray]:
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
        """

        # Convert the dB SL to linear magnitude values. Because of the auditory
        # filter bank, the OHC compression, and auditory threshold, the linear
        # values are closely related to specific loudness.
        processed_linear_magnitude = 10 ** (processed_sl / 20)

        # Normalize the level of the reference and degraded signals to have the
        # same loudness. Thus overall level is ignored while differences in
        # spectral shape are measured.
        processed_sum = np.sum(processed_linear_magnitude)
        processed_linear_magnitude /= processed_sum

        # Compute the spectrum difference
        dloud = np.zeros(3)
        diff_spectrum = (
            self.reference_linear_magnitude - processed_linear_magnitude
        )  # Difference in specific loudness in each band
        dloud[0] = np.sum(np.abs(diff_spectrum))
        dloud[1] = self.num_bands * np.std(diff_spectrum)  # Biased std: second moment
        dloud[2] = np.max(np.abs(diff_spectrum))

        # Compute the normalized spectrum difference
        dnorm = np.zeros(3)
        # Relative difference in specific loudness
        diff_normalised_spectrum = (
            self.reference_linear_magnitude - processed_linear_magnitude
        ) / (self.reference_linear_magnitude + processed_linear_magnitude)
        dnorm[0] = np.sum(np.abs(diff_normalised_spectrum))
        dnorm[1] = self.num_bands * np.std(diff_normalised_spectrum)
        dnorm[2] = np.max(np.abs(diff_normalised_spectrum))

        # Compute the slope difference
        dslope = np.zeros(3)
        reference_slope = (
            self.reference_linear_magnitude[1 : self.num_bands]
            - self.reference_linear_magnitude[0 : self.num_bands - 1]
        )
        processed_slope = (
            processed_linear_magnitude[1 : self.num_bands]
            - processed_linear_magnitude[0 : self.num_bands - 1]
        )
        diff_slope = reference_slope - processed_slope  # Slope difference
        dslope[0] = np.sum(np.abs(diff_slope))
        dslope[1] = self.num_bands * np.std(diff_slope)
        dslope[2] = np.max(np.abs(diff_slope))

        return dloud, dnorm, dslope

    def bm_covary(
        self,
        reference_basilar_membrane: ndarray,
        processed_basilar_membrane: ndarray,
    ) -> tuple[ndarray, ndarray, ndarray]:
        """
        Compute the cross-covariance (normalized cross-correlation) between the
        reference and processed signals in each auditory band. The signals are
        divided into segments  having 50% overlap.

        Arguments:
            reference_basilar_membrane (): Basilar Membrane movement, reference signal
            processed_basilar_membrane (): Basilar Membrane movement, processed signal

        Returns:
            signal_cross_covariance (np.array) : [nchan,nseg] of cross-covariance
                values.
            reference_mean_square (np.array) : [nchan,nseg] of MS input signal energy
                values.
            processed_mean_square (np.array) : [nchan,nseg] of MS processed signal
                energy values.

        Updates:
            James M. Kates, 28 August 2012.
            Output amplitude adjustment added, 30 october 2012.
            Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
        """

        # Lag for computing the cross-covariance
        lagsize = 1.0  # Lag (+/-) in msec
        maxlag = np.around(lagsize * (0.001 * self.EAR_SAMPLE_RATE))  # Lag in samples

        # Compute the segment size in samples
        nwin = int(np.around(self.segment_covariance * (0.001 * self.EAR_SAMPLE_RATE)))

        nwin += nwin % 2 == 1  # Force window length to be even
        window = np.hanning(nwin).conj().transpose()  # Raised cosine von Hann window

        # compute inverted Window autocorrelation
        win_corr = correlate(window, window, "same")
        start_sample = int(len(window) / 2 - maxlag)
        end_sample = int(maxlag + 1 + len(window) / 2)
        if start_sample < 0:
            raise ValueError("segment size too small")
        win_corr = 1 / win_corr[start_sample:end_sample]
        win_sum2 = 1.0 / np.sum(window**2)  # Window power, inverted

        # The first segment has a half window
        nhalf = int(nwin / 2)
        half_window = window[nhalf:nwin]
        half_corr = correlate(half_window, half_window, "same")
        start_sample = int(len(half_window) / 2 - maxlag)
        end_sample = int(maxlag + 1 + len(half_window) / 2)
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
            correlation = correlate(reference_seg, processed_seg, "same")
            correlation = correlation[
                int(len(reference_seg) / 2 - maxlag) : int(
                    maxlag + 1 + len(reference_seg) / 2
                )
            ]

            unbiased_cross_correlation = np.max(np.abs(correlation * half_corr))
            if (ref_mean_square > self.SMALL_VALUE) and (
                proc_mean_squared > self.SMALL_VALUE
            ):
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
                correlation = correlate(reference_seg, processed_seg, "same")
                correlation = correlation[
                    int(len(reference_seg) / 2 - maxlag) : int(
                        maxlag + 1 + len(reference_seg) / 2
                    )
                ]

                unbiased_cross_correlation = np.max(np.abs(correlation * win_corr))
                if (ref_mean_square > self.SMALL_VALUE) and (
                    proc_mean_squared > self.SMALL_VALUE
                ):
                    # Normalize cross-covariance
                    signal_cross_covariance[
                        k, n
                    ] = unbiased_cross_correlation / np.sqrt(
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
            processed_seg = (
                y[nstart:nstop] * window[0:nhalf]
            )  # Window the processed signal
            reference_seg = reference_seg - np.mean(reference_seg)  # Make 0-mean
            processed_seg = processed_seg - np.mean(processed_seg)
            # Normalize signal MS value by the window
            ref_mean_square = np.sum(reference_seg**2) * halfsum2
            proc_mean_squared = np.sum(processed_seg**2) * halfsum2

            correlation = correlate(reference_seg, processed_seg, "same")
            correlation = correlation[
                int(len(reference_seg) / 2 - maxlag) : int(
                    maxlag + 1 + len(reference_seg) / 2
                )
            ]

            unbiased_cross_correlation = np.max(np.abs(correlation * half_corr))
            if (ref_mean_square > self.SMALL_VALUE) and (
                proc_mean_squared > self.SMALL_VALUE
            ):
                # Normalized cross-covariance
                signal_cross_covariance[
                    k, nseg - 1
                ] = unbiased_cross_correlation / np.sqrt(
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
        self,
        signal_cross_covariance: np.ndarray,
        reference_signal_mean_square: np.ndarray,
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
            ihc_sync_covariance (): cross-covariance array, 6 different weightings for
                loss of IHC synchronization at high frequencies:
                  LP Filter Order     Cutoff Freq, kHz
                         1              1.5
                         3              2.0
                         5              2.5, 3.0, 3.5, 4.0
        """

        # Array dimensions
        n_channels = signal_cross_covariance.shape[0]

        # Initialize the LP filter for loss of IHC synchronization
        # Center frequencies in Hz on an ERB scale
        _center_freq = self.ear_model.center_freq
        # Default LP filter order
        if lp_filter_order is None:
            lp_filter_order = np.array([1, 3, 5, 5, 5, 5])
        # Default cutoff frequencies in Hz
        if freq_cutoff is None:
            freq_cutoff = 1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

        fc2p = (
            np.atleast_2d(freq_cutoff ** (2 * lp_filter_order))
            .repeat(n_channels, axis=0)
            .T
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
        index = np.argwhere(reference_mean > self.silence_threshold).T
        if index.size != 1:
            index = index.squeeze()
        nseg = index.shape[0]  # Number of segments above threshold

        # Exit if not enough segments above zero
        if nseg <= 1:
            logger.warning(
                "Function AveCovary2: Ave signal below threshold, outputs set to 0."
            )
            return 0, np.zeros(6)

        # Remove the silent segments
        signal_cross_covariance = signal_cross_covariance[:, index]
        signal_rms = signal_rms[:, index]

        # Compute the time-frequency weights. The weight=1 if a segment in a
        # frequency band is above threshold, and weight=0 if below threshold.
        weight = np.zeros((n_channels, nseg))  # No IHC synchronization roll-off
        weight[signal_rms > self.silence_threshold] = 1

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
        mask = signal_rms > self.silence_threshold
        fsync3d = np.repeat(fsync[..., None], nseg, axis=2)
        wsync[:, mask] = fsync3d[:, mask]

        # Sum the weighted covariance values
        # Sum of weighted time-freq tiles
        csum = np.sum(np.sum(weight * signal_cross_covariance))

        wsum = np.sum(np.sum(weight))  # Total number of tiles above threshold

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

    def __str__(self):
        return "HAAQI V1"
