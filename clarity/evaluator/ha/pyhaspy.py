"""Implementation of HASPI intelligibility Index."""
from __future__ import annotations

import logging
from math import floor
from typing import Final

import numpy as np
from numpy import ndarray
from scipy.signal import convolve, convolve2d, correlate, firwin

from clarity.evaluator.ha.earmodel import Ear
from clarity.utils.audiogram import Audiogram
from clarity.utils.signal_processing import resample

logger = logging.getLogger(__name__)


class HaspiV2:
    """HASPI evaluator class.

    Compute the HASPI intelligibility index using the
    auditory model followed by computing the envelope cepstral
    correlation and BM vibration high-level covariance. The reference
    signal presentation level for NH listeners is assumed to be 65 dB
    SPL. The same model is used for both normal and impaired hearing. This
    version of HASPI uses a modulation filterbank followed by an ensemble of
    neural networks to compute the estimated intelligibility.


    """

    HAAQI_AUDIOGRAM_FREQUENCIES: Final = np.array([250, 500, 1000, 2000, 4000, 6000])
    EAR_SAMPLE_RATE: Final = 24000
    SMALL_VALUE: Final = 1e-30

    def __init__(
        self,
        equalisation: int = 1,
        num_bands: int = 32,
        f_lp: float = 320.0,
        silence_threshold: float = 2.5,
        add_noise: float = 0.0,
        segment_size: int = 8,
        n_cepstral_coef: int = 6,
        segment_covariance: int = 16,
        ear_model_kwargs: dict | None = None,
    ) -> None:
        if ear_model_kwargs is None:
            ear_model_kwargs = {}

        self.num_bands = num_bands
        self.f_lp = f_lp
        self.audiogram: Audiogram | None = None
        self.level1: float = 65.0

        # Initialise Ear Model
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

        # for cepstral coefficients
        self.nbasis = 6  # Use 6 basis functions
        self.thresh_cep = 2.5  # Silence threshold in dB SL
        self.thresh_nerve = 0.1  # Dither in dB RMS to add to envelope signals

    def reset_reference(self) -> None:
        """Reset the reference signal."""

        self.ear_model.reset_reference()

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
        reference_sample_rate: int,
        processed: ndarray,
        processed_sample_rate: int,
        level1: float = 65.0,
    ) -> float:
        self.reset_reference()
        self.set_reference(reference, reference_sample_rate, level1)
        score, _, _, _ = self.score(processed, processed_sample_rate)
        return score

    def set_reference(
        self, reference: ndarray, sample_rate: int, level1: float = 65.0
    ) -> None:
        """Set the reference signal.

        Method computes the ear model and other variables from the reference
         signal that will be reused in the ```score``` method.

        ```set_audiogram``` must be called before this method.

        Args:
            reference (ndarray): Reference signal.
            reference_sample_rate (int): Reference signal sample rate.
            level1 (float): Reference signal level in dB SPL.

        Raises:
            ValueError: Audiogram must be set before calling this method.

        Example:
            >>> ha = HaspiV2()
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

        (
            reference_env,
            _,
            _,
        ) = self.ear_model.process_reference(reference, level1)

        # LP filter and subsample the envelope
        reference_lp = self.env_filter(reference_env, self.f_lp, self.f_lp * 8.0)

        ################################################################################
        # Precompute the cepstral coefficients for the reference signal
        # Mel cepstrum basis functions
        freq = np.arange(0, self.nbasis)
        k = np.arange(0, self.num_bands)
        self.cepm = np.zeros((self.num_bands, self.nbasis))

        for n in range(self.nbasis):
            basis = np.cos(freq[n] * np.pi * k / (self.num_bands - 1))
            self.cepm[:, n] = basis / np.sqrt(np.sum(basis**2))

        # Find the reference segments that lie sufficiently above the quiescent rate
        x_linear = 10 ** (
            reference_lp / 20
        )  # Convert envelope dB to linear (specific loudness)
        # Proportional to loudness in sones
        xsum = np.sum(x_linear, 1) / self.num_bands
        xsum = 20 * np.log10(xsum)  # Convert back to dB (loudness in phons)
        self.num_seg_avobe_threshold = np.where(xsum > self.thresh_cep)[
            0
        ]  # Identify those segments above threshold
        nsamp = len(self.num_seg_avobe_threshold)  # Number of segments above threshold

        # Exit if not enough segments above zero
        if nsamp <= 1:
            raise ValueError("Signal below threshold")

        # Compute the cepstral coefficients as a function of subsampled time
        reference_cep = self.cepstral_correlation_coef(reference_lp)

        ###############################################################################

        # Cepstral coefficients filtered at each modulation rate
        # Band center frequencies [2, 6, 10, 16, 25, 40, 64, 100, 160, 256] Hz
        # Band edges [0, 4, 8, 12.5, 20.5, 30.5, 52.4, 78.1, 128, 200, 328] Hz
        reference_mod = self.fir_modulation_filter(reference_cep, self.f_lp * 8.0)

    def env_filter(self, signal: ndarray, filter_cutoff: float, freq_sub_sample: float):
        # Check the filter design parameters
        if freq_sub_sample > self.EAR_SAMPLE_RATE:
            raise ValueError("upsampling rate too high.")

        if filter_cutoff > 0.5 * freq_sub_sample:
            raise ValueError("LP cutoff frequency too high.")

        nrow = signal.shape[0]  # number of rows
        ncol = signal.shape[1]  # number of columnts
        if ncol > nrow:
            signal = signal.T

        nsamp = signal.shape[0]

        # Compute the lowpass filter length in samples to give -3 dB at fcut Hz
        tfilt = 1000 * (1 / filter_cutoff)  # filter length in ms
        tfilt = 0.7 * tfilt  # Empirical adjustment to the filter length
        nfilt = round(0.001 * tfilt * self.EAR_SAMPLE_RATE)  # Filter length in samples
        nhalf = floor(nfilt / 2)
        nfilt = int(2 * nhalf)  # Force an even filter length

        # Design the FIR LP filter using a von Hann window to ensure that there are no
        # negative envelope values. The MATLAB code uses the hanning() function, which
        # returns the Hann window without the first and last zero-weighted window samples,
        # unlike np.hann and scipy.signal.windows.hann; the code below replicates this
        # behaviour
        window = 0.5 * (
            1 - np.cos(2 * np.pi * np.arange(1, nfilt / 2 + 1) / (nfilt + 1))
        )
        benv = np.concatenate((window, np.flip(window)))
        benv = benv / np.sum(benv)

        signal_env = convolve2d(signal, np.expand_dims(benv, 1), "full")
        signal_env = signal_env[nhalf : nhalf + nsamp, :]

        space = floor(self.EAR_SAMPLE_RATE / freq_sub_sample)
        index = np.arange(0, nsamp, space)

        return signal_env[index, :]

    def cepstral_correlation_coef(
        self,
        signal_db: ndarray,
    ) -> ndarray:
        """
        Compute the cepstral correlation coefficients between the reference signal
        and the distorted signal log envelopes. The silence portions of the
        signals are removed prior to the calculation based on the envelope of the
        reference signal. For each time sample, the log spectrum in dB SL is
        fitted with a set of half-cosine basis functions. The cepstral coefficients
        then form the input to the cepstral correlation
        calculation.

        Args:
            signal_db (): subsampled signal envelope in dB SL in each band

        Returns:
            ndarray: cepstral correlation coefficients
        """

        # Remove the silent samples
        signal_db = signal_db[self.num_seg_avobe_threshold, :]

        # Add low-level noise to provide IHC firing jitter
        signal_db = self.add_noise(signal_db, self.thresh_nerve)

        # Compute the mel cepstrum coefficients using only those samples above threshold
        signal_db = signal_db @ self.cepm

        # Remove the average value from the cepstral coefficients. The cepstral
        # cross-correlation will thus be a cross-covariance, and there is no effect of the
        # absolute signal level in dB.
        for n in range(self.nbasis):
            x = signal_db[:, n]
            x = x - np.mean(x)
            signal_db[:, n] = x

        return signal_db

    @staticmethod
    def add_noise(signal_db: ndarray, thresh_db: float) -> ndarray:
        """
        Add independent random Gaussian noise to the subsampled signal envelope
        in each auditory frequency band.

        Args:
            signal_db (): subsampled envelope in dB re:auditory threshold
            thresh_db (): additive noise RMS level (in dB)

        Returns:
            ndarray: signal with added noise
        """
        # Additive noise sequence
        # Gaussian noise with RMS=1, then scaled
        noise = thresh_db * np.random.standard_normal(signal_db.shape)

        # Add the noise to the signal envelope
        return signal_db + noise
