"""Implementation PyHAAQI evaluator."""

import logging
from typing import Final

import numpy as np
from numpy import ndarray
from scipy.signal import firwin

from clarity.evaluator.ha.earmodel import Ear
from clarity.utils.audiogram import Audiogram

logger = logging.getLogger(__name__)


class HAAQI:
    """HAAQI evaluator class."""

    HAAQI_AUDIOGRAM_FREQUENCIES: Final = np.array([250, 500, 1000, 2000, 4000, 6000])

    def __init__(self):
        """Initialise HAAQI evaluator."""
        self.reference_ear = Ear(
            itype=1,
        )
        self.audiogram = None
        self.level1 = None
        self.reference_smooth = None
        self.reference_basilar_membrane = None
        self.reference_sl = None
        self.reference_sum = None
        self.index = None
        self.segments_above_threshold = None
        self.cepm = None

        self.silence_threshold = 2.5
        self.segment_covariance = 16
        self.add_noise = 0.0

    def set_reference_signal(
        self, signal: ndarray, sample_rate: float, audiogram: Audiogram, level1: float
    ):
        """Set the reference signal."""
        self.reference_ear.set_audiogram(audiogram)
        self.audiogram = audiogram
        self.level1 = level1

        (
            reference_db,
            self.reference_basilar_membrane,
            self.reference_sl,
        ) = self.reference_ear.process(signal, sample_rate, level1)

        self.reference_smooth = self.env_smooth(reference_db)
        reference_linear = 10 ** (self.reference_smooth / 20)

        nbands = self.reference_smooth.shape[0]
        reference_sum = np.sum(reference_linear, 0) / nbands

        # Mel cepstrum basis functions (mel cepstrum because of auditory bands)
        n_cepstral_coef = 6
        freq = np.arange(n_cepstral_coef)
        k = np.arange(nbands)
        basis = np.cos(np.outer(k, freq) * np.pi / (nbands - 1))
        self.cepm = basis / np.linalg.norm(basis, axis=0, keepdims=True)

        self.reference_sum = 20 * np.log10(reference_sum)
        self.index = np.where(self.reference_sum > self.silence_threshold)[0]
        # Number of segments above threshold
        self.segments_above_threshold = self.index.shape[0]

        _reference = self.reference_smooth[:, self.index]
        _reference += self.add_noise, * np.random.standard_normal(_reference.shape)
        reference_cep = np.dot(self.cepm.T, _reference[:, :self.segments_above_threshold])
        reference_cep -= np.mean(reference_cep, axis=1, keepdims=True)

    def evaluate(
        self,
        processed_signal: ndarray,
        sample_rate: float,
    ):
        """Evaluate HAAQI."""
        processed_ear = Ear(
            itype=1,
        )
        processed_ear.set_audiogram(self.audiogram)
        processed_db, processed_basilar_membrane, processed_sl = processed_ear.process(
            processed_signal, sample_rate, self.level1
        )
        processed_smooth = self.env_smooth(processed_db)

        # 8 modulation freq bands
        _, _, mel_cepstral_high, _ = self.melcor9(
            self.reference_smooth,
            processed_smooth,
        )

        dloud_stats, dnorm_stats, _ = self.spectrum_diff(
            self.reference_sl, processed_sl
        )

        signal_cross_covariance, reference_mean_square, _ = self.bm_covary(
            self.reference_basilar_membrane,
            processed_basilar_membrane,
            self.segment_covariance,
            sample_rate,
        )

        _, ihc_sync_covariance = self.ave_covary2(
            signal_cross_covariance, reference_mean_square, self.silence_threshold
        )
        # Ave segment coherence with IHC loss of sync
        basilar_membrane_sync5 = ihc_sync_covariance[4]

        d_loud = dloud_stats[1] / 2.5  # Loudness difference std
        d_loud = 1.0 - d_loud  # 1=perfect, 0=bad
        d_loud = min(d_loud, 1)
        d_loud = max(d_loud, 0)

        # Dnorm:std
        d_norm = dnorm_stats[1] / 25  # Slope difference std
        d_norm = 1.0 - d_norm  # 1=perfect, 0=bad
        d_norm = min(d_norm, 1)
        d_norm = max(d_norm, 0)

        # Construct the models
        # Nonlinear model - Combined envelope and TFS
        nonlinear_model = (
            0.754 * (mel_cepstral_high**3) + 0.246 * basilar_membrane_sync5
        )

        # Linear model
        linear_model = 0.329 * d_loud + 0.671 * d_norm

        # Combined model
        combined_model = (
            0.336 * nonlinear_model
            + 0.001 * linear_model
            + 0.501 * (nonlinear_model**2)
            + 0.161 * (linear_model**2)
        )  # Polynomial sum

        # Raw data
        raw = [mel_cepstral_high, basilar_membrane_sync5, d_loud, d_norm]

        return combined_model, nonlinear_model, linear_model, raw

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
        # segment_size = 8
        # n_samples = int(np.around(segment_size * (0.001 * sample_rate)))
        # n_samples += n_samples % 2
        n_samples = 192

        window = np.hanning(n_samples)  # Raised cosine von Hann window
        # wsum = np.sum(window)  # Sum for normalization
        wsum = 95.5
        #  The first segment has a half window
        # nhalf = int(n_samples / 2)
        nhalf = 96
        halfwindow = window[nhalf:n_samples]
        # halfsum = np.sum(halfwindow)
        halfsum = 47.75
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
        reference: ndarray,
        distorted: ndarray,
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

        Returns:
            mel_cepstral_average (): average of the modulation correlations across analysis
                frequency bands and modulation frequency bands, basis functions 2 -6
            mel_cepstral_low (): average over the four lower mod freq bands, 0 - 20 Hz
            mel_cepstral_high (): average over the four higher mod freq bands, 20 - 125 Hz
            mel_cepstral_modulation (): vector of cross-correlations by modulation
                frequency, averaged over analysis frequency band
        """
        segment_size = 8  # Segment size in ms
        # Processing parameters
        nbands = reference.shape[0]

        # Modulation filter bands, segment size is 8 msec
        # 8 bands covering 0 to 125 Hz
        edge = [4.0, 8.0, 12.5, 20.0, 32.0, 50.0, 80.0]
        n_modulation_filter_bands = 1 + len(edge)  # Number of modulation filter bands

        # Exit if not enough segments above zero
        mel_cepstral_average = 0.0
        mel_cepstral_low = 0.0
        mel_cepstral_high = 0.0
        mel_cepstral_modulation = np.zeros(n_modulation_filter_bands)
        if self.segments_above_threshold <= 1:
            logger.warning(
                "Function melcor9: Signal below threshold, outputs set to 0."
            )
            return (
                mel_cepstral_average,
                mel_cepstral_low,
                mel_cepstral_high,
                mel_cepstral_modulation,
            )

        # Remove the silent intervals
        _distorted = distorted[:, self.index]
        # Add the low-level noise to the envelopes
        _distorted += self.add_noise * np.random.standard_normal(_distorted.shape)
        # Compute the mel cepstrum coefficients using only those segments
        # above threshold
        distorted_cep = np.dot(self.cepm.T, _distorted[:, :self.segments_above_threshold])
        distorted_cep -= np.mean(distorted_cep, axis=1, keepdims=True)

        # Envelope sampling parameters
        # Envelope sampling frequency in Hz
        sampling_freq = 1000.0 / (0.5 * segment_size)
        nyquist_freq = 0.5 * sampling_freq  # Envelope Nyquist frequency

        # Design the linear-phase envelope modulation filters
        # Adjust filter length to sampling rate
        n_fir = np.around(128 * (nyquist_freq / 125))
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

    def __str__(self):
        return "HAAQI"
