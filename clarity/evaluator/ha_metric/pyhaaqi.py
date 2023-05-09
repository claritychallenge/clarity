"""Module with the HAAQI metric implementation."""
from __future__ import annotations

# pylint: disable=import-error
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import convolve, correlate, firwin

from clarity.evaluator.ha_metric.ear_model import EarModel
from clarity.utils.signal_processing import compute_rms

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class HAAQI:
    """
    Compute the HAAQI music quality index using the auditory model followed by
    computing the envelope cepstral correlation and Basilar Membrane vibration
    average short-time coherence signals.

    The reference signal presentation level for NH listeners is assumed
    to be 65 dB SPL. The same model is used for both normal and
    impaired hearing.
    """

    def __init__(
        self,
        signal_sample_rate: float = 44100.0,
        ear_model_sample_rate: float = 24000.0,
        equalisation: int = 1,
        level1: float = 65.0,
        silence_threshold: float = 2.5,
        add_noise: float = 0.0,
        segment_covariance: int = 16,
        segment_size: int = 8,
        earmodel_kwards: dict | None = None,
    ):
        """
        Constructor

        Args:
            signal_sample_rate (float): Sampling rate in Hz for reference
                and processed signal. Default: 44100
            ear_model_sample_rate (float): Sampling rate in Hz for the ear model.
                Haaqi assumes that both signals have the same sampling rate.
                This parameter will be used to resample the signal if needed.
                Note that will override the `target_freq` parameter if passed
                as part of the `earmodel_kwards` params.
                Default: 24000
            equalisation (int): hearing loss equalisation mode for reference signal:
                1 = no EQ has been provided, the function will add NAL-R
                2 = NAL-R EQ has already been added to the reference signal
            level1 (int): Optional input specifying level in dB SPL that corresponds
                to a signal RMS = 1. Default is 65 dB SPL if argument not provided.
                Default: 65
            silence_threshold (float): Silence threshold sum across bands,
                dB above auditory threshold. Default : 2.5
            add_noise (float): Additive noise dB SL to condition cross-covariances.
                Defaults to 0.0
            segment_covariance (int): Number of segments to compute the covariance
                Default: 16
            segment_size (int): Size of the window to smooth the envelope
                Default: 8
            earmodel_kwards (dict | None): kwargs for the EarModel class. See
                clarity/evaluator/ha_metric/ear_model.py for more information.
        """
        self.sample_rate = signal_sample_rate
        self.level1 = level1
        self.silence_threshold = silence_threshold
        self.add_noise = add_noise
        self.segment_covariance = segment_covariance
        self.segment_size = segment_size
        earmodel_kwards = earmodel_kwards or {}
        if "target_freq" not in earmodel_kwards.keys():
            earmodel_kwards["target_freq"] = ear_model_sample_rate
        self.ear_model = EarModel(equalisation=equalisation, **earmodel_kwards)
        self.n_samples = 0
        self.wsum = 0
        self.window = None

    def compute(
        self, reference: ndarray, processed: ndarray, hearing_loss: ndarray
    ) -> tuple[float, float, float, list[float]]:
        (
            reference_db,
            reference_basilar_membrane,
            processed_db,
            processed_basilar_membrane,
            reference_sl,
            processed_sl,
            self.sample_rate,
        ) = self.ear_model.compute(
            reference=reference,
            reference_freq=self.sample_rate,
            processed=processed,
            processed_freq=self.sample_rate,
            hearing_loss=hearing_loss,
            level1=self.level1,
        )

        # Envelope and long-term average spectral features
        # Smooth the envelope outputs: 250 Hz sub-sampling rate
        # Averaging segment size in msec
        reference_smooth = self.env_smooth(reference_db)
        processed_smooth = self.env_smooth(processed_db)

        # Mel cepstrum correlation after passing through modulation filterbank
        # 8 modulation freq bands
        _, _, mel_cepstral_high, _ = self.melcor9(reference_smooth, processed_smooth)

        # Linear changes in the long-term spectra
        # dloud  vector: [sum abs diff, std dev diff, max diff] spectra
        # dnorm  vector: [sum abs diff, std dev diff, max diff] norm spectra
        # dslope vector: [sum abs diff, std dev diff, max diff] slope
        dloud_stats, dnorm_stats, _ = self.spectrum_diff(reference_sl, processed_sl)

        # Temporal fine structure (TFS) correlation measurements
        # Compute the time-frequency segment covariances
        signal_cross_covariance, reference_mean_square, _ = self.bm_covary(
            reference_basilar_membrane,
            processed_basilar_membrane,
        )

        _, ihc_sync_covariance = self.ave_covary2(
            signal_cross_covariance, reference_mean_square
        )
        # Ave segment coherence with IHC loss of sync
        basilar_membrane_sync5 = ihc_sync_covariance[4]

        # Extract and normalize the spectral features
        # Dloud:std
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

    def env_smooth_window(self) -> tuple[np.ndarray, float, int]:
        """
        Function to compute the window for smoothing the envelope returned by the
        cochlear model.
        It uses Hanning Window by default

        Returns:
            window: The Hanning window
            wsum: sum for normalization
            n_samples: segment size in samples
        """
        # Compute the window
        # Segment size in samples
        n_samples = int(np.around(self.segment_size * (0.001 * self.sample_rate)))
        # 0=even, 1=odd
        n_samples += 1 if n_samples % 2 > 0 else 0

        window = np.hanning(n_samples)  # Raised cosine von Hann window
        wsum = np.sum(window)  # Sum for normalization
        return window, float(wsum), n_samples

    def env_smooth(self, envelopes: np.ndarray) -> ndarray:
        """
        Function to smooth the envelope returned by the cochlear model. The
        envelope is divided into segments having a 50% overlap. Each segment is
        windowed, summed, and divided by the window sum to produce the average.
        A raised cosine window is used. The envelope sub-sampling frequency is
        2*(1000/segsize).

        Args:
            envelopes (np.ndarray): matrix of envelopes in each of the auditory bands

        Returns:
            smooth: matrix of subsampled windowed averages in each band
        """
        window, wsum, n_samples = self.env_smooth_window()
        #  The first segment has a half window
        nhalf = n_samples // 2
        halfwindow = window[nhalf:n_samples]
        halfsum = np.sum(halfwindow)

        # Number of segments and assign the matrix storage
        n_channels, npts = envelopes.shape
        nseg = int(1 + np.floor((npts - n_samples // 2) / nhalf))
        smooth = np.zeros((n_channels, nseg))

        # Loop to compute the envelope in each frequency band
        for k in range(n_channels):
            # Extract the envelope in the frequency band
            r = envelopes[k, :]  # pylint: disable=invalid-name

            # The first (half) windowed segment
            nstart = 0
            smooth[k, 0] = np.sum(r[nstart:nhalf] * np.conj(halfwindow)) / halfsum

            # Loop over the remaining full segments, 50% overlap
            for n in range(1, nseg - 1):
                nstart = int(nstart + nhalf)
                nstop = int(nstart + n_samples)
                smooth[k, n] = sum(r[nstart:nstop] * np.conj(window)) / wsum

            # The last (half) windowed segment
            nstart = (nseg - 1) * nhalf
            nstop = nstart + nhalf
            smooth[k, nseg - 1] = (
                np.sum(r[nstart:nstop] * np.conj(window[:nhalf])) / halfsum
            )

        return smooth

    def melcor9(
        self,
        reference: ndarray,
        distorted: ndarray,
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

        Args:
            reference (): subsampled input signal envelope in dB SL in each
                critical band
            distorted (): subsampled distorted output signal envelope
            n_cepstral_coef (int): Number of cepstral coefficients

        Returns:
            mel_cepstral_average (): average of the modulation correlations
                across analysis frequency bands and modulation frequency bands,
                basis functions 2 -6
            mel_cepstral_low (): average over the four lower mod freq bands,
                0 - 20 Hz
            mel_cepstral_high (): average over the four higher mod freq bands,
                20 - 125 Hz
            mel_cepstral_modulation (): vector of cross-correlations by modulation
                frequency, averaged over analysis frequency band
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
        index = np.where(reference_sum > self.silence_threshold)[0]

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
        _reference = reference[:, index]
        _distorted = distorted[:, index]

        # Add the low-level noise to the envelopes
        _reference += self.add_noise * np.random.standard_normal(_reference.shape)
        _distorted += self.add_noise * np.random.standard_normal(_distorted.shape)

        # Compute the mel cepstrum coefficients using only those segments
        # above threshold
        reference_cep = np.dot(cepm.T, _reference[:, :segments_above_threshold])
        distorted_cep = np.dot(cepm.T, _distorted[:, :segments_above_threshold])

        reference_cep -= np.mean(reference_cep, axis=1, keepdims=True)
        distorted_cep -= np.mean(distorted_cep, axis=1, keepdims=True)

        # Envelope sampling parameters
        # Envelope sampling frequency in Hz
        sampling_freq = 1000.0 / (0.5 * self.segment_size)
        # Envelope Nyquist frequency
        nyquist_freq = 0.5 * sampling_freq

        # Design the linear-phase envelope modulation filters

        n_fir = np.around(128 * (nyquist_freq / 125))

        # Force an even filter length
        n_fir = int(2 * np.floor(n_fir / 2))
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

        mel_cepstral_cross_covar = self.melcor9_crosscovmatrix(
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

    @staticmethod
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
        nfir2 = nfir // 2
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

    @staticmethod
    def spectrum_diff(
        reference_sl: ndarray, processed_sl: ndarray
    ) -> tuple[ndarray, ndarray, ndarray]:
        """
        Method to compute changes in the long-term spectrum and spectral slope.

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
        # Loudness sum = 1 (arbitrary amplitude, proportional to sones)
        reference_linear_magnitude /= reference_sum
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
        self,
        reference_basilar_membrane: ndarray,
        processed_basilar_membrane: ndarray,
    ) -> tuple[ndarray, ndarray, ndarray]:
        """
        Compute the cross-covariance (normalized cross-correlation) between
        the reference and processed signals in each auditory band.
        The signals are divided into segments having 50% overlap.

        Arguments:
            reference_basilar_membrane (): Basilar Membrane movement, reference signal
            processed_basilar_membrane (): Basilar Membrane movement, processed signal

        Returns:
            signal_cross_covariance (np.array) : [nchan,nseg] of cross-covariance values
            reference_mean_square (np.array) : [nchan,nseg] of MS input signal
                energy values
            processed_mean_square (np.array) : [nchan,nseg] of MS processed signal
                energy values

        Updates:
            James M. Kates, 28 August 2012.
            Output amplitude adjustment added, 30 october 2012.
            Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
        """

        # Initialize parameters
        small = 1.0e-30

        # Lag for computing the cross-covariance
        lagsize = 1.0  # Lag (+/-) in msec
        maxlag = np.around(lagsize * (0.001 * self.sample_rate))  # Lag in samples

        # Compute the segment size in samples
        nwin = int(np.around(self.segment_size * (0.001 * self.sample_rate)))

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
                    int(len(reference_seg) - 1 - maxlag) : int(
                        maxlag + len(reference_seg)
                    )
                ]
                unbiased_cross_correlation = np.max(np.abs(correlation * win_corr))
                if (ref_mean_square > small) and (proc_mean_squared > small):
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

            correlation = correlate(reference_seg, processed_seg, "full")
            correlation = correlation[
                int(len(reference_seg) - 1 - maxlag) : int(maxlag + len(reference_seg))
            ]

            unbiased_cross_correlation = np.max(np.abs(correlation * half_corr))
            if (ref_mean_square > small) and (proc_mean_squared > small):
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
            ihc_sync_covariance (): cross-covariance array, 6 different weightings
                for loss of IHC synchronization at high frequencies:
                  LP Filter Order     Cutoff Freq, kHz
                    1              1.5
                    3              2.0
                    5              2.5, 3.0, 3.5, 4.0
        """

        # Array dimensions
        n_channels = signal_cross_covariance.shape[0]

        # Initialize the LP filter for loss of IHC synchronization
        # Center frequencies in Hz on an ERB scale
        _center_freq = self.ear_model.center_frequency(n_channels)
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
        weight[signal_rms > self.silence_threshold] = 1

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


def compute_haaqi(
    processed_signal: ndarray,
    reference_signal: ndarray,
    audiogram: ndarray,
    audiogram_frequencies: ndarray,
    sample_rate: float = 24000.0,
    equalisation: int = 1,
    level1: float = 65.0,
    scale_reference: bool = True,
) -> float:
    """Compute HAAQI metric

    Args:
        processed_signal (np.ndarray): Output signal with noise, distortion, HA gain,
            and/or processing.
        reference_signal (np.ndarray): Input reference speech signal with no noise
            or distortion. If a hearing loss is specified, NAL-R equalization
            is optional
        audiogram (np.ndarray): Vector of hearing loss at the audiogram_frequencies
        audiogram_frequencies (np.ndarray): Audiogram frequencies
        sample_rate (int): Sample rate in Hz.
            Defaults to 24000.0.
        equalisation (int): hearing loss equalization mode for reference signal:
            1 = no EQ has been provided, the function will add NAL-R
            2 = NAL-R EQ has already been added to the reference signal
            Defaults to 1.
        level1 (float): Reference level in dB SPL. Defaults to 65.0.
        scale_reference (bool): Scale the reference signal to RMS=1. Defaults to True.
    """

    haaqi_audiogram_frequencies = [250, 500, 1000, 2000, 4000, 6000]
    audiogram_adjusted = np.array(
        [
            audiogram[i]
            for i in range(len(audiogram_frequencies))
            if audiogram_frequencies[i] in haaqi_audiogram_frequencies
        ]
    )

    if len(reference_signal) == 0:
        if len(processed_signal) == 0:
            # No scoring if no music
            return 1.0
        logger.error("If `Reference` is empty, `Processed` must be empty as well")
        return 0.0

    if scale_reference:
        reference_signal /= compute_rms(reference_signal)

    haaqi_metric = HAAQI(
        signal_sample_rate=sample_rate,
        ear_model_sample_rate=24000.0,
        equalisation=equalisation,
        level1=level1,
        silence_threshold=2.5,
        add_noise=0.0,
        segment_covariance=16,
        segment_size=8,
        earmodel_kwards={"nchan": 32},
    )

    score, _, _, _ = haaqi_metric.compute(
        reference=reference_signal,
        processed=processed_signal,
        hearing_loss=audiogram_adjusted,
    )
    return score
