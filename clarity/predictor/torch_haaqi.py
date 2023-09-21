"""Torch HAAQI module"""
from __future__ import annotations

# pylint: disable=import-error
import logging
import numpy as np
import torch

from numpy import ndarray
from typing import Final

from clarity.predictor.ha_ear_model.ear_model import EarModel
from clarity.utils.audiogram import Audiogram

logger = logging.getLogger(__name__)

# HAAQI assumes the following audiogram frequencies:
HAAQI_AUDIOGRAM_FREQUENCIES: Final = np.array([250, 500, 1000, 2000, 4000, 6000])


class TorchHAAQI(torch.nn.Module):
    """Torch HAAQI module"""

    def __init__(
        self,
        processed_freq: float,
        reference_freq: float,
        silence_threshold: float = 2.5,
        add_noise: float = 0.0,
        segment_covariance: int = 16,
    ):
        """

        Args:
            processed_freq (float): Sampling rate in Hz for processed signal.
            reference_freq (float): Sampling rate in Hz for reference signal.
            silence_threshold (float): Silence threshold sum across bands,
                dB above auditory threshold. Default : 2.5
            add_noise (float): Additive noise dB SL to condition cross-covariances.
                Defaults to 0.0
            segment_covariance (int): Segment size for the covariance calculation.
                Defaults to 16
        """
        super().__init__()
        self.processed_freq = processed_freq
        self.reference_freq = reference_freq

        self.silence_threshold = silence_threshold
        self.add_noise = add_noise
        self.segment_covariance = segment_covariance

    def forward(
        self,
        reference: torch.Tensor,
        processed: torch.Tensor,
        audiogram: Audiogram,
        equalisation: int = 1,
        level1: float = 65.0,
    ):
        if not audiogram.has_frequencies(HAAQI_AUDIOGRAM_FREQUENCIES):
            logging.warning(
                "Audiogram does not have all HAAQI frequency measurements"
                "Measurements will be interpolated"
            )
        audiogram = audiogram.resample(HAAQI_AUDIOGRAM_FREQUENCIES)

        ear_model = EarModel(audiometric_freq=audiogram.frequencies)

        (
            reference_db,
            reference_basilar_membrane,
            processed_db,
            processed_basilar_membrane,
            reference_sl,
            processed_sl,
            freq_sample,
        ) = self.ear_model.forward(
            reference,
            self.reference_freq,
            processed,
            self.processed_freq,
            audiogram.levels,
            equalisation,
            level1,
        )

        segment_size = 8  # Averaging segment size in msec
        reference_smooth = self.env_smooth(
            reference_db, segment_size, self.reference_freq
        )
        processed_smooth = self.env_smooth(
            processed_db, segment_size, self.processed_freq
        )

        # Mel cepstrum correlation after passing through modulation filterbank
        _, _, mel_cepstral_high, _ = eb.melcor9(
            reference_smooth,
            processed_smooth,
            silence_threshold,
            add_noise,
            segment_size,
        )  # 8 modulation freq bands

        # Linear changes in the long-term spectra
        # dloud  vector: [sum abs diff, std dev diff, max diff] spectra
        # dnorm  vector: [sum abs diff, std dev diff, max diff] norm spectra
        # dslope vector: [sum abs diff, std dev diff, max diff] slope
        dloud_stats, dnorm_stats, _ = eb.spectrum_diff(reference_sl, processed_sl)

        # Temporal fine structure (TFS) correlation measurements
        # Compute the time-frequency segment covariances
        signal_cross_covariance, reference_mean_square, _ = eb.bm_covary(
            reference_basilar_membrane,
            processed_basilar_membrane,
            segment_covariance,
            sample_rate,
        )

        # Average signal segment cross-covariance
        # avecov=weighted ave of cross-covariances, using only data above threshold
        # syncov=ave cross-covariance with added IHC loss of synchronization at HF
        _, ihc_sync_covariance = eb.ave_covary2(
            signal_cross_covariance, reference_mean_square, silence_threshold
        )
        basilar_membrane_sync5 = ihc_sync_covariance[
            4
        ]  # Ave segment coherence with IHC loss of sync

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

    def env_smooth(
        self, envelopes: torch.tensor, segment_size: int, sample_rate: float
    ) -> ndarray:
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

        n_samples = int(
            torch.round(segment_size * (0.001 * sample_rate))
        )  # Segment size in samples
        test = n_samples - 2 * torch.floor(n_samples / 2)  # 0=even, 1=odd
        if test > 0:
            # Force window length to be even
            n_samples = n_samples + 1
        window = torch.hann_window(n_samples)  # Raised cosine von Hann window
        wsum = torch.sum(window)  # Sum for normalization

        # The first segment has a half window
        nhalf = int(n_samples / 2)
        halfwindow = window[nhalf:n_samples]
        halfsum = torch.sum(halfwindow)

        # Number of segments and assign the matrix storage
        n_channels = envelopes.size(0)
        npts = envelopes.size(1)
        nseg = int(
            1
            + torch.floor(npts / n_samples)
            + torch.floor((npts - n_samples / 2) / n_samples)
        )
        smooth = torch.zeros((n_channels, nseg))

        # Loop to compute the envelope in each frequency band
        for k in range(n_channels):
            # Extract the envelope in the frequency band
            r = envelopes[k, :]

            # The first (half) windowed segment
            nstart = 0
            smooth[k, 0] = (
                torch.sum(r[nstart:nhalf] * halfwindow.conj().transpose()) / halfsum
            )

            # Loop over the remaining full segments, 50% overlap
            for n in range(1, nseg - 1):
                nstart = int(nstart + nhalf)
                nstop = int(nstart + n_samples)
                smooth[k, n] = (
                    torch.sum(r[nstart:nstop] * window.conj().transpose()) / wsum
                )

            # The last (half) windowed segment
            nstart = nstart + nhalf
            nstop = nstart + nhalf
            smooth[k, nseg - 1] = (
                torch.sum(r[nstart:nstop] * window[:nhalf].conj().transpose()) / halfsum
            )

        return smooth
