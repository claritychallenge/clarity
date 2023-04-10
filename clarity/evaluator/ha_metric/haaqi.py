from __future__ import annotations

import numpy as np
import torch
from ear_model import EarModel


class Haaqi(torch.nn.Module):
    def __init__(
        self,
        silence_threshold: float = 2.5,
        add_noise: float = 0.0,
        segment_covariance: int = 16,
        segment_size: int = 8,
    ):
        super().__init__()
        self.silence_threshold = silence_threshold
        self.add_noise = add_noise
        self.segment_covariance = segment_covariance
        self.segment_size = segment_size
        self.ear_model = EarModel()

    def forward(
        self,
        reference: torch.FloatTensor | np.ndarray,
        reference_freq: int,
        processed: torch.FloatTensor | np.ndarray,
        processed_freq: int,
        hearing_loss: torch.FloatTensor | np.ndarray,
        equalisation: int,
        level1: int = 65,
    ):
        (
            reference_db,
            reference_basilar_membrane,
            processed_db,
            processed_basilar_membrane,
            reference_sl,
            processed_sl,
            freq_sample,
        ) = self.ear_model(
            reference,
            reference_freq,
            processed,
            processed_freq,
            hearing_loss,
            equalisation,
            level1,
        )

        segment_size = 8  # Averaging segment size in msec
        reference_smooth = self.env_smooth(reference_db, freq_sample)
        processed_smooth = self.env_smooth(processed_db, freq_sample)

        _, _, mel_cepstral_high, _ = self.melcor9(reference_smooth, processed_smooth)

        dloud_vector, dnorm_vector, _ = self.spectrum_diff(reference_sl, processed_sl)

        signal_cross_covariance, reference_mean_square, _ = self.bm_covary(
            reference_basilar_membrane,
            processed_basilar_membrane,
            freq_sample,
        )

        _, ihc_sync_covariance = self.ave_covary2(
            signal_cross_covariance,
            reference_mean_square,
        )
        basilar_membrane_sync5 = ihc_sync_covariance[4]

        d_loud = dloud_vector[1] / 2.5  # Loudness difference std
        d_loud = 1.0 - d_loud  # 1=perfect, 0=bad
        d_loud = min(d_loud, 1)
        d_loud = max(d_loud, 0)

        # Dnorm:std
        d_norm = dnorm_vector[1] / 25  # Slope difference std
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

    def env_smooth(self, reference_db, freq_sample):
        segment_size = self.segment_size
        return reference_db

    def melcor9(self, reference_smooth, processed_smooth):
        silence_threshold = self.silence_threshold
        add_noise = self.add_noise
        segment_size = self.segment_size
        return None, None, 8, 8

    def spectrum_diff(self, reference_sl, processed_sl):
        return reference_sl, processed_sl, None

    def bm_covary(
        self,
        reference_basilar_membrane,
        processed_basilar_membrane,
        freq_sample,
    ):
        segment_covariance = self.segment_covariance
        return None, None, None

    def ave_covary2(self, signal_cross_covariance, reference_mean_square):
        silence_threshold = self.silence_threshold
        return torch.zeros(32), torch.zeros(32)


if __name__ == "__main__":
    torch.random.manual_seed(0)
    haaqi = Haaqi()

    sample_rate = 44100
    duration = 1

    reference = torch.randn(1, sample_rate * duration)
    processed = torch.randn(1, sample_rate * duration)
    hearing_loss = torch.Tensor([45, 45, 35, 45, 60, 65])
    equalisation = 0

    combined_model, nonlinear_model, linear_model, raw = haaqi(
        reference,
        sample_rate,
        processed,
        sample_rate,
        hearing_loss,
        equalisation,
    )

# if __name__ == "__main__":
#     np.random.seed(0)
#
#     sample_rate = 44100
#     duration = 1
#
#     reference = np.random.randn(1, sample_rate * duration)
#     processed = np.random.randn(1, sample_rate * duration)
#     hearing_loss = np.array([45, 45, 35, 45, 60, 65])
#     equalisation = 0
#
#     combined_model, nonlinear_model, linear_model, raw = haaqi_v1(
#         reference,
#         sample_rate,
#         processed,
#         sample_rate,
#         hearing_loss,
#         equalisation,
#     )
