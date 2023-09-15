"""Torch HAAQI module"""
from __future__ import annotations

# pylint: disable=import-error
import torch
from clarity.predictor.ha_ear_model.ear_model import EarModel


class TorchHAAQI(torch.nn.Module):
    def __init__(
        self,
        processed_freq: float,
        reference_freq: float,
        silence_threshold: float = 2.5,
        add_noise: float = 0.0,
        segment_covariance: int = 16,
        audiometric_freq: list | None = None,
    ):
        super().__init__()
        self.processed_freq = processed_freq
        self.reference_freq = reference_freq

        self.silence_threshold = silence_threshold
        self.add_noise = add_noise
        self.segment_covariance = segment_covariance
        if audiometric_freq is None:
            audiometric_freq = [250, 500, 1000, 2000, 4000, 6000]
        self.ear_model = EarModel(audiometric_freq=audiometric_freq)

    def forward(
        self,
        reference: torch.Tensor,
        processed: torch.Tensor,
        audiogram: torch.Tensor,
        equalisation: int = 1,
        level1: float = 65.0,
    ):
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
            audiogram,
            equalisation,
            level1,
        )

        # Compute the envelope cepstral correlation and Basilar Membrane vibration
