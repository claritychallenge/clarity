from __future__ import annotations

import numpy as np
import torch


class HAAQI(torch.nn.Module):
    def __init__(
        self,
        level1: int = 65,
        silence_threshold: float = 2.5,
        add_noise: float = 0.0,
        segment_covariance: int = 16,
    ):
        super().__init__()
        self.level1 = level1
        self.silence_threshold = silence_threshold
        self.add_noise = add_noise
        self.segment_covariance = segment_covariance

    def forward(
        self,
        reference: torch.FloatTensor | np.ndarray,
        reference_freq: int,
        processed: torch.FloatTensor | np.ndarray,
        processed_freq: int,
        hearing_loss: torch.FloatTensor | np.ndarray,
        equalisation: int,
    ):
        pass

    def ear_model(self):
        pass
