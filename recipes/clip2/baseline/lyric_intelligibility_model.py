"""
lyric_intelligibility_model.py — Whisper-based Lyric Intelligibility Model
===========================================================================

Architecture
------------
Given a raw audio waveform (16 kHz), predicts a scalar in [0, 1] representing
how intelligible the lyrics are to a listener with a given hearing profile.

The model uses a pre-trained Whisper encoder as a feature extractor, selecting
one or more encoder layer outputs and fusing them via cross-layer attention
before passing the result to a small regression head.

Pipeline (one sample)
---------------------
    raw waveform [T]
        │
        ▼  WhisperProcessor
    log-mel spectrogram  [80, 3000]
        │
        ▼  Whisper encoder  (frozen)
    hidden states: list of (encoder_layers + 1) tensors, each [T', d_model]
        │
        ├── selected layers (e.g. the last encoder layer)
        │       │  temporal mean-pool → [d_model]
        │       │  linear projection  → [hidden_dim]
        │       │
        │  (if more than one layer selected)
        │       └── cross-layer attention → [hidden_dim]
        │
        ▼
    regression head (MLP + sigmoid) → score in (0, 1)

Better-ear inference
--------------------
The model always operates on mono input [B, T'].  For stereo audio use the
better-ear strategy: run the model on left and right channels separately
and take max(score_L, score_R) as the final prediction.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-layer attention
# ---------------------------------------------------------------------------

class CrossLayerAttention(nn.Module):
    """
    Fuse L layer representations into a single context vector.

    A single learned query attends over the L layer vectors (keys and values).
    Used only when more than one encoder layer is selected.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of each layer embedding.
    num_heads : int
        Number of parallel attention heads.  hidden_dim must be divisible
        by num_heads.
    dropout : float
        Dropout probability inside the attention module.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn  = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, layer_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        layer_embeddings : [B, L, hidden_dim]

        Returns
        -------
        context : [B, hidden_dim]
        """
        B = layer_embeddings.size(0)
        q = self.query.expand(B, -1, -1)                          # [B, 1, H]
        context, _ = self.attn(q, layer_embeddings, layer_embeddings)
        return self.norm(context.squeeze(1))                       # [B, H]


# ---------------------------------------------------------------------------
# Whisper intelligibility model
# ---------------------------------------------------------------------------

class WhisperIntelligibilityModel(nn.Module):
    """
    Whisper encoder-based intelligibility predictor.

    Encoder layer outputs are mean-pooled over the time axis, projected to
    hidden_dim, and (if more than one layer is selected) fused via
    CrossLayerAttention before the regression head.

    Parameters
    ----------
    backbone : str
        Whisper variant.  One of the keys in SUPPORTED_MODELS.
        Default: "whisper-large-v3".
    hidden_dim : int
        Hidden dimension for projections, cross-layer attention, and the
        regression MLP.  Default: 256.
    use_encoder_layers : list[int] or None
        1-indexed encoder layer indices to extract.
        None = use only the last encoder layer.
    freeze_backbone : bool
        Freeze all Whisper parameters at init; only the head is trained.
        The backbone is partially unfrozen later via unfreeze_backbone().
        Default: True.
    attn_heads : int
        Number of attention heads in CrossLayerAttention.  Only used when
        more than one encoder layer is selected.  Default: 4.
    dropout : float
        Dropout probability throughout the prediction head.  Default: 0.1.
    """

    SUPPORTED_MODELS: dict[str, str] = {
        "whisper-tiny":     "openai/whisper-tiny",
        "whisper-base":     "openai/whisper-base",
        "whisper-small":    "openai/whisper-small",
        "whisper-medium":   "openai/whisper-medium",
        "whisper-large-v3": "openai/whisper-large-v3",
    }

    def __init__(
        self,
        backbone: str = "whisper-large-v3",
        hidden_dim: int = 256,
        use_encoder_layers: Optional[list[int]] = None,
        freeze_backbone: bool = True,
        attn_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if backbone not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Choose from: {list(self.SUPPORTED_MODELS)}"
            )

        model_id = self.SUPPORTED_MODELS[backbone]
        # Load encoder only — the decoder is never used and for large-v3 costs
        # ~3 GB of GPU memory and ~1.5 GB per checkpoint if saved.
        # WhisperModel is loaded in fp32 to match the fp32 spectrograms produced
        # by WhisperProcessor (large-v3 is stored as fp16 on HuggingFace).
        _model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16)
        self.encoder = _model.encoder.to(torch.float32)
        del _model

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        d_model: int = self.encoder.config.d_model
        n_enc: int   = self.encoder.config.encoder_layers

        # Default: use only the last encoder layer.
        self.use_encoder_layers: list[int] = (
            use_encoder_layers if use_encoder_layers is not None
            else [n_enc]
        )
        num_layers = len(self.use_encoder_layers)

        # Learned per-layer scalar weights, softmax-normalised in forward().
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # Per-layer projection: d_model → hidden_dim.
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        # Cross-layer attention is used only when more than one layer is selected.
        self.cross_layer_attn: Optional[CrossLayerAttention] = (
            CrossLayerAttention(hidden_dim=hidden_dim, num_heads=attn_heads, dropout=dropout)
            if num_layers > 1 else None
        )

        # Regression head: hidden_dim → scalar score in (0, 1).
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.backbone_name = backbone
        self.hidden_dim    = hidden_dim

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Predict intelligibility from a log-mel spectrogram batch.

        Parameters
        ----------
        input_features : torch.Tensor, shape [B, 80/128, 3000]
            Log-mel spectrogram produced by WhisperProcessor.

        Returns
        -------
        scores : torch.Tensor, shape [B]
            Intelligibility score per sample in (0, 1).
        """
        backbone_frozen = not next(self.encoder.parameters()).requires_grad
        with torch.set_grad_enabled(not backbone_frozen):
            outputs = self.encoder(
                input_features       = input_features,
                output_hidden_states = True,
            )

        enc_states: tuple = outputs.hidden_states
        layer_weights = F.softmax(self.layer_weights, dim=0)  # [L]

        layer_pooled: list[torch.Tensor] = []
        for i, layer_idx in enumerate(self.use_encoder_layers):
            h = enc_states[layer_idx]          # [B, T', d_model]
            h = h * layer_weights[i]           # scale by learned weight
            h = h.mean(dim=1)                  # temporal mean-pool → [B, d_model]
            h = self.layer_projections[i](h)   # project → [B, hidden_dim]
            layer_pooled.append(h)

        if self.cross_layer_attn is not None:
            stack = torch.stack(layer_pooled, dim=1)   # [B, L, hidden_dim]
            context = self.cross_layer_attn(stack)     # [B, hidden_dim]
        else:
            context = layer_pooled[0]                  # [B, hidden_dim]

        return self.regressor(context).squeeze(-1)     # [B]

    def unfreeze_backbone(
        self,
        num_top_layers: int = 4,
        context_layers: int = 2,
    ) -> None:
        """
        Unfreeze Whisper encoder layers for fine-tuning.

        Unfreezes the selected encoder layers plus context_layers layers below
        each of them (so gradients can flow through the layers feeding into
        each selected layer).

        Parameters
        ----------
        num_top_layers : int
            Fallback cap — at most this many layers are unfrozen in total.
            Default: 4.
        context_layers : int
            Number of layers below each selected layer to also unfreeze.
            E.g. use_encoder_layers=[32], context_layers=2 → unfreezes 30, 31, 32.
            Default: 2.
        """
        enc_layers = self.encoder.layers
        total_enc  = len(enc_layers)

        indices_to_unfreeze: set[int] = set()
        for layer_idx in self.use_encoder_layers:
            enc_idx = layer_idx - 1   # convert 1-based to 0-based
            for c in range(context_layers + 1):
                idx = enc_idx - c
                if 0 <= idx < total_enc:
                    indices_to_unfreeze.add(idx)

        for idx in sorted(indices_to_unfreeze):
            for p in enc_layers[idx].parameters():
                p.requires_grad_(True)

        log.info(
            f"Unfrozen {len(indices_to_unfreeze)} encoder layers "
            f"(layer-aware, context={context_layers}): "
            f"{sorted(i + 1 for i in indices_to_unfreeze)}"
        )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class BetterEarLoss(nn.Module):
    """
    MSE loss for stereo (better-ear) training.

    The model is called once per channel (left and right) and this loss
    computes MSE against the better-ear prediction: max(score_L, score_R).
    Only the better channel contributes gradient at each step — this is why
    Stage 1 mono pretraining is recommended before switching to this loss.
    """

    def forward(
        self,
        preds_L: torch.Tensor,
        preds_R: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        preds_L : [B]   Left-channel predictions.
        preds_R : [B]   Right-channel predictions.
        targets : [B]   Ground-truth scores.

        Returns
        -------
        loss : scalar   MSE against max(preds_L, preds_R).
        """
        better_ear = torch.maximum(preds_L, preds_R)  # [B]
        return F.mse_loss(better_ear, targets)
