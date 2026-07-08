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

Loading a pretrained model + running inference
------------------------------------------------
This file is self-contained: the architecture (WhisperIntelligibilityModel)
and a ready-to-use inference wrapper (IntelligibilityPredictor) both live
here, so downloading just this one file (plus config.json + model.safetensors
from the same Hub repo) is enough.

    from lyric_intelligibility_model import IntelligibilityPredictor

    predictor = IntelligibilityPredictor.from_pretrained(
        "your-username/lyric-intelligibility-whisper"
    )
    result = predictor.predict("path/to/song.wav")   # or a directory
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import WhisperModel, WhisperProcessor

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

    def __init__(
        self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
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
        q = self.query.expand(B, -1, -1)  # [B, 1, H]
        context, _ = self.attn(q, layer_embeddings, layer_embeddings)
        return self.norm(context.squeeze(1))  # [B, H]


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
        "whisper-tiny": "openai/whisper-tiny",
        "whisper-base": "openai/whisper-base",
        "whisper-small": "openai/whisper-small",
        "whisper-medium": "openai/whisper-medium",
        "whisper-large-v3": "openai/whisper-large-v3",
    }

    def __init__(
        self,
        backbone: str = "whisper-large-v3",
        hidden_dim: int = 256,
        use_encoder_layers: list[int] | None = None,
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
        n_enc: int = self.encoder.config.encoder_layers

        # Default: use only the last encoder layer.
        self.use_encoder_layers: list[int] = (
            use_encoder_layers if use_encoder_layers is not None else [n_enc]
        )
        num_layers = len(self.use_encoder_layers)

        # Learned per-layer scalar weights, softmax-normalised in forward().
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # Per-layer projection: d_model → hidden_dim.
        self.layer_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        # Cross-layer attention is used only when more than one layer is selected.
        self.cross_layer_attn: CrossLayerAttention | None = (
            CrossLayerAttention(
                hidden_dim=hidden_dim, num_heads=attn_heads, dropout=dropout
            )
            if num_layers > 1
            else None
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
        self.hidden_dim = hidden_dim

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
        backbone_trainable = any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(backbone_trainable):
            outputs = self.encoder(
                input_features=input_features,
                output_hidden_states=True,
            )

        enc_states: tuple = outputs.hidden_states
        layer_weights = F.softmax(self.layer_weights, dim=0)  # [L]

        layer_pooled: list[torch.Tensor] = []
        for i, layer_idx in enumerate(self.use_encoder_layers):
            h = enc_states[layer_idx]  # [B, T', d_model]
            h = h * layer_weights[i]  # scale by learned weight
            h = h.mean(dim=1)  # temporal mean-pool → [B, d_model]
            h = self.layer_projections[i](h)  # project → [B, hidden_dim]
            layer_pooled.append(h)

        if self.cross_layer_attn is not None:
            stack = torch.stack(layer_pooled, dim=1)  # [B, L, hidden_dim]
            context = self.cross_layer_attn(stack)  # [B, hidden_dim]
        else:
            context = layer_pooled[0]  # [B, hidden_dim]

        return self.regressor(context).squeeze(-1)  # [B]

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
        total_enc = len(enc_layers)

        indices_to_unfreeze: set[int] = set()
        for layer_idx in self.use_encoder_layers:
            enc_idx = layer_idx - 1  # convert 1-based to 0-based
            for c in range(context_layers + 1):
                idx = enc_idx - c
                if 0 <= idx < total_enc:
                    indices_to_unfreeze.add(idx)

        # Respect the requested cap: keep only the top-most layers (highest indices).
        if num_top_layers > 0 and len(indices_to_unfreeze) > num_top_layers:
            indices_to_unfreeze = set(sorted(indices_to_unfreeze)[-num_top_layers:])

        for idx in sorted(indices_to_unfreeze):
            for p in enc_layers[idx].parameters():
                p.requires_grad_(True)

        log.info(
            f"Unfrozen {len(indices_to_unfreeze)} encoder layers "
            f"(layer-aware, context={context_layers}): "
            f"{sorted(i + 1 for i in indices_to_unfreeze)}"
        )

    # -----------------------------------------------------------------
    # Hub / checkpoint interface — public loading API for anyone who
    # downloads this model. (Uploading is a separate, private concern;
    # see save_and_push.py, which is not part of this file/repo.)
    # -----------------------------------------------------------------

    _WEIGHTS_SAFETENSORS = "model.safetensors"
    _WEIGHTS_BIN = "pytorch_model.bin"
    _CONFIG_NAME = "config.json"

    def _to_config_dict(self) -> dict:
        """Hyperparameters needed to reconstruct this model's architecture."""
        return {
            "backbone": self.backbone_name,
            "hidden_dim": self.hidden_dim,
            "use_encoder_layers": self.use_encoder_layers,
            "attn_heads": (
                self.cross_layer_attn.attn.num_heads
                if self.cross_layer_attn is not None
                else 4
            ),
            "whisper_processor_id": self.SUPPORTED_MODELS[self.backbone_name],
        }

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save config.json + weights (safetensors) to a local directory, in a
        layout that from_pretrained() understands.
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config = self._to_config_dict()
        with open(save_directory / self._CONFIG_NAME, "w") as f:
            json.dump(config, f, indent=2)

        # safetensors requires contiguous, non-shared tensors on CPU.
        state_dict = {k: v.contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state_dict, save_directory / self._WEIGHTS_SAFETENSORS)

        log.info(f"Saved model + config to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        revision: str | None = None,
    ) -> "WhisperIntelligibilityModel":
        """
        Load a model previously saved with save_pretrained() / pushed via
        save_and_push.py, either from a local directory or a Hugging Face
        Hub repo id.

        Example
        -------
            from lyric_intelligibility_model import WhisperIntelligibilityModel
            model = WhisperIntelligibilityModel.from_pretrained(
                "your-username/lyric-intelligibility-whisper"
            )
        """
        local_dir = Path(pretrained_model_name_or_path)
        if local_dir.is_dir():
            config_path = local_dir / cls._CONFIG_NAME
            weights_path = local_dir / cls._WEIGHTS_SAFETENSORS
            if not weights_path.exists():
                weights_path = local_dir / cls._WEIGHTS_BIN
        else:
            from huggingface_hub import hf_hub_download

            repo_id = str(pretrained_model_name_or_path)
            config_path = hf_hub_download(repo_id, cls._CONFIG_NAME, revision=revision)
            try:
                weights_path = hf_hub_download(
                    repo_id, cls._WEIGHTS_SAFETENSORS, revision=revision
                )
            except Exception:
                weights_path = hf_hub_download(
                    repo_id, cls._WEIGHTS_BIN, revision=revision
                )

        with open(config_path) as f:
            config = json.load(f)

        model = cls(
            backbone=config.get("backbone", "whisper-large-v3"),
            hidden_dim=config.get("hidden_dim", 256),
            use_encoder_layers=config.get("use_encoder_layers"),
            freeze_backbone=True,  # irrelevant at inference time (no_grad anyway)
            attn_heads=config.get("attn_heads", 4),
            dropout=0.0,  # disable dropout for inference
        )

        weights_path = Path(weights_path)
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning(f"Missing keys when loading state_dict: {missing}")
        if unexpected:
            log.warning(f"Unexpected keys when loading state_dict: {unexpected}")

        return model


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------


class IntelligibilityPredictor:
    """
    Wraps WhisperIntelligibilityModel + WhisperProcessor for easy inference
    on raw audio files, including stereo "better-ear" scoring and scoring
    every audio file in a directory.
    """

    # Extensions attempted when scanning a directory. soundfile's actual
    # format support depends on the installed libsndfile version.
    AUDIO_EXTENSIONS = {
        ".wav",
        ".flac",
        ".mp3",
    }

    def __init__(
        self,
        model: WhisperIntelligibilityModel,
        processor: WhisperProcessor,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.processor = processor

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        revision: str | None = None,
        device: str | None = None,
    ) -> "IntelligibilityPredictor":
        """
        Load a WhisperIntelligibilityModel (config + weights) from a Hub repo
        or local directory, plus the matching WhisperProcessor.

        Example
        -------
            from lyric_intelligibility_model import IntelligibilityPredictor
            predictor = IntelligibilityPredictor.from_pretrained(
                "cadenzachallenge/CLIP2-BaselineMono"
            )
            results = predictor.predict("path/to/song.wav")
        """
        model = WhisperIntelligibilityModel.from_pretrained(
            pretrained_model_name_or_path, revision=revision
        )
        processor = WhisperProcessor.from_pretrained(
            WhisperIntelligibilityModel.SUPPORTED_MODELS[model.backbone_name]
        )
        return cls(model, processor, device=device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _score_mono(self, waveform: np.ndarray, sr: int) -> float:
        """Score a single mono channel. waveform: 1-D float32 array."""
        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        score = self.model(input_features)
        return float(score.item())

    def predict(self, audio_path: str | Path, better_ear: bool = True) -> dict:
        """
        Predict intelligibility for a single audio file, OR, if audio_path is
        a directory, for every audio file at its top level (non-recursive).

        Mono audio is always scored directly. For stereo audio:
          - better_ear=True  (default): score left and right channels
            separately and take max(score_L, score_R), matching how the
            model is trained.
          - better_ear=False: downmix to mono (average channels) first,
            then score once.

        Returns
        -------
        If audio_path is a file: a single result dict with keys
            score       : float, final intelligibility score in (0, 1)
            channel     : "mono" | "left" | "right" | "downmix"
            per_channel : dict of per-channel scores (only when better_ear
                          scoring was actually applied to stereo audio), or
                          None
        If audio_path is a directory: a dict mapping filename -> result dict
        (as above), or {"error": "..."} for any file that failed to load.
        """
        path = Path(audio_path)
        if path.is_dir():
            return self.predict_directory(path, better_ear=better_ear)
        return self._predict_file(path, better_ear=better_ear)

    def predict_directory(self, directory: str | Path, better_ear: bool = True) -> dict:
        """
        Run predict() on every audio file at the top level of `directory`
        (subdirectories are not searched). Returns {filename: result}, with
        {"error": "..."} in place of a result for any file that fails.
        """
        directory = Path(directory)
        files = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in self.AUDIO_EXTENSIONS
        )
        if not files:
            log.warning(f"No audio files found in {directory} (non-recursive).")

        results: dict = {}
        for f in files:
            try:
                results[f.name] = self._predict_file(f, better_ear=better_ear)
            except Exception as e:
                log.warning(f"Failed to score {f.name}: {e}")
                results[f.name] = {"error": str(e)}
        return results

    def _predict_file(self, audio_path: str | Path, better_ear: bool = True) -> dict:
        waveform, sr = self._load_audio(audio_path)

        if waveform.ndim == 1:
            score = self._score_mono(waveform, sr)
            return {"score": score, "channel": "mono", "per_channel": None}

        # Stereo input: [2, T]
        if not better_ear:
            mono = waveform.mean(axis=0)
            score = self._score_mono(mono, sr)
            return {"score": score, "channel": "downmix", "per_channel": None}

        score_l = self._score_mono(waveform[0], sr)
        score_r = self._score_mono(waveform[1], sr)
        winner = "left" if score_l >= score_r else "right"
        return {
            "score": max(score_l, score_r),
            "channel": winner,
            "per_channel": {"left": score_l, "right": score_r},
        }

    @staticmethod
    def _load_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
        """
        Load an audio file, resampled to 16 kHz. Returns (waveform, sr).
        waveform is [T] for mono or [2, T] for stereo, float32 in [-1, 1].
        """
        import soundfile as sf
        import soxr

        waveform, sr = sf.read(str(audio_path), always_2d=False, dtype="float32")

        # soundfile returns [T, channels] for multi-channel; transpose to [channels, T]
        if waveform.ndim == 2:
            waveform = waveform.T

        target_sr = 16000
        if sr != target_sr:
            if waveform.ndim == 1:
                waveform = soxr.resample(waveform, sr, target_sr, quality="HQ")
            else:
                waveform = np.stack(
                    [soxr.resample(ch, sr, target_sr, quality="HQ") for ch in waveform]
                )
            sr = target_sr

        return waveform, sr


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
