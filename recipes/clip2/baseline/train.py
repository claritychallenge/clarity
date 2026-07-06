"""
train.py — Lyric Intelligibility Model · Training Script
=========================================================

Two-stage training procedure
------------------------------
Stage 1 — Mono pretraining
    The Whisper backbone is frozen.  Only the projection layers and regression
    head are trained on downmixed mono audio.  At epoch cfg.train.finetune.epoch
    the top encoder layers are unfrozen for the remainder of training.

Stage 2 — Better-ear fine-tuning
    Start from the Stage 1 best checkpoint (train.resume_from).  The dataloader
    supplies stereo audio [B, 2, T]; both channels are scored independently and
    BetterEarLoss optimises MSE against max(score_L, score_R).

Usage
-----
    # Stage 1 — mono
    python train.py

    # Stage 2 — better-ear (resume from Stage 1 best checkpoint)
    python train.py train.better_ear=true train.resume_from=outputs/stage1/best_model.pt

    # Override any config value from CLI
    python train.py model.backbone=whisper-large-v3 train.lr=5e-4
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from clip_dataset import (
    LyricIntelligibilityDataset,
    build_dataloader,
    build_train_dev_loaders,
)
from lyric_intelligibility_model import BetterEarLoss, WhisperIntelligibilityModel
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm
from transformers import WhisperProcessor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------


def prepare_batch_whisper(
    audio_np: np.ndarray,
    processor: WhisperProcessor,
    device: torch.device,
    score_np: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Convert raw waveforms to Whisper log-mel spectrograms.

    Parameters
    ----------
    audio_np : np.ndarray, shape [B, T]
        Raw 16 kHz mono waveforms.
    processor : WhisperProcessor
        Processor matching the Whisper backbone.
    device : torch.device
        Target device.
    score_np : np.ndarray, shape [B], or None
        Ground-truth scores.  Pass None in inference mode.

    Returns
    -------
    input_features : [B, 80/128, 3000]
    scores         : [B] or None
    """
    audio_list = list(audio_np) if audio_np.ndim == 2 else [audio_np]
    feats = processor(audio_list, sampling_rate=16_000, return_tensors="pt")
    input_features = feats.input_features.to(device)

    scores = None
    if score_np is not None:
        scores = torch.tensor(np.array(score_np, dtype=np.float32)).to(device)

    return input_features, scores


def prepare_batch_whisper_stereo(
    audio_np: np.ndarray,
    processor: WhisperProcessor,
    device: torch.device,
    score_np: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Split a stereo batch [B, 2, T] into two sets of Whisper log-mel spectrograms.

    Parameters
    ----------
    audio_np : np.ndarray, shape [B, 2, T]
        Stereo batch from the dataloader (mono=False).
    processor : WhisperProcessor
    device : torch.device
    score_np : np.ndarray, shape [B], or None

    Returns
    -------
    feats_L : [B, 80/128, 3000]
    feats_R : [B, 80/128, 3000]
    scores  : [B] or None
    """
    if audio_np.ndim != 3 or audio_np.shape[1] < 2:
        raise ValueError(f"Expected audio_np with shape [B, 2, T]; got {audio_np.shape}")
    left_list = [audio_np[i, 0] for i in range(len(audio_np))]
    right_list = [audio_np[i, 1] for i in range(len(audio_np))]

    feats_L = processor(
        left_list, sampling_rate=16_000, return_tensors="pt"
    ).input_features.to(device)
    feats_R = processor(
        right_list, sampling_rate=16_000, return_tensors="pt"
    ).input_features.to(device)

    scores = None
    if score_np is not None:
        scores = torch.tensor(np.array(score_np, dtype=np.float32)).to(device)

    return feats_L, feats_R, scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute MSE and PCC between predictions and targets."""
    if len(preds) > 1:
        pcc = float(pearsonr(preds, targets)[0])
        if np.isnan(pcc):
            pcc = 0.0
    else:
        pcc = 0.0
    mse = float(np.mean((preds - targets) ** 2))
    return {
        "mse": mse,
        "pcc": pcc,
    }


# ---------------------------------------------------------------------------
# Unified train / eval epoch
# ---------------------------------------------------------------------------


def run_epoch(
    model: WhisperIntelligibilityModel,
    loader,
    processor: WhisperProcessor,
    device: torch.device,
    better_ear: bool,
    criterion: nn.Module,
    epoch: int,
    total_epochs: int,
    # Training-only arguments (all None → eval mode)
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
) -> dict[str, float]:
    """
    Run one epoch in either train or eval mode.

    Pass optimizer/scheduler/scaler to run a training epoch; leave them as
    None for evaluation (the function calls model.eval() and torch.no_grad()).

    Parameters
    ----------
    model : WhisperIntelligibilityModel
    loader : DataLoader
        Must supply stereo batches [B, 2, T] when better_ear=True.
    processor : WhisperProcessor
    device : torch.device
    better_ear : bool
        If True, score left and right channels separately and use BetterEarLoss.
    criterion : nn.Module
        nn.MSELoss (mono) or BetterEarLoss (better-ear).
    epoch : int
        Current epoch (1-based), used only for the progress bar label.
    total_epochs : int
        Total epochs, used only for the progress bar label.
    optimizer : Optimizer or None
    scheduler : LRScheduler or None
    scaler : GradScaler or None

    Returns
    -------
    dict with keys "loss", "mse", "pcc"
    """
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    split_label = "train" if training else "dev"
    if better_ear:
        split_label += "/be"

    running_loss = 0.0
    all_preds: list[float] = []
    all_targets: list[float] = []

    ctx = torch.enable_grad() if training else torch.no_grad()

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{total_epochs} [{split_label}]",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    with ctx:
        for audio, _audio2, _sev_str, _sev, _transcript, score in pbar:
            if better_ear:
                feats_L, feats_R, scores = prepare_batch_whisper_stereo(
                    audio, processor, device, score
                )
                if training:
                    optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    preds_L = model(feats_L)
                    preds_R = model(feats_R)
                    loss = criterion(preds_L, preds_R, scores)
                preds = torch.maximum(preds_L, preds_R).detach()
            else:
                feats, scores = prepare_batch_whisper(audio, processor, device, score)
                if training:
                    optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    preds = model(feats)
                    loss = criterion(preds, scores)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # OneCycleLR steps per batch; ReduceLROnPlateau steps per epoch.
                if not isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step()

            running_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(scores.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    model: WhisperIntelligibilityModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics: dict[str, float],
    cfg: DictConfig,
    is_best: bool,
    best_epoch: int,
    keep_last: int = 3,
) -> None:
    """
    Save a training checkpoint and prune old ones.

    Always writes checkpoint_e<NNN>.pt for the current epoch and
    best_model.pt whenever is_best=True.  After saving, any
    checkpoint_e*.pt files that are neither among the last keep_last
    epochs nor the current best epoch are deleted.

    Only trainable parameters are saved when the backbone is still frozen,
    keeping file sizes small during Stage 1 head-only training.

    Parameters
    ----------
    out_dir : Path
    epoch : int
        Current epoch number.
    model, optimizer, scheduler, metrics, cfg : standard training state.
    is_best : bool
        If True, also write best_model.pt.
    best_epoch : int
        Epoch number of the current best checkpoint — always kept on disk.
    keep_last : int
        Number of most-recent epoch checkpoints to retain.  Default: 3.
    """
    backbone_unfrozen = any(p.requires_grad for p in model.encoder.parameters())

    if backbone_unfrozen:
        # Save fine-tuned encoder weights. The decoder no longer exists in the
        # model (dropped at construction time) so no decoder keys are present.
        model_state = model.state_dict()
        saved_backbone = "encoder_only"
    else:
        # Backbone frozen: skip encoder.* — reloaded from HuggingFace at init.
        model_state = {
            k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
        }
        saved_backbone = "none (frozen)"

    state = {
        "epoch": epoch,
        "model_state": model_state,
        "saved_backbone": saved_backbone,
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "metrics": metrics,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    path = out_dir / f"checkpoint_e{epoch:03d}.pt"
    torch.save(state, path)

    if is_best:
        best_path = out_dir / "best_model.pt"
        torch.save(state, best_path)
        log.info(f"  ✓ New best → {best_path}  (val_loss={metrics['loss']:.4f})")

    # -- Prune old checkpoints ----------------------------------------------------
    # Collect all epoch checkpoints sorted oldest → newest.
    all_ckpts = sorted(out_dir.glob("checkpoint_e*.pt"))
    # Epochs to keep: the last `keep_last` on disk + the best epoch.
    protected = {best_epoch} | {
        int(p.stem.replace("checkpoint_e", "")) for p in all_ckpts[-keep_last:]
    }
    for ckpt in all_ckpts:
        ckpt_epoch = int(ckpt.stem.replace("checkpoint_e", ""))
        if ckpt_epoch not in protected:
            ckpt.unlink()
            log.debug(f"  Pruned checkpoint: {ckpt.name}")


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when val loss has not improved for patience epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------


def train(cfg: DictConfig, train_loader, val_loader) -> None:
    """
    Core training loop — called by main() after the dataloaders are built.

    Stage 1 (better_ear=False): backbone frozen → head trains on mono audio.
    At epoch cfg.train.finetune.epoch the top encoder layers are unfrozen.

    Stage 2 (better_ear=True): resume from Stage 1 best_model.pt, stereo
    dataloaders, BetterEarLoss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Device : {device}")
    log.info(f"Config :\n{OmegaConf.to_yaml(cfg)}")

    # -- Processor & model ------------------------------------------------
    model_id = WhisperIntelligibilityModel.SUPPORTED_MODELS[cfg.model.backbone]
    processor = WhisperProcessor.from_pretrained(model_id)

    enc_layers = list(cfg.model.get("use_encoder_layers") or [])
    model = WhisperIntelligibilityModel(
        backbone=cfg.model.backbone,
        hidden_dim=cfg.model.hidden_dim,
        use_encoder_layers=enc_layers or None,
        freeze_backbone=True,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
    ).to(device)

    # -- Resume from checkpoint (Stage 2 loads Stage 1 head weights) ------
    resume_from: str | None = cfg.train.get("resume_from", None)
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        # Remap legacy checkpoints saved with self.whisper (old key prefix) to
        # the current self.encoder layout.
        model_state = {
            k.replace("whisper.encoder.", "encoder.", 1): v
            for k, v in ckpt["model_state"].items()
        }
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        # Expected missing: encoder.* when backbone was frozen at save time.
        # Unexpected keys indicate an architecture mismatch.
        non_bb_missing = [k for k in missing if not k.startswith("encoder.")]
        if non_bb_missing:
            log.warning(f"resume_from: missing non-backbone keys: {non_bb_missing}")
        if unexpected:
            log.warning(f"resume_from: unexpected keys: {unexpected}")
        log.info(
            f"Resumed from {resume_from}  "
            f"(epoch {ckpt.get('epoch', '?')}, backbone={ckpt.get('saved_backbone', '?')})"
        )

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Params — total: {n_total:,}  trainable: {n_trainable:,}")

    # -- Loss -------------------------------------------------------------
    better_ear: bool = cfg.train.get("better_ear", False)
    criterion = BetterEarLoss() if better_ear else nn.MSELoss()
    log.info(f"Loss: {'BetterEarLoss' if better_ear else 'MSELoss'}")

    # -- Optimiser — head parameters only ---------------------------------
    head_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("encoder.") and p.requires_grad
    ]
    optimizer = AdamW(head_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # -- Scheduler --------------------------------------------------------
    es_patience: int = cfg.train.get("early_stopping", {}).get("patience", 10)
    es_min_delta: float = cfg.train.get("early_stopping", {}).get("min_delta", 1e-4)
    use_early_stopping = es_patience > 0

    if use_early_stopping:
        lr_patience: int = cfg.train.get("lr_patience", 3)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-7,
        )
        log.info(f"Scheduler: ReduceLROnPlateau  patience={lr_patience}")
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg.train.lr,
            total_steps=cfg.train.epochs * len(train_loader),
            pct_start=0.1,
        )
        log.info("Scheduler: OneCycleLR")

    scaler = (
        torch.cuda.amp.GradScaler()
        if cfg.train.fp16 and device.type == "cuda"
        else None
    )

    # -- Training loop -----------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 1
    backbone_unfrozen = False
    history: list = []
    early_stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta)
    keep_last: int = cfg.train.get("keep_last_checkpoints", 3)

    context_layers: int = cfg.train.finetune.get("context_layers", 2)
    log.info(
        f"Training for up to {cfg.train.epochs} epochs  "
        f"(early stopping patience={es_patience})"
    )
    log.info(
        f"Backbone unfreeze at epoch {cfg.train.finetune.epoch}  "
        f"(context_layers={context_layers}, lr={cfg.train.finetune.lr})"
    )

    epoch_pbar = tqdm(
        range(1, cfg.train.epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_pbar:
        t0 = time.time()

        # -- Backbone unfreeze ---------------------------------------------
        if epoch == cfg.train.finetune.epoch and not backbone_unfrozen:
            log.info(f"[Epoch {epoch}] Unfreezing backbone layers")
            model.unfreeze_backbone(
                num_top_layers=cfg.train.finetune.layers,
                context_layers=context_layers,
            )
            backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
            optimizer.add_param_group(
                {
                    "params": backbone_params,
                    "lr": cfg.train.finetune.lr,
                    "weight_decay": cfg.train.weight_decay,
                }
            )
            # Rebuild scheduler so the new param group is included.
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=cfg.train.get("lr_patience", 3),
                    min_lr=1e-7,
                )
            else:
                remaining = (cfg.train.epochs - epoch + 1) * len(train_loader)
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=[cfg.train.lr, cfg.train.finetune.lr],
                    total_steps=remaining,
                    pct_start=0.05,
                )
            backbone_unfrozen = True

        # -- Train and eval ------------------------------------------------
        train_m = run_epoch(
            model,
            train_loader,
            processor,
            device,
            better_ear,
            criterion,
            epoch,
            cfg.train.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        val_m = run_epoch(
            model,
            val_loader,
            processor,
            device,
            better_ear,
            criterion,
            epoch,
            cfg.train.epochs,
        )

        elapsed = time.time() - t0
        is_best = val_m["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_m["loss"]
            best_epoch = epoch

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_str = (
            f"lr={lrs[0]:.2e}"
            if len(lrs) == 1
            else f"lr_head={lrs[0]:.2e}  lr_bb={lrs[1]:.2e}"
        )

        summary = (
            f"Epoch {epoch:03d}/{cfg.train.epochs}  ({elapsed:.1f}s) | "
            f"train  loss={train_m['loss']:.4f}  pcc={train_m['pcc']:.3f} | "
            f"dev    loss={val_m['loss']:.4f}  pcc={val_m['pcc']:.3f}  "
            f" | {lr_str}" + ("  ★ best" if is_best else "")
        )
        tqdm.write(summary)
        log.info(summary)
        epoch_pbar.set_postfix(
            val_loss=f"{val_m['loss']:.4f}",
            val_pcc=f"{val_m['pcc']:.3f}",
            best=f"{best_val_loss:.4f}",
            es=f"{early_stopper.counter}/{es_patience}",
        )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_m["loss"])

        save_checkpoint(
            out_dir,
            epoch,
            model,
            optimizer,
            scheduler,
            val_m,
            cfg,
            is_best,
            best_epoch=best_epoch,
            keep_last=keep_last,
        )
        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        if use_early_stopping and early_stopper.step(val_m["loss"]):
            msg = (
                f"Early stopping at epoch {epoch} — "
                f"val loss did not improve for {es_patience} epochs."
            )
            tqdm.write(msg)
            log.info(msg)
            break

    # -- Post-training -----------------------------------------------------
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    final = f"\nTraining complete — best dev loss: {best_val_loss:.4f}  →  {out_dir}/best_model.pt"
    tqdm.write(final)
    log.info(final)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    better_ear: bool = cfg.train.get("better_ear", False)
    num_workers: int = min(cfg.data.get("num_workers", 4), os.cpu_count() or 1)

    if better_ear:
        shared = dict(
            root_path=cfg.data.root_path,
            strategy=cfg.data.strategy,
            seed=cfg.data.seed,
            mono=False,
        )
        train_ds = LyricIntelligibilityDataset(**shared, split="train")
        dev_ds = LyricIntelligibilityDataset(**shared, split="dev")
        train_loader = build_dataloader(
            train_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = build_dataloader(
            dev_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        train_loader, val_loader = build_train_dev_loaders(
            root_path=cfg.data.root_path,
            strategy=cfg.data.strategy,
            seed=cfg.data.seed,
            train_batch_size=cfg.data.batch_size,
            num_workers=num_workers,
            mono=True,
        )

    train(cfg, train_loader, val_loader)


if __name__ == "__main__":
    main()
