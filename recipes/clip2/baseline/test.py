"""
test.py — Lyric Intelligibility Model · Evaluation / Inference Script
======================================================================

Two modes
---------
Evaluation mode  (default, test.inference_only=false)
    Requires ground-truth scores.  Computes MSE, RMSE, MAE, PCC, SCC overall
    and per hearing-loss severity level.  Writes predictions_complete.csv,
    results.json, and diagnostic plots to test.output_dir.

Inference mode   (test.inference_only=true)
    No ground-truth scores needed.  Writes predictions.csv only.

The model architecture is always reconstructed from the config embedded in
the checkpoint, so no separate model config is needed.

Usage
-----
    # Evaluate against labelled eval set
    python test.py test.checkpoint=./runs/best_model.pt test.split=eval

    # Evaluate against the dev split used during training
    python test.py test.checkpoint=./runs/best_model.pt test.split=valid

    # Inference only (no ground-truth labels required)
    python test.py test.checkpoint=./runs/best_model.pt test.inference_only=true

    # Better-ear evaluation (Stage 2 checkpoint, stereo data)
    python test.py test.checkpoint=./runs/stage2/best_model.pt test.better_ear=true
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from clip_dataset import LyricIntelligibilityDataset, build_dataloader
from lyric_intelligibility_model import WhisperIntelligibilityModel
from omegaconf import DictConfig
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from train import prepare_batch_whisper, prepare_batch_whisper_stereo
from transformers import WhisperProcessor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute MSE, RMSE, MAE, PCC, and SCC between predictions and targets."""
    if len(preds) > 1:
        pcc = float(pearsonr(preds, targets)[0])
        scc = float(spearmanr(preds, targets)[0])
        if np.isnan(pcc):
            pcc = 0.0
        if np.isnan(scc):
            scc = 0.0
    else:
        pcc = 0.0
        scc = 0.0
    return {
        "mse": float(np.mean((preds - targets) ** 2)),
        "rmse": float(np.sqrt(np.mean((preds - targets) ** 2))),
        "mae": float(np.mean(np.abs(preds - targets))),
        "pcc": pcc,
        "scc": scc,
    }


def metrics_by_severity(
    preds: np.ndarray,
    targets: np.ndarray,
    severities: list,
) -> dict[int, dict[str, float]]:
    """
    Compute per-severity-level metrics.

    Severity groups with fewer than 2 samples are skipped (correlation
    metrics are undefined for a single point).

    Parameters
    ----------
    severities : list
        Integer labels or one-hot vectors (argmax is applied to convert).

    Returns
    -------
    dict mapping integer severity level → metrics dict.
    """
    sev = np.array(severities)
    if sev.ndim == 2:
        sev = sev.argmax(axis=1)

    return {
        int(level): compute_metrics(preds[sev == level], targets[sev == level])
        for level in np.unique(sev)
        if (sev == level).sum() >= 2
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_scatter(preds: np.ndarray, targets: np.ndarray, out_path: Path) -> None:
    """Scatter plot of predicted vs ground-truth scores."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.5, s=20, edgecolors="none", color="steelblue")
    lims = [
        min(targets.min(), preds.min()) - 0.05,
        max(targets.max(), preds.max()) + 0.05,
    ]
    ax.plot(lims, lims, "k--", lw=1, label="Ideal (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground-Truth Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Predicted vs Ground-Truth")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Scatter plot       → {out_path}")


def plot_error_histogram(
    preds: np.ndarray, targets: np.ndarray, rmse: float, out_path: Path
) -> None:
    """Histogram of per-sample absolute errors with overall RMSE marked."""
    errors = np.abs(preds - targets)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=30, color="tomato", edgecolor="white", linewidth=0.5)
    ax.axvline(
        rmse, color="black", linestyle="--", linewidth=1.5, label=f"RMSE = {rmse:.4f}"
    )
    ax.set_xlabel("Absolute Error  |pred − target|")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Overall Error Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Error histogram    → {out_path}")


def plot_metric_by_severity(
    sev_metrics: dict[int, dict[str, float]],
    metric: str,
    out_path: Path,
    color: str = "steelblue",
) -> None:
    """Bar chart of a single metric broken down by severity level."""
    levels = sorted(sev_metrics)
    values = [sev_metrics[lvl][metric] for lvl in levels]
    fig, ax = plt.subplots(figsize=(max(6, len(levels)), 4))
    ax.bar([str(lvl) for lvl in levels], values, color=color)
    ax.set_xlabel("Severity Level")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} per Severity Level")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"{metric.upper()} by severity → {out_path}")


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    model: WhisperIntelligibilityModel,
    loader,
    processor: WhisperProcessor,
    device: torch.device,
    inference_only: bool,
    better_ear: bool = False,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Run the model over a full dataset split and collect predictions.

    Parameters
    ----------
    model : WhisperIntelligibilityModel
    loader : DataLoader
        Must supply stereo batches [B, 2, T] when better_ear=True.
    processor : WhisperProcessor
    device : torch.device
    inference_only : bool
        If True, ground-truth scores are not read; targets_arr is None.
    better_ear : bool
        If True, score left and right channels separately and return
        max(score_L, score_R) per sample.

    Returns
    -------
    preds_arr    : [N]  predicted scores in (0, 1)
    targets_arr  : [N]  ground-truth scores, or None if inference_only=True
    severities   : list of per-sample severity labels
    ground_truths: list of per-sample transcript strings
    """
    model.eval()
    all_preds: list[float] = []
    all_targets: list[float] = []
    all_severities: list = []
    all_gts: list[str] = []

    pbar = tqdm(loader, desc="Inference", unit="batch", dynamic_ncols=True)
    for audio, _audio2, _sev_str, severity, ground_truth, score in pbar:
        score_np = None if inference_only else score

        dtype = next(model.parameters()).dtype
        if better_ear:
            feats_L, feats_R, scores = prepare_batch_whisper_stereo(
                audio, processor, device, score_np
            )
            preds = torch.maximum(model(feats_L.to(dtype)), model(feats_R.to(dtype)))
        else:
            feats, scores = prepare_batch_whisper(audio, processor, device, score_np)
            preds = model(feats.to(next(model.parameters()).dtype))

        all_preds.extend(preds.cpu().numpy())
        if not inference_only:
            all_targets.extend(scores.cpu().numpy())
            all_severities.extend(
                severity.tolist()
                if isinstance(severity, np.ndarray)
                else list(severity)
            )
            all_gts.extend(
                ground_truth
                if isinstance(ground_truth, (list, tuple))
                else list(ground_truth)
            )

    preds_arr = np.array(all_preds, dtype=np.float32)
    targets_arr = np.array(all_targets, dtype=np.float32) if all_targets else None
    return preds_arr, targets_arr, all_severities, all_gts


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def evaluate(cfg: DictConfig, test_loader) -> dict:
    """
    Load a checkpoint and evaluate (or run inference) on a test set.

    The model is always reconstructed from the config embedded in the
    checkpoint so the architecture is guaranteed to match the saved weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.test.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Device    : {device}")
    log.info(f"Checkpoint: {cfg.test.checkpoint}")

    # -- Load checkpoint ------------------------------------------------
    ckpt = torch.load(cfg.test.checkpoint, map_location=device)
    saved_cfg = ckpt["cfg"]

    model = WhisperIntelligibilityModel(
        backbone=saved_cfg["model"]["backbone"],
        hidden_dim=saved_cfg["model"]["hidden_dim"],
        use_encoder_layers=saved_cfg["model"].get("use_encoder_layers") or None,
        freeze_backbone=False,
        attn_heads=saved_cfg["model"]["attn_heads"],
        dropout=saved_cfg["model"].get("dropout", 0.1),
    ).to(device)

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
        log.warning(f"Missing non-backbone keys: {non_bb_missing}")
    if unexpected:
        log.warning(f"Unexpected keys in checkpoint: {unexpected}")
    log.info(
        f"Loaded checkpoint — epoch {ckpt['epoch']}  "
        f"backbone={ckpt.get('saved_backbone', '?')}  "
        f"missing={len(missing)}  unexpected={len(unexpected)}"
    )

    if cfg.test.fp16 and device.type == "cuda":
        model = model.half()
        log.info("Running in FP16")

    # -- Processor -------------------------------------------------------
    model_id = WhisperIntelligibilityModel.SUPPORTED_MODELS[
        saved_cfg["model"]["backbone"]
    ]
    processor = WhisperProcessor.from_pretrained(model_id)

    # -- Inference --------------------------------------------------------
    better_ear: bool = cfg.test.get("better_ear", False)
    mode = (
        "inference"
        if cfg.test.inference_only
        else ("evaluation (better-ear)" if better_ear else "evaluation (mono)")
    )
    log.info(f"Running {mode} on {len(test_loader.dataset)} samples...")
    t0 = time.time()

    preds, targets, severities, ground_truths = run_inference(
        model,
        test_loader,
        processor,
        device,
        inference_only=cfg.test.inference_only,
        better_ear=better_ear,
    )
    elapsed = time.time() - t0
    log.info(
        f"{len(preds)} samples in {elapsed:.1f}s  "
        f"({len(preds) / elapsed:.1f} samples/sec)"
    )

    # -- Metrics (evaluation mode only) ------------------------------------
    results: dict = {}

    if not cfg.test.inference_only:
        overall = compute_metrics(preds, targets)
        results["overall"] = overall
        log.info(
            f"\n{'─' * 52}\n"
            f"  MSE  : {overall['mse']:.4f}\n"
            f"  RMSE : {overall['rmse']:.4f}\n"
            f"  MAE  : {overall['mae']:.4f}\n"
            f"  PCC  : {overall['pcc']:.4f}\n"
            f"  SCC  : {overall['scc']:.4f}\n"
            f"{'─' * 52}"
        )

        sev_metrics = metrics_by_severity(preds, targets, severities)
        if sev_metrics:
            results["by_severity"] = sev_metrics
            log.info("Per-severity breakdown:")
            for level, m in sorted(sev_metrics.items()):
                log.info(
                    f"  Severity {level:>3d} — "
                    f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  PCC={m['pcc']:.4f}"
                )

        plot_scatter(preds, targets, out_dir / "scatter_pred_vs_gt.png")
        plot_error_histogram(
            preds, targets, overall["rmse"], out_dir / "error_histogram.png"
        )
        if sev_metrics:
            plot_metric_by_severity(
                sev_metrics, "mae", out_dir / "mae_by_severity.png", color="steelblue"
            )
            plot_metric_by_severity(
                sev_metrics, "rmse", out_dir / "rmse_by_severity.png", color="tomato"
            )

    # -- Save results.json (evaluation mode only) -------------------------
    if results:
        results_path = out_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2))
        log.info(f"Metrics JSON       → {results_path}")

    # -- Save CSV ---------------------------------------------------------
    signal_ids = [str(r["signal"]) for r in test_loader.dataset.records]

    if cfg.test.inference_only:
        # Save predictions either in [0, 1] (range_01=true) or [0, 100] (range_01=false)
        if not cfg.test.get("range_01", False):
            preds *= 100

        csv_path = out_dir / "predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for sig_id, pred in zip(signal_ids, preds):
                writer.writerow([sig_id, f"{pred:.6f}"])
    else:
        csv_path = out_dir / "predictions_complete.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "index",
                    "signal_id",
                    "ground_truth_transcript",
                    "severity",
                    "target_score",
                    "predicted_score",
                    "abs_error",
                ]
            )
            for i, (sig_id, gt, sev, tgt, pred) in enumerate(
                zip(signal_ids, ground_truths, severities, targets, preds)
            ):
                sev_str = (
                    json.dumps(sev.tolist())
                    if isinstance(sev, np.ndarray)
                    else str(sev)
                )
                writer.writerow(
                    [
                        i,
                        sig_id,
                        gt,
                        sev_str,
                        f"{tgt:.6f}",
                        f"{pred:.6f}",
                        f"{abs(tgt - pred):.6f}",
                    ]
                )

    log.info(f"Predictions CSV    → {csv_path}")
    log.info(f"All outputs in     → {out_dir}")
    return results


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    better_ear: bool = cfg.test.get("better_ear", False)
    test_ds = LyricIntelligibilityDataset(
        root_path=cfg.data.root_path,
        split=cfg.test.split,
        mono=not better_ear,
    )
    test_loader = build_dataloader(
        test_ds,
        batch_size=cfg.test.get("batch_size", cfg.data.batch_size),
        shuffle=False,
        num_workers=(
            0
            if not torch.cuda.is_available()
            else min(cfg.data.get("num_workers", 4), os.cpu_count() or 1)
        ),
    )
    evaluate(cfg, test_loader)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
