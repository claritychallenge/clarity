"""Compute the STOI scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pystoi import stoi as compute_stoi
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from tqdm import tqdm

from clarity.utils.file_io import read_jsonl, write_jsonl
from clarity.utils.flac_encoder import read_flac_signal
from clarity.utils.signal_processing import resample
from recipes.cad_icassp_2026.baseline_stoi.shared_predict_utils import (
    input_align,
    load_vocals,
)

logger = logging.getLogger(__name__)


def compute_stoi_for_signal(
    cfg: DictConfig, record: dict, data_root: str, estimated_vocals: np.ndarray
) -> float:
    """Compute the stoi score for a given signal.

    Args:
        cfg (DictConfig): configuration object
        record (dict): the metadata dict for the signal
        data_root (str): root path to the dataset
        estimated_vocals (np.ndarray): estimated vocals signal

    Returns:
        float: stoi score
    """
    signal_name = record["signal"]

    # Load processed signal
    signal_path = (
        Path(data_root) / "audio" / cfg.split / "signals" / f"{signal_name}.flac"
    )
    signal, proc_sr = read_flac_signal(signal_path)
    if proc_sr != cfg.data.sample_rate:
        logger.info(f"Resampling {signal_path} to {cfg.data.sample_rate} Hz")
        signal = resample(signal, proc_sr, cfg.data.sample_rate)

    signal_norm_factor = np.max(np.abs(signal))
    signal /= signal_norm_factor
    estimated_vocals /= signal_norm_factor

    # Compute STOI score
    stoi_score_left = compute_single_stoi(
        estimated_vocals[:, 0], signal[:, 0], proc_sr, cfg.compute.stoi_sample_rate
    )
    stoi_score_right = compute_single_stoi(
        estimated_vocals[:, 1], signal[:, 1], proc_sr, cfg.compute.stoi_sample_rate
    )

    return np.max([stoi_score_left, stoi_score_right])


def compute_single_stoi(
    reference: np.ndarray, processed: np.ndarray, fsamp: int, stoi_fsamp: int = 10000
) -> float:
    """Compute the STOI score between a reference and processed signal.

    Args:
        reference (np.ndarray): Reference signal.
        processed (np.ndarray): Processed signal.
        fsamp (int): Sampling frequency.
        stoi_fsamp (int): Sampling frequency for STOI computation. Default is 10000 Hz.

    Returns:
        float: STOI score.
    """
    reference_side = resample(reference, fsamp, stoi_fsamp)
    processed_side = resample(processed, fsamp, stoi_fsamp)

    reference_side, processed_side = input_align(
        reference_side, processed_side, fsamp=int(stoi_fsamp)
    )
    stoi_score = compute_stoi(reference_side, processed_side, int(stoi_fsamp))
    return stoi_score


# pylint: disable = no-value-for-parameter
@hydra.main(config_path=".", config_name="config", version_base=None)
def run_compute_stoi(cfg: DictConfig) -> None:
    """Run the STOI score computation."""
    # Load the set of signal for which we need to compute scores
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset

    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"

    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    total_records = len(records)
    # Load existing results file if present
    batch_str = (
        f".{cfg.compute.batch}_{cfg.compute.n_batches}"
        if cfg.compute.n_batches > 1
        else ""
    )

    results_file = (
        Path("..")
        / "precomputed_stoi"
        / f"{cfg.data.dataset}.{cfg.split}.stoi{batch_str}.jsonl"
    )
    results = read_jsonl(str(results_file)) if results_file.exists() else []
    results_index = {result["signal"]: result for result in results}

    # Find signals for which we don't have scores
    records = [
        record for record in records if record["signal"] not in results_index.keys()
    ]
    records = records[cfg.compute.batch - 1 :: cfg.compute.n_batches]

    # Prepare audio source separation model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    separation_model = HDEMUCS_HIGH_MUSDB_PLUS.get_model()
    separation_model.to(device)

    # Iterate over the signals that need scoring
    logger.info(f"Computing scores for {len(records)} out of {total_records} signals")
    if cfg.separator.keep_vocals:
        logger.info(f"Saving estimated vocals. If exist, they will not be recomputed.")

    for record in tqdm(records):
        signal_name = record["signal"]
        # Load unprocessed signal to estimate vocals
        estimated_vocals = load_vocals(
            dataroot, record, cfg, separation_model, device=device
        )
        stoi = compute_stoi_for_signal(cfg, record, dataroot, estimated_vocals)

        # Results are appended to the results file to allow interruption
        result = {"signal": signal_name, "stoi": stoi}
        write_jsonl(str(results_file), [result])


if __name__ == "__main__":
    run_compute_stoi()
