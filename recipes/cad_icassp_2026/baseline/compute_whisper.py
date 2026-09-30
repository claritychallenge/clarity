"""Compute the Whisper correctness scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import whisper
from omegaconf import DictConfig
from torch.nn import Module
from tqdm import tqdm

from clarity.utils.file_io import read_jsonl, write_jsonl, write_signal
from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    load_mixture,
)
from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer

logger = logging.getLogger(__name__)


def compute_asr_for_signal(
    cfg: DictConfig, record: dict, signal: np.ndarray, asr_model: Module
) -> float:
    """Compute the correctness score for a given signal.

    Args:

        cfg (DictConfig): configuration object.
        record (dict): the metadata dict for the signal.
        signal (np.ndarray): the signal to compute the score for.
        asr_model (Module): the ASR model to use for transcription.

    Returns:

        float: correctness score
    """
    reference = record["prompt"]

    score_left = compute_correctness(
        signal[:, 0],
        cfg.data.sample_rate,
        reference,
        asr_model,
        cfg.baseline.contractions_file,
    )
    score_right = compute_correctness(
        signal[:, 1],
        cfg.data.sample_rate,
        reference,
        asr_model,
        cfg.baseline.contractions_file,
    )

    return np.max([score_left, score_right])


def compute_correctness(
    signal: np.ndarray,
    sample_rate: int,
    reference: str,
    asr_model: Module,
    contraction_file: str,
) -> float:
    """Compute the correctness score for a given signal.

    Args:
        signal (np.ndarray): the signal to compute the score for
        sample_rate (int): the sample rate of the signal
        reference (str): the reference transcription
        asr_model (Module): the ASR model to use for transcription
        contraction_file (str): path to the contraction file for the scorer

    Returns:
        float: correctness score.
    """
    scorer = SentenceScorer(contraction_file)

    # create a temporary file to store the signal as flac
    # for Whisper to open it
    path_temp = Path("temp.flac")
    write_signal(
        filename=path_temp, signal=signal, sample_rate=sample_rate, floating_point=False
    )

    # Run Whisper ASR
    hypothesis = asr_model.transcribe(
        str(path_temp), fp16=False, language="en", temperature=0.0
    )["text"]

    # Score the transcription
    results = scorer.score([reference], [hypothesis])
    total_words = results.substitutions + results.deletions + results.hits

    # Delete temporal file
    Path(path_temp).unlink()

    return results.hits / total_words


def run_asr_from_mixture(
    dataroot: Path, records: list, results_file: Path, cfg: DictConfig
) -> None:
    """Load the mixture signal for a given record.

    Args:

        dataroot (Path): the root path to the dataset.
        records (list): list of records to process.
        results_file (Path): path to the results file.
        cfg (DictConfig): configuration object.
    """
    # Prepare dnn models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    asr_model = whisper.load_model(cfg.baseline.whisper_version, device=device)

    # Iterate through the signals that need scoring
    for record in tqdm(records):
        signal_name = record["signal"]

        # Load mixture
        signal_to_whisper, _ = load_mixture(dataroot, record, cfg)

        # Compute ASR
        correct = compute_asr_for_signal(cfg, record, signal_to_whisper, asr_model)

        # Results are appended to the results file to allow interruption
        result = {"signal": signal_name, f"{cfg.baseline.system}": correct}
        write_jsonl(str(results_file), [result])


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_whisper(cfg: DictConfig) -> None:
    """Run the Whisper to compute correctness hits/total words."""
    assert cfg.baseline.name == "whisper"

    logger.info(f"Running {cfg.baseline.system} baseline on {cfg.split} set...")

    # Load the set of signal for which we need to compute scores
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset

    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"

    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    total_records = len(records)
    # Load existing results file if present
    batch_str = (
        f".{cfg.baseline.batch}_{cfg.baseline.n_batches}"
        if cfg.baseline.n_batches > 1
        else ""
    )

    results_file = Path(
        f"{cfg.data.dataset}.{cfg.split}.{cfg.baseline.system}{batch_str}.jsonl"
    )
    results = read_jsonl(str(results_file)) if results_file.exists() else []
    results_index = {result["signal"]: result for result in results}

    # Find signals for which we don't have scores
    records = [
        record for record in records if record["signal"] not in results_index.keys()
    ]
    records = records[cfg.baseline.batch - 1 :: cfg.baseline.n_batches]

    # Iterate over the signals that need scoring
    logger.info(f"Computing scores for {len(records)} out of {total_records} signals")

    run_asr_from_mixture(dataroot, records, results_file, cfg)


if __name__ == "__main__":
    run_compute_whisper()
