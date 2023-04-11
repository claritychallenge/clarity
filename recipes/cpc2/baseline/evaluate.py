"""Evaluate the predictions against the ground truth correctness values"""
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import kendalltau, pearsonr

logger = logging.getLogger(__name__)


def rmse_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the root mean squared error between two arrays"""
    return np.sqrt(np.mean((x - y) ** 2))


def ncc_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the normalized cross correlation between two arrays"""
    return pearsonr(x, y)[0]


def kt_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Kendall's tau correlation between two arrays"""
    return kendalltau(x, y)[0]


def std_err(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the standard error between two arrays"""
    return np.std(x - y) / np.sqrt(len(x))


def compute_scores(predictions, labels) -> dict:
    """Compute the scores for the predictions"""
    return {
        "RMSE": rmse_score(predictions, labels),
        "Std": std_err(predictions, labels),
        "NCC": ncc_score(predictions, labels),
        "KT": kt_score(predictions, labels),
    }


# pylint: disable = no-value-for-parameter
@hydra.main(config_path=".", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate the predictions against the ground truth correctness values"""

    # Load the intelligibility dataset records
    dataset_filename = Path(cfg.path.metadata_dir) / f"{cfg.dataset}.json"
    with open(dataset_filename, encoding="utf-8") as fp:
        records = json.load(fp)
    record_index = {record["signal"]: record for record in records}

    # Load the predictions
    df = pd.read_csv(
        f"{cfg.dataset}.predict.csv", names=["signal", "predicted"], header=0
    )

    df["correctness"] = [record_index[signal]["correctness"] for signal in df.signal]

    # Compute and report the scores
    scores = compute_scores(df["predicted"], df["correctness"])

    with open(f"{cfg.dataset}.evaluate.jsonl", "a", encoding="utf-8") as fp:
        fp.write(json.dumps(scores) + "\n")

    # Output the scores to the console
    print(scores)


if __name__ == "__main__":
    evaluate()
