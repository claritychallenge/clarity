"""Compute the HASPI scores."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from clarity.evaluator.haspi import haspi_v2_be
from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


# Standard audiograms for each severity level
#
# These are based on those defined in the Clarity library utils/audiogram.py
# which originated from ones used by Moore, Stone, Baer and Glasberg.
#
# See utils/audiogram.py for more details.

MILD_LISTENER = {
    "name": "Mild Listener",
    "audiogram_cfs": np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000]),
    "audiogram_levels_l": np.array([10, 15, 19, 25, 28, 31, 35, 38]),
    "audiogram_levels_r": np.array([10, 15, 19, 25, 28, 31, 35, 38]),
}

MOD_LISTENER = {
    "name": "Moderate Listener",
    "audiogram_cfs": np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000]),
    "audiogram_levels_l": np.array([20, 20, 25, 35, 40, 45, 50, 55]),
    "audiogram_levels_r": np.array([20, 20, 25, 35, 40, 45, 50, 55]),
}

MOD_SEV_LISTENER = {
    "name": "Moderately Severe Listener",
    "audiogram_cfs": np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000]),
    "audiogram_levels_l": np.array([19, 28, 40, 52, 56, 58, 58, 63]),
    "audiogram_levels_r": np.array([19, 28, 40, 52, 56, 58, 58, 63]),
}


def set_seed_with_string(seed_string: str) -> None:
    """Set the random seed with a string."""
    md5_int = int(hashlib.md5(seed_string.encode("utf-8")).hexdigest(), 16) % (10**8)
    np.random.seed(md5_int)


def parse_signal_name(signal_name: str) -> dict:
    """Parse the signal name."""
    # e.g. CEC2_E032_S09318_L0254.wav
    cec, system, scene, listener = signal_name.split("_")
    if scene == "" or listener == "" or system == []:
        raise ValueError(f"Invalid CEC2 signal name: {signal_name}")
    info = {
        "scene": scene,
        "listener": listener,
        "system": system,
        "cec": cec,
    }
    return info


def compute_haspi_for_signal(signal_name: str, data_root: str, split: str) -> float:
    """Compute the HASPI score for a given signal.

    Args:
        signal (str): name of the signal to process
        signal_dir (str): paths to where the HA output signals are stored
        ref_dir (str): path to where the reference signals are stored

    Returns:
        float: HASPI score
    """

    signal_dir = Path(data_root) / split / "signals"
    ref_dir = Path(data_root) / split / "references"

    listener_data_dict = {
        "Mild": MILD_LISTENER,
        "Moderate": MOD_LISTENER,
        "Moderately severe": MOD_SEV_LISTENER,
    }

    signal_parts = parse_signal_name(signal_name)
    listener_id = signal_parts["listener"]
    scene = signal_parts["scene"]
    cec = signal_parts["cec"]

    with open(Path(data_root) / "metadata" / "listeners.csv", encoding="utf8") as f:
        listener_dict = csv.DictReader(f)
    listener_severity_dict = {
        row["listener_id"]: row["severity"] for row in listener_dict
    }

    listener_severity = listener_severity_dict[listener_id]
    listener_data = listener_data_dict[listener_severity]
    listener = Listener.from_dict(listener_data)

    # Retrieve signals and convert to float32 between -1 and 1
    sr_proc, proc = wavfile.read(Path(signal_dir) / f"{signal_name}.wav")
    sr_ref, ref = wavfile.read(Path(ref_dir) / f"{cec}_{scene}_ref.wav")
    assert sr_ref == sr_proc

    proc = proc / 32768.0
    ref = ref / 32768.0

    # Compute haspi score using library code
    haspi_score = haspi_v2_be(
        reference_left=ref[:, 0],
        reference_right=ref[:, 1],
        processed_left=proc[:, 0],
        processed_right=proc[:, 1],
        sample_rate=sr_proc,
        listener=listener,
    )

    return haspi_score


# pylint: disable = no-value-for-parameter
@hydra.main(config_path=".", config_name="config", version_base=None)
def run_calculate_haspi(cfg: DictConfig) -> None:
    """Run the HASPI score computation."""
    # Load the set of signal for which we need to compute scores
    dataset_filename = Path(cfg.path.metadata_dir) / f"CPC3.{cfg.split}.json"

    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    dataroot = Path(cfg.path.clarity_data_dir) / cfg.dataset

    # Load existing results file if present
    batch_str = (
        f".{cfg.compute_haspi.batch}_{cfg.compute_haspi.n_batches}"
        if cfg.compute_haspi.n_batches > 1
        else ""
    )
    results_file = Path(f"{cfg.dataset}.haspi{batch_str}.jsonl")
    results = read_jsonl(str(results_file)) if results_file.exists() else []
    results_index = {result["signal"]: result for result in results}

    # Find signals for which we don't have scores
    records = [
        record for record in records if record["signal"] not in results_index.keys()
    ]
    records = records[cfg.compute_haspi.batch - 1 :: cfg.compute_haspi.n_batches]

    # Iterate over the signals that need scoring
    logger.info(f"Computing scores for {len(records)} signals")
    for record in tqdm(records):
        signal_name = record["signal"]
        if cfg.compute_haspi.set_random_seed:
            set_seed_with_string(signal_name)
        haspi = compute_haspi_for_signal(signal_name, dataroot, cfg.split)

        # Results are appended to the results file to allow interruption
        result = {"signal": signal_name, "haspi": haspi}
        write_jsonl(str(results_file), [result])


if __name__ == "__main__":
    run_calculate_haspi()
