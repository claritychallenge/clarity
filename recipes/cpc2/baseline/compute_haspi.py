""" Compute the HASPI scores. """
from __future__ import annotations

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


def set_seed_with_string(seed_string: str) -> None:
    """Set the random seed with a string."""
    md5_int = int(hashlib.md5(seed_string.encode("utf-8")).hexdigest(), 16) % (10**8)
    np.random.seed(md5_int)


def parse_cec2_signal_name(signal_name: str) -> tuple[str, str, str]:
    """Parse the CEC2 signal name."""
    # e.g. S0001_L0001_E001_hr -> S0001, L0001, E001_hr
    scene, listener, *system = signal_name.split("_")
    if scene == "" or listener == "" or system == []:
        raise ValueError(f"Invalid CEC2 signal name: {signal_name}")
    return scene, listener, "_".join(system)


def compute_haspi_for_signal(signal_name: str, path: dict) -> float:
    """Compute the HASPI score for a given signal.

    Args:
        signal (str): name of the signal to process
        path (dict): paths to the signals and metadata, as defined in the config

    Returns:
        float: HASPI score
    """

    scene, listener_id, _ = parse_cec2_signal_name(signal_name)

    # Retrieve the listeners
    listener_dict = Listener.load_listener_dict(
        Path(path["metadata_dir"]) / "listeners.json"
    )
    listener = listener_dict[listener_id]

    # Retrieve signals and convert to float32 between -1 and 1
    sr_proc, proc = wavfile.read(Path(path["signal_dir"]) / f"{signal_name}.wav")
    sr_ref, ref = wavfile.read(Path(path["scene_dir"]) / f"{scene}_target_ref.wav")
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
@hydra.main(config_path=".", config_name="config")
def run_calculate_haspi(cfg: DictConfig) -> None:
    """Run the HASPI score computation."""
    # Load the set of signal for which we need to compute scores
    dataset_filename = Path(cfg.path.metadata_dir) / f"{cfg.dataset}.json"
    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

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
        haspi = compute_haspi_for_signal(signal_name, cfg.path)

        # Results are appended to the results file to allow interruption
        result = {"signal": signal_name, "haspi": haspi}
        write_jsonl(str(results_file), [result])


if __name__ == "__main__":
    run_calculate_haspi()
