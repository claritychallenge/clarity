""" Compute the HASPI scores. """
# pylint: disable=too-many-locals

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

logger = logging.getLogger(__name__)


def read_jsonl(filename: str) -> list[dict]:
    """Read a jsonl file into a list of dictionaries."""
    with open(filename, "r", encoding="utf-8") as fp:
        records = [json.loads(line) for line in fp]
    return records


def write_jsonl(filename: str, records: list[dict]) -> None:
    """Write a list of dictionaries to a jsonl file."""
    with open(filename, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")


def set_seed_with_string(seed_string: str) -> None:
    """Set the random seed with a string."""
    md5_int = int(hashlib.md5(seed_string.encode("utf-8")).hexdigest(), 16) % (10**8)
    np.random.seed(md5_int)


def compute_haspi_for_signal(signal_name: str, clarity_data_dir: str) -> float:
    """Compute the HASPI score for a given signal.

    Args:
        signal (str): name of the signal to process

    Returns:
        float: HASPI score
    """

    # Parse signal name
    scene, listener, *_ = signal_name.split("_")

    # Define paths
    data_dir = Path(clarity_data_dir) / "clarity_data"
    metadata_dir = data_dir / "metadata"
    signal_dir = data_dir / "HA_outputs" / "signals" / "CEC2"
    scene_dir = data_dir / "scenes" / "CEC2"

    # Retrieve audiograms
    with open(metadata_dir / "listeners.json", "r", encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)
    cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
    audiogram_left = np.array(listener_audiograms[listener]["audiogram_levels_l"])
    audiogram_right = np.array(listener_audiograms[listener]["audiogram_levels_r"])

    # Retrieve signals and convert to float32 between -1 and 1
    fs_proc, proc = wavfile.read(Path(signal_dir) / f"{signal_name}.wav")
    fs_ref, ref = wavfile.read(scene_dir / f"{scene}_target_ref.wav")
    assert fs_ref == fs_proc

    proc = proc / 32768.0
    ref = ref / 32768.0

    haspi_score = haspi_v2_be(
        reference_left=ref[:, 0],
        reference_right=ref[:, 1],
        processed_left=proc[:, 0],
        processed_right=proc[:, 1],
        fs_signal=fs_proc,
        audiogram_left=audiogram_left,
        audiogram_right=audiogram_right,
        audiogram_cfs=cfs,
    )

    return haspi_score


# pylint: disable = no-value-for-parameter
@hydra.main(config_path=".", config_name="config")
def run_calculate_haspi(cfg: DictConfig) -> None:
    """Run the HASPI score computation."""
    # Load the set of signal for which we need to compute scores
    dataset_filename = Path(cfg.path.metadata_dir) / f"{cfg.dataset}.json"
    with open(dataset_filename, "r", encoding="utf-8") as fp:
        records = json.load(fp)

    # Load existing results file if present
    results_file = Path(cfg.path.results_dir) / f"{cfg.results_file}.jsonl"
    results = read_jsonl(str(results_file)) if Path(results_file).exists() else []
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
        haspi = compute_haspi_for_signal(signal_name, cfg.path.clarity_data_dir)
        results.append({"signal": signal_name, "haspi": haspi})

    # Write out modified results
    exp_dir = Path(cfg.path.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    batch_str = (
        f".{cfg.compute_haspi.batch}_{cfg.compute_haspi.n_batches}"
        if cfg.compute_haspi.n_batches > 1
        else ""
    )
    results_outfile = exp_dir / f"{cfg.results_file}{batch_str}.jsonl"
    write_jsonl(str(results_outfile), results)


if __name__ == "__main__":
    run_calculate_haspi()
