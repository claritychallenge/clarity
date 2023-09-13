"""Evaluate the enhanced signals using the HAAQI metric."""
from __future__ import annotations

# pylint: disable=too-many-locals
# pylint: disable=import-error
import csv
import hashlib
import itertools
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.io import wavfile

from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.audiogram import Listener
from clarity.utils.signal_processing import compute_rms

logger = logging.getLogger(__name__)


class ResultsFile:
    """A utility class for writing results to a CSV file.

    Attributes:
        file_name (str): The name of the file to write results to.
    """

    def __init__(self, file_name: str):
        """Initialize the ResultsFile instance.

        Args:
            file_name (str): The name of the file to write results to.
        """
        self.file_name = file_name

    def write_header(self):
        """Write the header row to the CSV file."""
        with open(self.file_name, "w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.writer(
                csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [
                    "song",
                    "listener",
                    "score",
                    "left_bass",
                    "right_bass",
                    "left_drums",
                    "right_drums",
                    "left_other",
                    "right_other",
                    "left_vocals",
                    "right_vocals",
                ]
            )

    def add_result(
        self,
        listener_id: str,
        song: str,
        score: float,
        instruments_scores: dict[str, float],
    ):
        """Add a result to the CSV file.

        Args:
            listener_id (str): The name of the listener who submitted the result.
            song (str): The name of the song that the result is for.
            score (float): The combined score for the result.
            instruments_scores (dict): A dictionary of scores for each instrument
                channel in the result.
        """
        logger.info(f"The combined score is {score}")

        with open(self.file_name, "a", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.writer(
                csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [
                    song,
                    listener_id,
                    str(score),
                    str(instruments_scores["left_bass"]),
                    str(instruments_scores["right_bass"]),
                    str(instruments_scores["left_drums"]),
                    str(instruments_scores["right_drums"]),
                    str(instruments_scores["left_other"]),
                    str(instruments_scores["right_other"]),
                    str(instruments_scores["left_vocals"]),
                    str(instruments_scores["right_vocals"]),
                ]
            )


def set_song_seed(song: str) -> None:
    """Set a seed that is unique for the given song"""
    song_encoded = hashlib.md5(song.encode("utf-8")).hexdigest()
    song_md5 = int(song_encoded, 16) % (10**8)
    np.random.seed(song_md5)


def make_song_listener_list(
    songs: list[str], listeners: dict[str, Listener], small_test: bool = False
) -> list[tuple[str, str]]:
    """Make the list of scene-listener pairing to process"""
    song_listener_pairs = list(itertools.product(songs, listeners.keys()))

    if small_test:
        song_listener_pairs = song_listener_pairs[::15]

    return song_listener_pairs


def _evaluate_song_listener(
    song: str,
    listener: Listener,
    config: DictConfig,
    split_dir: str,
    enhanced_folder: Path,
) -> tuple[float, dict]:
    """Evaluate a single song-listener pair

    Args:
        song (str): The name of the song to evaluate.
        listener (str): The name of the listener to evaluate.
        config (DictConfig): The configuration object.
        split_dir (str): The name of the split directory.
        enhanced_folder (Path): The path to the folder containing the enhanced signals.

    Returns:
        combined_score (float): The combined score for the result.
        per_instrument_score (dict): A dictionary of scores for each
            instrument channel in the result.

    """

    logger.info(f"Evaluating {song} for {listener.id}")

    if config.evaluate.set_random_seed:
        set_song_seed(song)

    per_instrument_score = {}
    for instrument in [
        "drums",
        "bass",
        "other",
        "vocals",
    ]:
        logger.info(f"...evaluating {instrument}")

        # Read instrument reference signal
        sample_rate_reference_signal, reference_signal = wavfile.read(
            Path(config.path.music_dir) / split_dir / song / f"{instrument}.wav"
        )
        reference_signal = (reference_signal / 32768.0).astype(np.float32)

        # Load enhanced instrument signals
        # Load left channel
        sample_rate_left_enhanced_signal, left_enhanced_signal = wavfile.read(
            enhanced_folder
            / f"{listener.id}"
            / f"{song}"
            / f"{listener.id}_{song}_left_{instrument}.wav"
        )
        left_enhanced_signal = (left_enhanced_signal / 32768.0).astype(np.float32)

        # Load right channel
        sample_rate_right_enhanced_signal, right_enhanced_signal = wavfile.read(
            enhanced_folder
            / f"{listener.id}"
            / f"{song}"
            / f"{listener.id}_{song}_right_{instrument}.wav"
        )
        right_enhanced_signal = (right_enhanced_signal / 32768.0).astype(np.float32)

        assert (
            sample_rate_reference_signal
            == sample_rate_left_enhanced_signal
            == sample_rate_right_enhanced_signal
            == config.sample_rate
        )

        per_instrument_score[f"left_{instrument}"] = compute_haaqi(
            left_enhanced_signal,
            reference_signal[:, 0],
            config.sample_rate,
            config.sample_rate,
            listener.audiogram_left,
            equalisation=1,
            level1=65 - 20 * np.log10(compute_rms(reference_signal[:, 0])),
        )
        per_instrument_score[f"right_{instrument}"] = compute_haaqi(
            right_enhanced_signal,
            reference_signal[:, 1],
            config.sample_rate,
            config.sample_rate,
            listener.audiogram_right,
            equalisation=1,
            level1=65 - 20 * np.log10(compute_rms(reference_signal[:, 1])),
        )

    # Compute the combined score
    combined_score = np.mean(list(per_instrument_score.values()))

    return float(combined_score), per_instrument_score


@hydra.main(config_path="", config_name="config")
def run_calculate_aq(config: DictConfig) -> None:
    """Evaluate the enhanced signals using the HAAQI-RMS metric."""
    # Load test songs
    with open(config.path.music_valid_file, encoding="utf-8") as fp:
        songs = json.load(fp)
    songs = pd.DataFrame.from_dict(songs)

    # Load listener data
    listener_dict = Listener.load_listener_dict(config.path.listeners_valid_file)

    enhanced_folder = Path("enhanced_signals")
    logger.info(f"Evaluating from {enhanced_folder} directory")

    results_file = ResultsFile(
        f"scores_{config.evaluate.batch + 1}-{config.evaluate.batch_size}.csv"
    )
    results_file.write_header()

    song_listener_pair = make_song_listener_list(
        songs["Track Name"].tolist(), listener_dict, config.evaluate.small_test
    )

    song_listener_pair = song_listener_pair[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    for song, listener_id in song_listener_pair:
        split_dir = "train"
        if songs[songs["Track Name"] == song]["Split"].tolist()[0] == "test":
            split_dir = "test"
        listener = listener_dict[listener_id]
        combined_score, per_instrument_score = _evaluate_song_listener(
            song,
            listener,
            config,
            split_dir,
            enhanced_folder,
        )
        results_file.add_result(
            listener.id,
            song,
            score=combined_score,
            instruments_scores=per_instrument_score,
        )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    run_calculate_aq()
