"""Evaluate the enhanced signals using the HAAQI metric."""
from __future__ import annotations

# pylint: disable=too-many-locals
# pylint: disable=import-error
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

from clarity.evaluator.ha import HaaqiV1
from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.audiogram import Listener
from clarity.utils.flac_encoder import read_flac_signal
from clarity.utils.results_support import ResultsFile
from clarity.utils.signal_processing import compute_rms, resample

logger = logging.getLogger(__name__)


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
    haaqi_left: HaaqiV1,
    haaqi_right: HaaqiV1,
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

        # Read left instrument enhanced
        left_enhanced_signal, sample_rate_left_enhanced_signal = read_flac_signal(
            enhanced_folder
            / f"{listener.id}"
            / f"{song}"
            / f"{listener.id}_{song}_left_{instrument}.flac"
        )

        # Read right instrument enhanced
        right_enhanced_signal, sample_rate_right_enhanced_signal = read_flac_signal(
            enhanced_folder
            / f"{listener.id}"
            / f"{song}"
            / f"{listener.id}_{song}_right_{instrument}.flac"
        )

        if (
            sample_rate_left_enhanced_signal
            != sample_rate_right_enhanced_signal
            != config.stem_sample_rate
        ):
            raise ValueError(
                "The sample rates of the left and right enhanced signals are not "
                "the same"
            )

        if sample_rate_reference_signal != config.sample_rate:
            raise ValueError(
                f"The sample rate of the reference signal is not {config.sample_rate}"
            )

        per_instrument_score[f"left_{instrument}"] = haaqi_left.process(
            reference=resample(
                reference_signal[:, 0],
                sample_rate_reference_signal,
                config.stem_sample_rate,
            ),
            reference_sample_rate=config.stem_sample_rate,
            enhanced=left_enhanced_signal,
            enhanced_sample_rate=config.stem_sample_rate,
            level1=65 - 20 * np.log10(compute_rms(reference_signal[:, 0])),
        )

        per_instrument_score[f"right_{instrument}"] = haaqi_right.process(
            reference=resample(
                reference_signal[:, 1],
                sample_rate_reference_signal,
                config.stem_sample_rate,
            ),
            reference_sample_rate=config.stem_sample_rate,
            enhanced=right_enhanced_signal,
            enhanced_sample_rate=config.stem_sample_rate,
            level1=65 - 20 * np.log10(compute_rms(reference_signal[:, 1])),
        )

    # Compute the combined score
    combined_score = np.mean(list(per_instrument_score.values()))

    return float(combined_score), per_instrument_score


@hydra.main(config_path="", config_name="config")
def run_calculate_aq(config: DictConfig) -> None:
    """Evaluate the enhanced signals using the HAAQI-RMS metric."""
    # Load test songs
    with open(config.path.music_file, encoding="utf-8") as fp:
        songs = json.load(fp)
    songs_df = pd.DataFrame.from_dict(songs)

    # Load listener data
    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    enhanced_folder = Path("enhanced_signals")
    logger.info(f"Evaluating from {enhanced_folder} directory")

    scores_headers = [
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

    results_file_name = "scores.csv"
    if config.evaluate.batch_size > 1:
        results_file_name = (
            f"scores_{config.evaluate.batch + 1}-{config.evaluate.batch_size}.csv"
        )

    results_file = ResultsFile(
        file_name=results_file_name,
        header_columns=scores_headers,
    )

    song_listener_pair = make_song_listener_list(
        songs_df["Track Name"].tolist(), listener_dict, config.evaluate.small_test
    )

    song_listener_pair = song_listener_pair[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    ear_model_kwargs = {"signals_same_size": True}
    haaqi_left = HaaqiV1(equalisation=1, ear_model_kwargs=ear_model_kwargs)
    haaqi_right = HaaqiV1(equalisation=1, ear_model_kwargs=ear_model_kwargs)

    for song, listener_id in song_listener_pair:
        split_dir = "train"
        if songs_df[songs_df["Track Name"] == song]["Split"].tolist()[0] == "test":
            split_dir = "test"

        listener = listener_dict[listener_id]
        haaqi_left.set_audiogram(listener.audiogram_left)
        haaqi_right.set_audiogram(listener.audiogram_right)

        combined_score, per_instrument_score = _evaluate_song_listener(
            song=song,
            listener=listener,
            haaqi_left=haaqi_left,
            haaqi_right=haaqi_right,
            config=config,
            split_dir=split_dir,
            enhanced_folder=enhanced_folder,
        )
        results_file.add_result(
            {
                "song": song,
                "listener": listener.id,
                "score": combined_score,
                **per_instrument_score,
            }
        )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    run_calculate_aq()
