"""Evaluate the enhanced signals using HAAQI and Whisper"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import whisper
from jiwer import compute_measures
from numpy import ndarray
from omegaconf import DictConfig

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.audiogram import Audiogram, Listener
from clarity.utils.file_io import read_signal
from clarity.utils.flac_encoder import read_flac_signal
from clarity.utils.results_support import ResultsFile
from clarity.utils.signal_processing import compute_rms, resample
from recipes.cad2.common.amplification import HearingAid
from recipes.cad2.task1.baseline.enhance import make_scene_listener_list

logger = logging.getLogger(__name__)


def compute_intelligibility(
    enhanced_path: Path,
    segment_metadata: dict,
    config: DictConfig,
) -> float:
    """
    Compute the Intelligibility score for the enhanced signal
    using the Whisper model
    """
    scorer = whisper.load_model(config.evaluate.whisper_version)
    hypotesis = scorer.transcribe(enhanced_path.as_posix())["text"]
    reference = segment_metadata["text"]
    results = compute_measures(reference, hypotesis)
    total_words = results["substitutions"] + results["deletions"] + results["hits"]
    return results["hits"] / total_words


def compute_quality(
    reference_signal: np.ndarray,
    enhanced_signal: np.ndarray,
    listener: Listener,
    config: DictConfig,
) -> tuple[float, float]:
    """Compute the HAAQI score for the left and right channels"""
    scores = []

    for channel in range(2):
        audiogram = (
            listener.audiogram_left if channel == 0 else listener.audiogram_right
        )
        s = compute_haaqi(
            processed_signal=resample(
                enhanced_signal[:, channel],
                config.remix_sample_rate,
                config.HAAQI_sample_rate,
            ),
            reference_signal=resample(
                reference_signal[:, channel],
                config.input_sample_rate,
                config.HAAQI_sample_rate,
            ),
            processed_sample_rate=config.HAAQI_sample_rate,
            reference_sample_rate=config.HAAQI_sample_rate,
            audiogram=audiogram,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_signal[:, channel])),
        )
        scores.append(s)

    return scores[0], scores[1]


@hydra.main(config_path="", config_name="config")
def run_compute_scores(config: DictConfig) -> None:
    """Compute the scores for the enhanced signals"""

    enhanced_folder = Path("enhanced_signals")
    logger.info(f"Evaluating from {enhanced_folder} directory")

    # Load listener audiograms and songs
    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    # Load alphas
    with Path(config.path.alphas_file).open("r", encoding="utf-8") as file:
        alphas = json.load(file)

    # Load scenes
    with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    # Load scene-listeners
    with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    # Load songs
    with Path(config.path.musics_file).open("r", encoding="utf-8") as file:
        songs = json.load(file)

    scores_headers = [
        "scene",
        "song",
        "listener",
        "haaqi_left",
        "haaqi_right",
        "haaqi_avg",
        "whisper",
        "alpha",
        "score",
    ]

    if config.evaluate.batch_size == 1:
        results_file = ResultsFile(
            "scores.csv",
            header_columns=scores_headers,
        )
    else:
        results_file = ResultsFile(
            f"scores_{config.evaluate.batch + 1}-{config.evaluate.batch_size}.csv",
            header_columns=scores_headers,
        )

    # Create the list of scene-listener pairs
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )
    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    # Hearing aid object
    hearing_aid = HearingAid(
        config.ha.compressor,
        config.ha.camfit_gain_table,
    )

    # Loop over the scene-listener pairs
    for idx, scene_listener_ids in enumerate(scene_listener_pairs, 1):
        # Iterate over the scene-listener pairs
        # The reference is the original signal

        scene_id, listener_id = scene_listener_ids

        logger.info(
            f"[{idx:04d}/{len(scene_listener_pairs):04d}] Processing scene-listener"
            f" pair: {scene_id}-{listener_id}"
        )

        # Load scene details
        scene = scenes[scene_id]
        listener = listener_dict[listener_id]
        alpha = alphas[scene["alpha"]]

        # Load the reference signal
        start_sample = int(
            songs[scene["segment_id"]]["start_time"] * config.input_sample_rate
        )
        end_sample = int(
            songs[scene["segment_id"]]["end_time"] * config.input_sample_rate
        )
        reference = read_signal(
            Path(config.path.music_dir)
            / songs[scene["segment_id"]]["path"]
            / "mixture.wav",
            offset=start_sample,
            offset_is_samples=True,
            n_samples=int(end_sample - start_sample),
        )
        # Apply hearing aid
        hearing_aid.set_compressors(listener)
        reference = hearing_aid(reference)

        # Load the enhanced signals
        enhanced_signal_path = (
            enhanced_folder / f"{scene_id}_{listener_id}_A{alpha}_remix.flac"
        )
        enhanced_signal, _ = read_flac_signal(enhanced_signal_path)

        # Compute the HAAQI and Whisper scores
        haaqi_scores = compute_quality(reference, enhanced_signal, listener, config)
        whisper_score = compute_intelligibility(
            enhanced_signal_path,
            songs[scene["segment_id"]],
            config,
        )

        results_file.add_result(
            {
                "scene": scene_id,
                "song": songs[scene["segment_id"]]["track_name"],
                "listener": listener_id,
                "haaqi_left": haaqi_scores[0],
                "haaqi_right": haaqi_scores[1],
                "haaqi_avg": np.mean(haaqi_scores),
                "whisper": whisper_score,
                "alpha": alpha,
                "score": alpha * whisper_score + (1 - alpha) * np.mean(haaqi_scores),
            }
        )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    run_compute_scores()
