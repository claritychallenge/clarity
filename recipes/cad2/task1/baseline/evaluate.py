"""Evaluate the enhanced signals using HAAQI and Whisper"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path

import hydra
import numpy as np
import pyloudnorm as pyln
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
        "whisper_left",
        "whisper_right",
        "score_left",
        "score_right",
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

    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )
    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    # create hearing aid
    hearing_aid = HearingAid(
        config.ha.compressor,
        config.ha.camfit_gain_table,
    )

    for idx, scene_listener_ids in enumerate(scene_listener_pairs, 1):
        logger.info(
            f"[{idx:04d}/{len(scene_listener_pairs):04d}] Processing scene-listener"
            f" pair: {scene_listener_ids}"
        )

        scene_id, listener_id = scene_listener_ids
        scene = scenes[scene_id]
        listener = listener_dict[listener_id]
        alpha = alphas[scene["alpha"]]

        reference = read_signal(
            Path(config.path.music_dir)
            / songs[scene["segment_id"]]["path"]
            / "mixture.wav",
            offset=int(
                songs[scene["segment_id"]]["start_time"] * config.input_sample_rate
            ),
            offset_is_samples=True,
            n_samples=int(
                (
                    songs[scene["segment_id"]]["end_time"]
                    - songs[scene["segment_id"]]["start_time"]
                )
                * config.input_sample_rate
            ),
        )

        # Load the enhanced signals
        enhanced_signal, _ = read_flac_signal(
            enhanced_folder / f"{scene_id}_{listener_id}_A{alpha}_remix.flac"
        )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    run_compute_scores()
