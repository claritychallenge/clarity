"""Evaluate the enhanced signals using HAAQI and Whisper"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pyloudnorm as pyln
import torch.nn
import whisper
from alt_eval import compute_metrics
from omegaconf import DictConfig

from clarity.enhancer.multiband_compressor import MultibandCompressor
from clarity.evaluator.haaqi import compute_haaqi
from clarity.evaluator.msbg.msbg import Ear
from clarity.utils.audiogram import Listener
from clarity.utils.flac_encoder import read_flac_signal, save_flac_signal
from clarity.utils.results_support import ResultsFile
from clarity.utils.signal_processing import compute_rms, resample

logger = logging.getLogger(__name__)


def make_scene_listener_list(scenes_listeners: dict, small_test: bool = False) -> list:
    """Make the list of scene-listener pairing to process

    Args:
        scenes_listeners (dict): Dictionary of scenes and listeners.
        small_test (bool): Whether to use a small test set.

    Returns:
        list: List of scene-listener pairings.

    """
    scene_listener_pairs = [
        (scene, listener)
        for scene in scenes_listeners
        for listener in scenes_listeners[scene]
    ]

    # Can define a standard 'small_test' with just 1/50 of the data
    if small_test:
        scene_listener_pairs = scene_listener_pairs[::400]

    return scene_listener_pairs


def compute_intelligibility(
    enhanced_signal: np.ndarray,
    segment_metadata: dict,
    scorer: torch.nn.Module,
    listener: Listener,
    sample_rate: int,
    save_intermediate: bool = False,
    path_intermediate: str | Path | None = None,
    equiv_0db_spl: float = 100,
) -> tuple[float, float]:
    """
    Compute the Intelligibility score for the enhanced signal
    using the Whisper model.

    To the enhanced signal, we apply the MSGB hearing loss model
    before transcribing with Whisper.

    Args:
        enhanced_signal: The enhanced signal
        segment_metadata: The metadata of the segment
        scorer: The Whisper model
        listener: The listener
        sample_rate: The sample rate of the signal
        save_intermediate: Save the intermediate signal
        path_intermediate: The path to save the intermediate signal
        equiv_0db_spl: The equivalent 0 dB SPL

    Returns:
        The intelligibility score for the left and right channels
    """
    if path_intermediate is None:
        path_intermediate = Path.cwd()
    if isinstance(path_intermediate, str):
        path_intermediate = Path(path_intermediate)

    ear = Ear(
        equiv_0db_spl=equiv_0db_spl,
        sample_rate=sample_rate,
        verbose=False,
    )

    reference = segment_metadata["text"]

    # Compute left ear
    ear.set_audiogram(listener.audiogram_left)
    enhanced_left = ear.process(enhanced_signal[:, 0])[0]
    left_path = Path(f"{path_intermediate.as_posix()}_left.flac")
    save_flac_signal(
        enhanced_signal,
        left_path,
        44100,
        sample_rate,
    )
    hypothesis = scorer.transcribe(left_path, fp16=False)["text"]
    left_results = compute_metrics(
        [reference], [hypothesis], languages="en", include_other=False
    )

    # Compute right ear
    ear.set_audiogram(listener.audiogram_right)
    enhanced_right = ear.process(enhanced_signal[:, 1])[0]
    right_path = Path(f"{path_intermediate.as_posix()}_right.flac")
    save_flac_signal(
        enhanced_signal,
        right_path,
        44100,
        sample_rate,
    )
    hypothesis = scorer.transcribe(right_path, fp16=False)["text"]
    right_results = compute_metrics(
        [reference], [hypothesis], languages="en", include_other=False
    )

    # Compute the average score for both ears
    total_words = (
        right_results["substitutions"]
        + right_results["deletions"]
        + right_results["hits"]
    )

    if save_intermediate:
        enhanced_signal = np.stack([enhanced_left, enhanced_right], axis=1)
        save_flac_signal(
            enhanced_signal,
            path_intermediate,
            44100,
            sample_rate,
        )

    Path(left_path).unlink()
    Path(right_path).unlink()

    return left_results["hits"] / total_words, right_results["hits"] / total_words


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


def load_reference_signal(
    path: str | Path,
    start_sample: int | None,
    end_sample: int | None,
    level_luft: float = -40.0,
) -> np.ndarray:
    """Load the reference signal"""

    if isinstance(path, str):
        path = Path(path)

    if start_sample is None:
        start_sample = 0
    if end_sample is None:
        end_sample = -1

    vocal, _ = read_flac_signal(path / "vocals.flac")
    accompaniment = np.zeros_like(vocal)

    for instrument in ["bass", "drums", "other"]:
        instrument_signal, sample_rate = read_flac_signal(path / f"{instrument}.flac")
        accompaniment += instrument_signal

    mixture = vocal * 10 ** (1 / 20) + accompaniment * 10 ** (-1 / 20)
    mixture = normalise_luft(mixture, sample_rate, level_luft)
    return mixture[start_sample:end_sample, :]


def normalise_luft(
    signal: np.ndarray, sample_rate: float, target_luft=-40
) -> np.ndarray:
    """
    Normalise the signal to a target loudness level.
    Args:
        signal: input signal to normalise
        sample_rate: sample rate of the signal
        target_luft: target loudness level in LUFS

    Returns:
        np.ndarray: normalised signal
    """
    level_meter = pyln.Meter(int(sample_rate))
    input_level = level_meter.integrated_loudness(signal)
    return signal * (10 ** ((target_luft - input_level) / 20))


@hydra.main(config_path="", config_name="config", version_base=None)
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

    # Load compressor params
    with Path(config.path.enhancer_params_file).open("r", encoding="utf-8") as file:
        enhancer_params = json.load(file)

    scores_headers = [
        "scene",
        "song",
        "listener",
        "haaqi_left",
        "haaqi_right",
        "haaqi_avg",
        "whisper_left",
        "whisper_rigth",
        "whisper_be",
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

    # create hearing aid
    enhancer = MultibandCompressor(
        crossover_frequencies=config.enhancer.crossover_frequencies,
        sample_rate=config.input_sample_rate,
    )

    intelligibility_scorer = whisper.load_model(config.evaluate.whisper_version)

    # Loop over the scene-listener pairs
    for idx, scene_listener_ids in enumerate(scene_listener_pairs, 1):
        # Iterate over the scene-listener pairs
        # The reference is the original signal
        logger.info(
            f"[{idx:04d}/{len(scene_listener_pairs):04d}] Processing scene-listener"
            f" pair: {scene_listener_ids}"
        )

        scene_id, listener_id = scene_listener_ids

        # Load scene details
        scene = scenes[scene_id]
        listener = listener_dict[listener_id]
        alpha = alphas[scene["alpha"]]

        #############################################################
        # REFERENCE SIGNAL

        # Load the reference signal
        start_sample = int(
            songs[scene["segment_id"]]["start_time"] * config.input_sample_rate
        )
        end_sample = int(
            songs[scene["segment_id"]]["end_time"] * config.input_sample_rate
        )
        reference = load_reference_signal(
            Path(config.path.music_dir) / songs[scene["segment_id"]]["path"],
            start_sample,
            end_sample,
        )

        # Get the listener's compressor params
        mbc_params_listener: dict[str, dict] = {"left": {}, "right": {}}

        for ear in ["left", "right"]:
            mbc_params_listener[ear]["release"] = config.enhancer.release
            mbc_params_listener[ear]["attack"] = config.enhancer.attack
            mbc_params_listener[ear]["threshold"] = config.enhancer.threshold
        mbc_params_listener["left"]["ratio"] = enhancer_params[listener_id]["cr_l"]
        mbc_params_listener["right"]["ratio"] = enhancer_params[listener_id]["cr_r"]
        mbc_params_listener["left"]["makeup_gain"] = enhancer_params[listener_id][
            "gain_l"
        ]
        mbc_params_listener["right"]["makeup_gain"] = enhancer_params[listener_id][
            "gain_r"
        ]

        # Apply compressor to reference signal
        enhancer.set_compressors(**mbc_params_listener["left"])
        left_reference = enhancer(signal=reference[:, 0])

        enhancer.set_compressors(**mbc_params_listener["right"])
        right_reference = enhancer(signal=reference[:, 1])

        # Reference signal amplified
        reference = np.stack([left_reference[0], right_reference[0]], axis=1)

        #############################################################
        # ENHANCED SIGNAL

        # Load the enhanced signals
        enhanced_signal_path = (
            enhanced_folder / f"{scene_id}_{listener_id}_A{alpha}_remix.flac"
        )
        enhanced_signal, _ = read_flac_signal(enhanced_signal_path)

        #############################################################
        # COMPUTE SCORES

        # Compute the HAAQI and Whisper scores
        haaqi_scores = compute_quality(reference, enhanced_signal, listener, config)
        whisper_scores = compute_intelligibility(
            enhanced_signal=enhanced_signal,
            segment_metadata=songs[scene["segment_id"]],
            scorer=intelligibility_scorer,
            listener=listener,
            sample_rate=config.remix_sample_rate,
            save_intermediate=config.evaluate.save_intermediate,
            path_intermediate=enhanced_signal_path.parent
            / f"{scene_id}_{listener_id}_A{alpha}_remix_hl.flac",
            equiv_0db_spl=config.evaluate.equiv_0db_spl,
        )

        results_file.add_result(
            {
                "scene": scene_id,
                "song": songs[scene["segment_id"]]["track_name"],
                "listener": listener_id,
                "haaqi_left": haaqi_scores[0],
                "haaqi_right": haaqi_scores[1],
                "haaqi_avg": np.mean(haaqi_scores),
                "whisper_left": whisper_scores[0],
                "whisper_rigth": whisper_scores[1],
                "whisper_be": np.max(whisper_scores),
                "alpha": alpha,
                "score": alpha * np.max(whisper_scores) + (1 - alpha) * np.mean(
                    haaqi_scores
                ),
            }
        )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    run_compute_scores()

    logger.info("Evaluation completed")
