"""Evaluate the enhanced signals using the HAAQI metric."""
from __future__ import annotations

# pylint: disable=import-error
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

logger = logging.getLogger(__name__)


def apply_ha(
    enhancer: NALR,
    compressor: Compressor | None,
    signal: ndarray,
    audiogram: Audiogram,
    apply_compressor: bool = False,
) -> np.ndarray:
    """
    Apply NAL-R prescription hearing aid to a signal.

    Args:
        enhancer (NALR): A NALR object that enhances the signal.
        compressor (Compressor | None): A Compressor object that compresses the signal.
        signal (ndarray): An ndarray representing the audio signal.
        audiogram (Audiogram): An Audiogram object representing the listener's
            audiogram.
        apply_compressor (bool): Whether to apply the compressor.

    Returns:
        An ndarray representing the processed signal.
    """
    nalr_fir, _ = enhancer.build(audiogram)
    proc_signal = enhancer.apply(nalr_fir, signal)
    if apply_compressor:
        if compressor is None:
            raise ValueError("Compressor must be provided to apply compressor.")

        proc_signal, _, _ = compressor.process(proc_signal)
    return proc_signal


def apply_gains(stems: dict, sample_rate: float, gains: dict) -> dict:
    """Apply gain to the signal by using LUFS.

    Args:
        stems (dict): Dictionary of stems.
        sample_rate (float): Sample rate of the signal.
        gains (dict): Dictionary of gains.

    Returns:
        dict: Dictionary of stems with applied gains.
    """
    meter = pyln.Meter(int(sample_rate))
    stems_gain = {}
    for stem_str, stem_signal in stems.items():
        if stem_signal.shape[0] < stem_signal.shape[1]:
            stem_signal = stem_signal.T

        stem_lufs = meter.integrated_loudness(stem_signal)
        if stem_lufs == -np.inf:
            stem_lufs = -80

        gain = stem_lufs + gains[stem_str]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Possible clipped samples in output"
            )
            stems_gain[stem_str] = pyln.normalize.loudness(stem_signal, stem_lufs, gain)
    return stems_gain


def level_normalisation(
    signal: ndarray, reference_signal: ndarray, sample_rate: float
) -> ndarray:
    """Normalise the signal to the LUFS level of the reference signal.

    Args:
        signal (ndarray): Signal to normalise.
        reference_signal (ndarray): Reference signal.
        sample_rate (float): Sample rate of the signal.

    Returns:
        ndarray: Normalised signal.
    """
    meter = pyln.Meter(int(sample_rate))
    signal_lufs = meter.integrated_loudness(signal)
    reference_signal_lufs = meter.integrated_loudness(reference_signal)

    if signal_lufs == -np.inf:
        signal_lufs = -80

    if reference_signal_lufs == -np.inf:
        reference_signal_lufs = -80

    gain = reference_signal_lufs - signal_lufs

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Possible clipped samples in output")
        normed_signal = pyln.normalize.loudness(signal, signal_lufs, signal_lufs + gain)
    return normed_signal


def remix_stems(stems: dict, reference_signal, sample_rate: float) -> ndarray:
    """Remix the stems into a stereo signal.

    The remixing is done by summing the stems.
    Then, the signal is normalised to the LUFS level of the reference signal.

    Args:
        stems (dict): Dictionary of stems.
        reference_signal (ndarray): Reference signal.
        sample_rate (float): Sample rate of the signal.

    Returns:
        ndarray: Stereo signal.
    """
    remix_signal = np.zeros(stems["vocals"].shape)
    for _, stem_signal in stems.items():
        remix_signal += stem_signal
    return level_normalisation(remix_signal, reference_signal, sample_rate)


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

    # Can define a standard 'small_test' with just 1/15 of the data
    if small_test:
        scene_listener_pairs = scene_listener_pairs[::15]

    return scene_listener_pairs


def set_song_seed(song: str) -> None:
    """Set a seed that is unique for the given song"""
    song_encoded = hashlib.md5(song.encode("utf-8")).hexdigest()
    song_md5 = int(song_encoded, 16) % (10**8)
    np.random.seed(song_md5)


def load_reference_stems(music_dir: str | Path) -> tuple[dict[str, ndarray], ndarray]:
    """Load the reference stems for a given scene.

    Args:
        scene (dict): The scene to load the stems for.
        music_dir (str | Path): The path to the music directory.
    Returns:
        reference_stems (dict): A dictionary of reference stems.
        original_mixture (ndarray): The original mixture.
    """
    reference_stems = {}
    for instrument in ["drums", "bass", "other", "vocals"]:
        stem = read_signal(Path(music_dir) / f"{instrument}.wav")
        reference_stems[instrument] = stem

    return reference_stems, read_signal(Path(music_dir) / "mixture.wav")


@hydra.main(config_path="", config_name="config")
def run_calculate_aq(config: DictConfig) -> None:
    """Evaluate the enhanced signals using the HAAQI metric."""

    enhanced_folder = Path("enhanced_signals")
    logger.info(f"Evaluating from {enhanced_folder} directory")

    # Load listener audiograms and songs
    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    with Path(config.path.gains_file).open("r", encoding="utf-8") as file:
        gains = json.load(file)

    with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    with Path(config.path.music_file).open("r", encoding="utf-8") as file:
        songs = json.load(file)

    enhancer = NALR(**config.nalr)

    scores_headers = [
        "scene",
        "song",
        "listener",
        "left_score",
        "right_score",
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
    # results_file.write_header()

    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )
    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]
    num_scenes = len(scene_listener_pairs)
    for idx, scene_listener_pair in enumerate(scene_listener_pairs, 1):
        scene_id, listener_id = scene_listener_pair

        scene = scenes[scene_id]
        song_name = f"{scene['music']}-{scene['head_loudspeaker_positions']}"

        logger.info(
            f"[{idx:03d}/{num_scenes:03d}] "
            f"Evaluating {scene_id} for listener {listener_id}"
        )

        # Load reference signals
        reference_stems, original_mixture = load_reference_stems(
            Path(config.path.music_dir) / songs[song_name]["Path"]
        )
        reference_stems = apply_gains(
            reference_stems, config.sample_rate, gains[scene["gain"]]
        )
        reference_mixture = remix_stems(
            reference_stems, original_mixture, config.sample_rate
        )

        # Set the random seed for the scene
        if config.evaluate.set_random_seed:
            set_song_seed(scene_id)

        # Evaluate listener
        listener = listener_dict[listener_id]

        # Load enhanced signal
        enhanced_signal, _ = read_flac_signal(
            Path(enhanced_folder) / f"{scene_id}_{listener.id}_remix.flac"
        )

        # Apply hearing aid to reference signals
        left_reference = apply_ha(
            enhancer=enhancer,
            compressor=None,
            signal=reference_mixture[:, 0],
            audiogram=listener.audiogram_left,
            apply_compressor=False,
        )
        right_reference = apply_ha(
            enhancer=enhancer,
            compressor=None,
            signal=reference_mixture[:, 1],
            audiogram=listener.audiogram_right,
            apply_compressor=False,
        )

        # Compute the scores
        left_score = compute_haaqi(
            processed_signal=resample(
                enhanced_signal[:, 0],
                config.remix_sample_rate,
                config.HAAQI_sample_rate,
            ),
            reference_signal=resample(
                left_reference, config.sample_rate, config.HAAQI_sample_rate
            ),
            processed_sample_rate=config.HAAQI_sample_rate,
            reference_sample_rate=config.HAAQI_sample_rate,
            audiogram=listener.audiogram_left,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 0])),
        )

        right_score = compute_haaqi(
            processed_signal=resample(
                enhanced_signal[:, 1],
                config.remix_sample_rate,
                config.HAAQI_sample_rate,
            ),
            reference_signal=resample(
                right_reference, config.sample_rate, config.HAAQI_sample_rate
            ),
            processed_sample_rate=config.HAAQI_sample_rate,
            reference_sample_rate=config.HAAQI_sample_rate,
            audiogram=listener.audiogram_right,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 1])),
        )

        # Save scores
        results_file.add_result(
            {
                "scene": scene_id,
                "song": song_name,
                "listener": listener.id,
                "left_score": left_score,
                "right_score": right_score,
                "score": float(np.mean([left_score, right_score])),
            }
        )

    logger.info("Done!")


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    run_calculate_aq()
