""" Run the dummy enhancement. """
# pylint: disable=too-many-locals
# pylint: disable=import-error
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Final

import hydra
import numpy as np
import pyloudnorm as pyln
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from clarity.utils.audiogram import Listener
from recipes.cad1.task2.baseline.baseline_utils import (
    make_scene_listener_list,
    read_mp3,
)
from recipes.cad1.task2.baseline.evaluate import load_listeners_and_scenes

logger = logging.getLogger(__name__)


def compute_average_hearing_loss(listener: Listener) -> float:
    """
    Compute the average hearing loss of a listener.

    Args:
        listener (dict): The audiogram of the listener.

    Returns:
        average_hearing_loss (float): The average hearing loss of the listener.

    """
    CFS: Final = np.array([500, 1000, 2000, 4000])
    left_levels = listener.audiogram_left.resample(CFS).levels
    right_levels = listener.audiogram_right.resample(CFS).levels
    return max(float(np.mean(left_levels)), float(np.mean(right_levels)))


def enhance_song(
    waveform: np.ndarray,
    listener: Listener,
    config: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhance a single song for a listener.

    Baseline enhancement returns the signal with a loudness
    of -14 LUFS if the average hearing loss is below 50 dB HL,
    and -11 LUFS otherwise.

    Args:
        waveform (np.ndarray): The waveform of the song.
        listener_dict (dict): The audiograms of the listener.
        config (dict): Dictionary of configuration options for enhancing music.

    Returns:
        out_left (np.ndarray): The enhanced left channel.
        out_right (np.ndarray): The enhanced right channel.

    """
    meter = pyln.Meter(config.sample_rate)
    original_loudness = meter.integrated_loudness(waveform.T)

    average_hearing_loss = compute_average_hearing_loss(listener)

    extra_reduction = (
        (average_hearing_loss - 50) / 5 if (average_hearing_loss - 50) / 5 > 0 else 0
    )
    target_level = (
        config.enhance.min_level - extra_reduction
        if average_hearing_loss >= 50
        else config.enhance.average_level
    )

    with warnings.catch_warnings(record=True):
        if original_loudness > target_level:
            waveform = pyln.normalize.loudness(
                waveform.T, original_loudness, target_level
            ).T

    out_left = waveform[0, :]
    out_right = waveform[1, :]

    return out_left, out_right


@hydra.main(config_path="", config_name="config")
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The baseline system is a dummy processor that returns the input signal.

    Args:
        config (dict): Dictionary of configuration options for enhancing music.
    """
    enhanced_folder = Path("enhanced_signals") / config.evaluate.split

    # Load scenes and listeners depending on config.evaluate.split
    scenes, listener_dict, scenes_listeners = load_listeners_and_scenes(config)
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )
    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    for scene_id, listener_id in tqdm(scene_listener_pairs):
        current_scene = scenes[scene_id]
        listener = listener_dict[listener_id]

        song_path = Path(config.path.music_dir) / f"{current_scene['song_path']}"

        # Read song
        song_waveform, _ = read_mp3(song_path, config.sample_rate)
        out_l, out_r = enhance_song(
            waveform=song_waveform, listener=listener, config=config
        )

        enhanced = np.stack([out_l, out_r], axis=1)
        filename = f"{scene_id}_{listener.id}_{current_scene['song']}.wav"

        enhanced_folder_listener = enhanced_folder / f"{listener.id}"
        enhanced_folder_listener.mkdir(parents=True, exist_ok=True)

        # Clip and save
        if config.soft_clip:
            enhanced = np.tanh(enhanced)
        n_clipped = np.sum(np.abs(enhanced) > 1.0)
        if n_clipped > 0:
            logger.warning(f"Writing {filename}: {n_clipped} samples clipped")
        np.clip(enhanced, -1.0, 1.0, out=enhanced)
        signal_16 = (32768.0 * enhanced).astype(np.int16)
        wavfile.write(
            enhanced_folder_listener / filename, config.sample_rate, signal_16
        )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
