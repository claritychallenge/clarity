""" Run the dummy enhancement. """
# pylint: disable=too-many-locals
# pylint: disable=import-error

import logging
import warnings
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pyloudnorm as pyln
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from recipes.cad1.task2.baseline.baseline_utils import (
    make_scene_listener_list,
    read_mp3,
)
from recipes.cad1.task2.baseline.evaluate import load_listeners_and_scenes

logger = logging.getLogger(__name__)


def compute_average_hearing_loss(listener: dict) -> float:
    """
    Compute the average hearing loss of a listener.

    Args:
        listener (dict): The audiogram of the listener.

    Returns:
        average_hearing_loss (float): The average hearing loss of the listener.

    """
    cfs = [500, 1000, 2000, 4000]
    left_loss = [
        listener["audiogram_levels_l"][i]
        for i in range(len(listener["audiogram_cfs"]))
        if listener["audiogram_cfs"][i] in cfs
    ]
    right_loss = [
        listener["audiogram_levels_l"][i]
        for i in range(len(listener["audiogram_cfs"]))
        if listener["audiogram_cfs"][i] in cfs
    ]
    return (np.mean(left_loss) + np.mean(right_loss)) / 2


def enhance_song(
    waveform: np.ndarray,
    listener_audiograms: dict,
    config: DictConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Enhance a single song for a listener.

    Baseline enhancement returns the signal with a loudness
    of -14 LUFS if the average hearing loss is below 50 dB HL,
    and -11 LUFS otherwise.

    Args:
        waveform (np.ndarray): The waveform of the song.
        listener_audiograms (dict): The audiograms of the listener.
        config (dict): Dictionary of configuration options for enhancing music.

    Returns:
        out_left (np.ndarray): The enhanced left channel.
        out_right (np.ndarray): The enhanced right channel.
        output_level (float): The output loudness level.
    """

    if waveform.ndim == 1:
        waveform = np.array([waveform, waveform])

    meter = pyln.Meter(config.sample_rate)
    original_loudness = meter.integrated_loudness(waveform.T)

    average_hearing_loss = compute_average_hearing_loss(listener_audiograms)
    target_level = (
        config.enhance.min_level
        if average_hearing_loss > 50
        else config.enhance.average_level
    )

    target_level = min(target_level, config.enhance.max_level)

    with warnings.catch_warnings(record=True):
        if target_level < original_loudness:
            waveform = pyln.normalize.loudness(
                waveform.T, original_loudness, target_level
            ).T

    out_left = waveform[0, :]
    out_right = waveform[1, :]

    return (
        out_left,
        out_right,
        target_level if target_level < original_loudness else original_loudness,
    )


@hydra.main(config_path="", config_name="config")
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The baseline system is a dummy processor that returns the input signal.

    Args:
        config (dict): Dictionary of configuration options for enhancing music.
    """
    enhanced_folder = Path("enhanced_signals") / config.evaluate.split
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    # Load scenes and listeners depending on config.evaluate.split
    scenes, listener_audiograms, scenes_listeners = load_listeners_and_scenes(config)
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )
    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    for scene_id, listener_id in tqdm(scene_listener_pairs):
        current_scene = scenes[scene_id]
        listener = listener_audiograms[listener_id]

        song_path = Path(config.path.music_dir) / f"{current_scene['song_path']}"

        # Read song
        song_waveform, _ = read_mp3(song_path, config.sample_rate)
        out_l, out_r, out_level = enhance_song(
            waveform=song_waveform, listener_audiograms=listener, config=config
        )
        logger.info(
            f"Enhanced {current_scene['song']} for {listener['name']}: "
            f"output loudness {out_level:.2f} LUFS"
        )

        enhanced = np.stack([out_l, out_r], axis=1)
        filename = f"{listener['name']}_{current_scene['song']}.wav"

        # Clip and save
        if config.soft_clip:
            enhanced = np.tanh(enhanced)
        n_clipped = np.sum(np.abs(enhanced) > 1.0)
        if n_clipped > 0:
            logger.warning(f"Writing {filename}: {n_clipped} samples clipped")
        np.clip(enhanced, -1.0, 1.0, out=enhanced)
        signal_16 = (32768.0 * enhanced).astype(np.int16)
        wavfile.write(enhanced_folder / filename, config.sample_rate, signal_16)


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
