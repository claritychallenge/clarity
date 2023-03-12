""" Run the dummy enhancement. """
# pylint: disable=too-many-locals
# pylint: disable=import-error

import logging
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


def enhance_song(
    waveform: np.ndarray, sample_rate: int, gain_db: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhance a single song for a listener.

    Baseline enhancement returns the original signal with a gain in dB LUFS.

    Args:
        waveform (np.ndarray): The waveform of the song.
        sample_rate (int): The sample rate of the song.
        gain_db (float): The gain to apply to the song in dB LUFS. Defaults to 0.0.

    Returns:
        out_left (np.ndarray): The enhanced left channel.
        out_right (np.ndarray): The enhanced right channel.


    Note:

    In your enhancement you may need access to listener_audiograms and config file.
    If that's the case, you can modify the function signature to include them.
    E.g.

    >> def enhance_song(
          waveform: np.ndarray,
          listener_audiograms,
          dict, cfg: DictConfig
       ) -> Tuple[np.ndarray, np.ndarray]:

    Then, left and right audiograms can be accessed as follows:

    >> left_audiogram = listener_audiograms["audiogram_levels_l"]
    >> right_audiogram = listener_audiograms["audiogram_levels_r"]

    Remember to add them to the function call in the main function.

    >> out_l, out_r = enhance_song(song_waveform, listener_audiograms, config)

    """

    if waveform.ndim == 1:
        waveform = np.array([waveform, waveform])

    meter = pyln.Meter(sample_rate)
    original_loudness = meter.integrated_loudness(waveform.T)
    waveform = pyln.normalize.loudness(
        waveform.T, original_loudness, original_loudness + gain_db
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

        song_id = current_scene["song"]
        song_path = Path(config.path.music_dir) / f"{current_scene['song_path']}"

        # Read song
        song_waveform, _ = read_mp3(song_path, config.sample_rate)
        out_l, out_r = enhance_song(song_waveform, config.sample_rate, gain_db=-5)

        enhanced = np.stack([out_l, out_r], axis=1)
        filename = f"{listener['name']}_{song_id}.wav"

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
