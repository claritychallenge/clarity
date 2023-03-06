""" Run the dummy enhancement. """
# pylint: disable=too-many-locals
# pylint: disable=import-error

import json
import logging
import warnings
from pathlib import Path
from typing import Tuple

import hydra
import librosa
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)


logger = logging.getLogger(__name__)


def read_mp3(file_path: str) -> Tuple[np.ndarray, int]:
    """Read an mp3 file and return its signal.

    Args:
     file_path (str): The path to the mp3 file.

    Returns:
        signal (np.ndarray): The signal of the mp3 file.
        sample_rate (int): The sampling frequency of the mp3 file.
    """
    if not isinstance(file_path, str):
        raise TypeError("Parameter song_path must be a string")

    try:
        signal, sample_rate = librosa.load(
            file_path, sr=None, mono=False, res_type="kaiser_fast", dtype=np.float32
        )
    except Exception as error:
        raise ValueError from error

    return signal, sample_rate


def enhance_song(waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhance a single song for a listener.

    Baseline enhancement returns the original signal.

    Args:
        waveform (np.ndarray): The waveform of the song.
        listener_audiograms (dict): The audiograms of the listener.
        cfg (DictConfig): The configuration file.

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
    enhanced_folder_path = Path(config.path.enhanced_folder)
    enhanced_folder_path.mkdir(parents=True, exist_ok=True)

    # Load songs and listeners
    with open(config.path.listeners_file, "r", encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)

    with open(config.path.scenes_file, "r", encoding="utf-8") as fp:
        scenes = pd.read_json(fp, orient="index")
        # You can load the train and validation scenes separately
        # In the baseline we are processing the validation set directly
        # and not training any system.
        #
        # Load train scenes as:
        # train_scenes = scenes[scenes["split"] == "train"].to_dict("index")
        valid_scenes = scenes[scenes["split"] == "valid"].to_dict("index")

    for _, current_scene in tqdm(valid_scenes.items()):
        song_id = current_scene["song"]
        song_path = (
            Path(config.path.music_dir)
            / f"{current_scene['split']}"
            / f"{song_id:06d}.mp3"
        )
        listener_id = current_scene["listener"]
        listener = listener_audiograms[listener_id]

        # Read song
        song_waveform, sample_rate = read_mp3(song_path.as_posix())
        out_l, out_r = enhance_song(song_waveform)

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
        wavfile.write(enhanced_folder_path / filename, sample_rate, signal_16)


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
