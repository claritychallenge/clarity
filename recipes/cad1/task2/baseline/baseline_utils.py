"""Utility functions for the baseline model."""
# pylint: disable=import-error

import json
import logging
import warnings
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def read_mp3(file_path: str, sample_rate=None) -> Tuple[np.ndarray, int]:
    """Read a MP3 file and return its signal.

    Args:
        file_path (str): The path to the mp3 file.
        sample_rate (int): The sampling frequency of the mp3 file.

    Returns:
        signal (np.ndarray): The signal of the mp3 file.
        sample_rate (int): The sampling frequency of the mp3 file.
    """
    if not isinstance(file_path, str):
        raise TypeError("Parameter song_path must be a string")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal, sample_rate = librosa.load(
                file_path,
                sr=sample_rate,
                mono=False,
                res_type="kaiser_fast",
                dtype=np.float32,
            )
    except Exception as error:
        raise ValueError from error

    return signal, sample_rate


def load_listeners_and_scenes(config: DictConfig) -> Tuple[dict, dict]:
    """Load listener and scene data

    Args:
        config (DictConfig): A dictionary-like object containing various configuration
            parameters for the evaluation. This includes the path to the scenes file,
            the path to the listeners train file, and the path to the listeners valid file.

    Returns:
        Tuple[dict, dict]: A tuple containing the scene data and the listener data.

    """
    # Load listener data
    with open(config.path.scenes_file, "r", encoding="utf-8") as fp:
        df_scenes = pd.read_json(fp, orient="index")

    # Load audiograms and scence data for the corresponding split
    if config.evaluate.split == "train":
        with open(config.path.listeners_train_file, "r", encoding="utf-8") as fp:
            listener_audiograms = json.load(fp)
        scenes = df_scenes[df_scenes["split"] == "train"].to_dict("index")

    elif config.evaluate.split == "valid":
        with open(config.path.listeners_valid_file, "r", encoding="utf-8") as fp:
            listener_audiograms = json.load(fp)
        scenes = df_scenes[df_scenes["split"] == "valid"].to_dict("index")

    return scenes, listener_audiograms
