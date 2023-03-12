"""Utility functions for the baseline model."""
# pylint: disable=import-error

import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def read_mp3(
    file_path: Union[str, Path], sample_rate: Optional[int] = None
) -> Tuple[np.ndarray, Optional[int]]:
    """Read a MP3 file and return its signal.

    Args:
        file_path (str, Path): The path to the mp3 file.
        sample_rate (int): The sampling frequency of the mp3 file.

    Returns:
        signal (np.ndarray): The signal of the mp3 file.
        sample_rate (int): The sampling frequency of the mp3 file.
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal, sample_rate = librosa.load(
                str(file_path),
                sr=sample_rate,
                mono=False,
                res_type="kaiser_fast",
                dtype=np.float32,
            )
    except Exception as error:
        raise ValueError from error

    return signal, sample_rate


def load_listeners_and_scenes(config: DictConfig) -> Tuple[dict, dict, dict]:
    """Load listener and scene data

    Args:
        config (DictConfig): A dictionary-like object containing various configuration
            parameters for the evaluation. This includes the path to the scenes file,
            the path to the listeners train file, and the path to the listeners valid file.

    Returns:
        Tuple[dict, dict, dict]: A tuple containing the scene data, the listener data and
            the pair scenes-listeners.

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

    with open(config.path.scenes_listeners_file, "r", encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
        scenes_listeners = {
            k: v for k, v in scenes_listeners.items() if k in scenes.keys()
        }

    return scenes, listener_audiograms, scenes_listeners


def make_scene_listener_list(scenes_listeners, small_test=False):
    """Make the list of scene-listener pairing to process"""
    scene_listener_pairs = [
        (scene, listener)
        for scene in scenes_listeners
        for listener in scenes_listeners[scene]
    ]

    # Can define a standard 'small_test' with just 1/15 of the data
    if small_test:
        scene_listener_pairs = scene_listener_pairs[::15]

    return scene_listener_pairs