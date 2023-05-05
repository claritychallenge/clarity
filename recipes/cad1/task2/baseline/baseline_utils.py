"""Utility functions for the baseline model."""
from __future__ import annotations

# pylint: disable=import-error
import json
import logging
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def read_mp3(
    file_path: str | Path, sample_rate: int | None = None
) -> tuple[np.ndarray, float | None]:
    """Read a MP3 file and return its signal.

    Args:
        file_path (str, Path): The path to the mp3 file.
        sample_rate (int | None): The sampling frequency of the mp3 file.

    Returns:
        signal (np.ndarray): The signal of the mp3 file.
        sample_rate (float): The sampling frequency of the mp3 file.
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal, sample_rate = librosa.load(
                str(file_path),
                sr=sample_rate,
                mono=False,
                res_type="soxr_hq",
                dtype=np.float32,
            )
    except Exception as error:
        raise ValueError from error

    if signal.ndim == 1:
        # If mono, duplicate to stereo
        signal = np.stack([signal, signal], axis=0)

    # Peak Normalization for cases when signal has
    # absolute values greater than 1
    if np.max(np.abs(signal)) > 1:
        signal = signal / np.max(np.abs(signal))

    return signal, sample_rate


def load_hrtf(config: DictConfig) -> dict:
    """Load the HRTF file.

    Args:
        config (DictConfig): A dictionary-like object containing various configuration
            parameters for the evaluation. This includes the path to the HRTF files.

    Returns:
        hrtf_data (dict): A dictionary containing the HRTF data for the dataset.

    """
    with open(config.path.hrtf_file, encoding="utf-8") as fp:
        hrtf_data = json.load(fp)
    return hrtf_data[config.evaluate.split]


def load_listeners_and_scenes(config: DictConfig) -> tuple[dict, dict, dict]:
    """Load listener and scene data

    Args:
        config (DictConfig): A dictionary-like object containing various configuration
            parameters for the evaluation. This includes the path to the scenes file,
            the path to the listeners train file, and the path to the listeners valid
            file.

    Returns:
        Tuple[dict, dict, dict]: A tuple containing the scene data, the listener data
            and the pair scenes-listeners.

    """
    # Load listener data
    with open(config.path.scenes_file, encoding="utf-8") as fp:
        df_scenes = pd.read_json(fp, orient="index")

    # Load audiograms and scence data for the corresponding split
    if config.evaluate.split == "train":
        with open(config.path.listeners_train_file, encoding="utf-8") as fp:
            listener_audiograms = json.load(fp)
        scenes = df_scenes[df_scenes["split"] == "train"].to_dict("index")

    elif config.evaluate.split == "valid":
        with open(config.path.listeners_valid_file, encoding="utf-8") as fp:
            listener_audiograms = json.load(fp)
        scenes = df_scenes[df_scenes["split"] == "valid"].to_dict("index")

    with open(config.path.scenes_listeners_file, encoding="utf-8") as fp:
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
