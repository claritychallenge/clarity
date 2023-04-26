"""Utility functions for the baseline model."""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from clarity.utils.audiogram import Listener

# pylint: disable=import-error


logger = logging.getLogger(__name__)


def read_mp3(
    file_path: str | Path, sample_rate: float | None = None
) -> tuple[np.ndarray, float]:
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
            signal, returned_sample_rate = librosa.load(
                str(file_path),
                sr=sample_rate,
                mono=False,
                res_type="kaiser_best",
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

    return signal, returned_sample_rate


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


def load_listeners_and_scenes(
    config: DictConfig,
) -> tuple[dict, dict[str, Listener], dict]:
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

    # Load audiograms and scene data for the corresponding split
    if config.evaluate.split == "train":
        listeners = Listener.load_listener_dict(config.path.listeners_train_file)
        scenes = df_scenes[df_scenes["split"] == "train"].to_dict("index")
    elif config.evaluate.split == "valid":
        listeners = Listener.load_listener_dict(config.path.listeners_valid_file)
        scenes = df_scenes[df_scenes["split"] == "valid"].to_dict("index")
    else:
        raise ValueError(f"Unknown split {config.evaluate.split}")

    with open(config.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
        scenes_listeners = {
            k: v for k, v in scenes_listeners.items() if k in scenes.keys()
        }

    return scenes, listeners, scenes_listeners


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
