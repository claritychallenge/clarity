"""Test for baseline_utils.py"""
# pylint: disable=import-error

from pathlib import Path

import librosa
import numpy as np
from omegaconf import OmegaConf

from recipes.cad1.task2.baseline.baseline_utils import (
    load_listeners_and_scenes,
    read_mp3,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources" / "recipes" / "cad1" / "task2"


def test_read_mp3():
    """Test read_mp3()"""
    signal, sample_rate = read_mp3(librosa.example("brahms"))
    assert isinstance(signal, np.ndarray)
    assert isinstance(sample_rate, int)


def test_load_listeners_and_scenes():
    """Test load_listeners_and_scenes()"""
    config = OmegaConf.create(
        {
            "path": {
                "scenes_file": (RESOURCES / "scenes.json").as_posix(),
                "listeners_train_file": (RESOURCES / "listeners.json").as_posix(),
            },
            "evaluate": {"split": "train"},
        }
    )
    scenes, listener_audiograms = load_listeners_and_scenes(config)
    assert isinstance(scenes, dict)
    assert isinstance(listener_audiograms, dict)
