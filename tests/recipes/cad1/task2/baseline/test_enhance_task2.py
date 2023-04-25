"""Test the enhance module."""
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
from omegaconf import DictConfig

from clarity.utils.audiogram import Audiogram, Listener
from recipes.cad1.task2.baseline.enhance import enhance_song

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources" / "recipes" / "cad1" / "task2"


def test_enhance_song():
    """Test the enhance_song function."""
    np.random.seed(42)

    # Set the sample rate and gain
    duration = 0.5

    config = DictConfig(
        {
            "sample_rate": 16000,
            "enhance": {"min_level": -11, "max_level": -19, "average_level": -14},
        }
    )

    levels = np.array([20, 30, 35, 45, 50, 60, 65, 60])
    frequencies = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])
    audiogram = Audiogram(levels=levels, frequencies=frequencies)

    listener = Listener(audiogram, audiogram)
    # Create a sample waveform
    waveform = np.random.rand(2, int(config.sample_rate * duration))

    # Call the function
    out_left, out_right = enhance_song(waveform, listener, config)

    expected_left = np.load(
        RESOURCES / "test_enhance.enhance_song_left.npy", allow_pickle=True
    )
    expected_right = np.load(
        RESOURCES / "test_enhance.enhance_song_right.npy", allow_pickle=True
    )

    # Check that the output is not equal to the input
    np.testing.assert_array_almost_equal(out_left, expected_left)
    np.testing.assert_array_almost_equal(out_right, expected_right)

    # Check that the output has the correct loudness
    meter = pyln.Meter(config.sample_rate)

    out_loudness = meter.integrated_loudness(np.array([out_left, out_right]).T)
    assert np.isclose(out_loudness, -14, atol=0.1)


@pytest.mark.skip(reason="Not implemented yet")
def test_enhance():
    """Test enhance function."""
