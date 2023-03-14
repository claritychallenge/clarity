"""Test the enhance module."""
# pylint: disable=import-error

from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from recipes.cad1.task2.baseline.enhance import enhance_song

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources" / "recipes" / "cad1" / "task2"


def test_enhance_song():
    """Test the enhance_song function."""
    np.random.seed(42)

    # Set the sample rate and gain
    sample_rate = 16000
    duration = 1.0
    gain_db = 3.0

    # Create a sample waveform
    waveform = np.random.rand(2, int(sample_rate * duration))

    # Call the function
    out_left, out_right = enhance_song(waveform, sample_rate, gain_db)

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
    meter = pyln.Meter(sample_rate)
    expected_loudness = meter.integrated_loudness(waveform.T) + gain_db
    out_loudness = meter.integrated_loudness(np.array([out_left, out_right]).T)
    assert np.isclose(out_loudness, expected_loudness, atol=0.1)
