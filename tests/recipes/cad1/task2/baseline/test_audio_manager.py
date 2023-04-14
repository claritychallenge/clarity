"""Test for AudioManager module"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from recipes.cad1.task2.baseline.audio_manager import AudioManager


def test_save_audios(tmp_path):
    """Test save_audios method."""
    np.random.seed(42)
    # Initialize an audio manager with temporary directory as output audio path
    audio_manager = AudioManager(output_audio_path=tmp_path.as_posix())

    # Create sample audio data
    audio_data = np.random.randn(2, 44100)

    # Add audio data to audio manager
    audio_manager.add_audios_to_save("test_audio", audio_data)

    # Save audio
    audio_manager.save_audios()

    # Check if audio file was saved
    audio_file = Path(tmp_path) / "test_audio.wav"
    assert audio_file.is_file()

    # Check if audio data was saved correctly
    sample_rate, _ = wavfile.read(audio_file)
    assert sample_rate == pytest.approx(
        audio_manager.sample_rate, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.skip(reason="Not implemented yet")
def test_clip_audio():
    """Test clip_audio method."""


@pytest.mark.skip(reason="Not implemented yet")
def test_get_lufs_level():
    """Test get_lufs_level method."""


@pytest.mark.skip(reason="Not implemented yet")
def test_scale_to_lufs():
    """Test scale_to_lufs method."""
