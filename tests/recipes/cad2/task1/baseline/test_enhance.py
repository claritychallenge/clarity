"""Test the enhance function."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from scipy.io import wavfile

from recipes.cad2.task1.baseline.enhance import (
    downmix_signal,
    enhance,
    get_device,
    load_separation_model,
    make_scene_listener_list,
    save_flac_signal,
    separate_sources,
)


@pytest.fixture
def signal():
    """Return a random signal of length 8000."""
    np.random.seed(0)
    return np.random.rand(8000)


@pytest.fixture
def filename(tmp_path):
    """Return a temporary file path."""
    return tmp_path / "test_signal.flac"


@pytest.fixture
def mock_model():
    """Return a mock model."""

    class MockModel(torch.nn.Module):
        def forward(self, x):
            return x * 0.5

    return MockModel()


def test_save_flac_signal(signal, filename):
    """Test saving a signal to a FLAC file."""
    save_flac_signal(
        signal=signal,
        filename=filename,
        signal_sample_rate=8000,
        output_sample_rate=4000,
        do_clip_signal=True,
        do_soft_clip=False,
        do_scale_signal=False,
    )
    assert filename.exists()


def test_separate_sources(mock_model, signal):
    """Test separating sources."""
    mix = torch.tensor(signal).unsqueeze(0).unsqueeze(0)
    separated = separate_sources(
        mock_model, mix, sample_rate=8000, segment=1.0, number_sources=1
    )
    assert separated.squeeze(0).shape == mix.shape
    assert np.sum(separated.numpy()) == pytest.approx(
        1884.4956, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_get_device():
    """Test getting the device."""
    device, device_type = get_device(None)
    assert device_type in ["cpu", "cuda"]


def test_load_separation_model():
    """Test loading a separation model."""
    with patch(
        "recipes.cad2.task1.ConvTasNet.local.tasnet.ConvTasNetStereo.from_pretrained",
        return_value=MagicMock(),
    ) as mock_method:
        model = load_separation_model("causal", torch.device("cpu"))
        mock_method.assert_called_once()


def test_make_scene_listener_list():
    """Test making scene-listener pairs."""
    scenes_listeners = {"scene1": ["listener1", "listener2"], "scene2": ["listener3"]}
    pairs = make_scene_listener_list(scenes_listeners)
    assert len(pairs) == 3


def test_downmix_signal():
    """Test downmixing signals."""
    vocals = np.random.rand(2, 500)
    accompaniment = np.random.rand(2, 500)
    beta = 0.5
    downmixed = downmix_signal(vocals, accompaniment, beta)
    assert downmixed.shape == vocals.shape


# Integration test for enhance function
@patch("recipes.cad2.task1.baseline.enhance.load_separation_model")
@patch("clarity.utils.audiogram.Listener.load_listener_dict")
@patch("clarity.utils.file_io.read_signal")
@patch("recipes.cad2.task1.baseline.enhance.save_flac_signal")
def test_enhance(
    mock_save_flac_signal,
    mock_read_signal,
    mock_load_listener_dict,
    mock_load_separation_model,
    tmp_path,
):
    """Test the enhance function."""
    sample_rate = 16000

    listeners_file = tmp_path / "listeners.json"
    alphas_file = tmp_path / "alphas.json"
    scenes_file = tmp_path / "scenes.json"
    scene_listeners_file = tmp_path / "scene_listeners.json"
    musics_file = tmp_path / "musics.json"
    music_dir = tmp_path / "music"
    music_dir.mkdir()

    mixture = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))[:, np.newaxis]

    (music_dir / "segment1").mkdir()
    wavfile.write(music_dir / "segment1" / "mixture.wav", sample_rate, mixture)

    listeners_file.write_text(
        json.dumps(
            {
                "listener1": {
                    "audiogram": {
                        "left": [20, 30, 40, 50, 60, 70, 80],
                        "right": [20, 30, 40, 50, 60, 70, 80],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    alphas_file.write_text(
        json.dumps({"scene1": 0.5, "scene2": 0.7, "scene3": 1.0}), encoding="utf-8"
    )

    scenes_file.write_text(
        json.dumps({"scene1": {"alpha": "scene1", "segment_id": "segment1"}}),
        encoding="utf-8",
    )

    scene_listeners_file.write_text(
        json.dumps({"scene1": ["listener1"]}), encoding="utf-8"
    )

    musics_file.write_text(
        json.dumps({"segment1": {"path": "segment1", "start_time": 0, "end_time": 10}}),
        encoding="utf-8",
    )

    config = DictConfig(
        {
            "separator": {
                "causality": "causal",
                "device": "cpu",
                "separation": {
                    "segment": 10.0,
                    "overlap": 0.1,
                    "number_sources": 4,
                    "sample_rate": sample_rate,
                },
            },
            "path": {
                "listeners_file": str(listeners_file),
                "alphas_file": str(alphas_file),
                "scenes_file": str(scenes_file),
                "scene_listeners_file": str(scene_listeners_file),
                "musics_file": str(musics_file),
                "music_dir": str(music_dir),
            },
            "evaluate": {"small_test": False},
            "input_sample_rate": sample_rate,
            "remix_sample_rate": sample_rate,
            "soft_clip": False,
            "ha": {
                "compressor": {
                    "crossover_frequencies": [
                        353.55,
                        707.11,
                        1414.21,
                        2828.43,
                        5656.85,
                    ],
                    "sample_rate": sample_rate,
                },
                "camfit_gain_table": {
                    "noisegate_levels": 40,
                    "noisegate_slope": 0,
                    "cr_level": 0,
                    "max_output_level": 100,
                },
            },
        }
    )

    mock_load_separation_model.return_value = MagicMock()
    mock_load_listener_dict.return_value = {"listener1": MagicMock()}
    mock_read_signal.return_value = mixture

    enhance(config)

    mock_save_flac_signal.assert_called()


if __name__ == "__main__":
    pytest.main()
