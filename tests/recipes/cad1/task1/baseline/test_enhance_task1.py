"""Tests for the enhance module"""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram, Listener

# pylint: disable=import-error, no-name-in-module
from recipes.cad1.task1.baseline.enhance import (
    apply_baseline_ha,
    clip_signal,
    decompose_signal,
    get_device,
    map_to_dict,
    process_stems_for_listener,
    separate_sources,
    to_16bit,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources" / "recipes" / "cad1" / "task1"


def test_map_to_dict():
    """Test that the map_to_dict returns the expected mapping"""
    sources = np.array([[1, 2], [3, 4], [5, 6]])
    sources_list = ["a", "b", "c"]
    output = map_to_dict(sources, sources_list)
    expected_output = {
        "left_a": 1,
        "right_a": 2,
        "left_b": 3,
        "right_b": 4,
        "left_c": 5,
        "right_c": 6,
    }
    assert output == expected_output


@pytest.mark.parametrize(
    "separation_model",
    [
        pytest.param("demucs"),
        pytest.param("openunmix", marks=pytest.mark.slow),
    ],
)
def test_decompose_signal(separation_model):
    """Takes a signal and decomposes it into VDBO sources using the HDEMUCS model"""
    np.random.seed(123456789)
    # Load Separation Model
    if separation_model == "demucs":
        model = HDEMUCS_HIGH_MUSDB.get_model().double()
    elif separation_model == "openunmix":
        model = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq").double()

    device = torch.device("cpu")
    model.to(device)

    # Create a mock signal to decompose
    sample_rate = 44100
    duration = 0.5
    signal = np.random.uniform(size=(1, 2, int(sample_rate * duration)))

    # config
    config = DictConfig(
        {
            "sample_rate": sample_rate,
            "separator": {
                "model": "demucs",
                "sources": ["drums", "bass", "other", "vocals"],
            },
        }
    )
    # Call the decompose_signal function and check that the output has the expected keys
    cfs = np.array([250, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000])
    audiogram = Audiogram(levels=np.ones(9), frequencies=cfs)
    listener = Listener(audiogram, audiogram)
    output = decompose_signal(
        config,
        model,
        signal,
        sample_rate,
        device,
        listener,
    )
    expected_results = np.load(
        RESOURCES / f"test_enhance.test_decompose_signal_{separation_model}.npy",
        allow_pickle=True,
    )[()]

    for key, item in output.items():
        np.testing.assert_array_almost_equal(item, expected_results[key])


def test_apply_baseline_ha():
    """Test the behaviour of the CAD1 - Task1 - baseline hearing aid"""
    np.random.seed(987654321)
    # Create mock inputs
    signal = np.random.normal(size=44100)
    listener_audiogram = Audiogram(
        levels=np.ones(9),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000]),
    )

    # Create mock objects for enhancer and compressor
    enhancer = NALR(nfir=220, sample_rate=44100)
    compressor = Compressor(
        threshold=0.35, attenuation=0.1, attack=50, release=1000, rms_buffer_size=0.064
    )

    # Call the apply_nalr function and check that the output is as expected
    output = apply_baseline_ha(enhancer, compressor, signal, listener_audiogram)

    expected_results = np.load(
        RESOURCES / "test_enhance.test_apply_baseline_ha.npy",
        allow_pickle=True,
    )
    np.testing.assert_array_almost_equal(output, expected_results)


def test_process_stems_for_listener():
    """Takes 2 stems and applies the baseline processing using a listeners audiograms"""
    np.random.seed(12357)
    # Create mock inputs
    stems = {
        "l_source1": np.random.normal(size=16000),
        "r_source1": np.random.normal(size=16000),
    }

    audiogram = Audiogram(
        levels=np.ones(9),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000]),
    )
    listener = Listener(audiogram_left=audiogram, audiogram_right=audiogram)
    # Create mock objects for enhancer and compressor
    enhancer = NALR(nfir=220, sample_rate=16000)
    compressor = Compressor(
        threshold=0.35, attenuation=0.1, attack=50, release=1000, rms_buffer_size=0.064
    )

    # Call the process_stems_for_listener function and check output is as expected
    output_stems = process_stems_for_listener(
        stems, enhancer, compressor, listener=listener
    )
    expected_results = np.load(
        RESOURCES / "test_enhance.test_process_stems_for_listener.npy",
        allow_pickle=True,
    )[()]

    for key, item in output_stems.items():
        np.testing.assert_array_almost_equal(item, expected_results[key])


def test_separate_sources():
    """Test that the separate_sources function returns the expected output"""
    np.random.seed(123456789)

    # Create a dummy model
    class DummyModel(torch.nn.Module):  # pylint: disable=too-few-public-methods
        """Dummy source separation model"""

        def __init__(self, sources):
            """dummy init"""
            super().__init__()
            self.sources = sources

        def forward(self, x):
            """dummy forward"""
            return torch.Tensor(
                np.random.uniform(size=(x.shape[0], len(self.sources), *x.shape[1:]))
            )

    # Set up some dummy input data
    batch_size = 1
    num_channels = 1
    length = 1
    sample_rate = 16000
    sources = ["vocals", "drums", "bass", "other"]
    mix = np.random.randn(batch_size, num_channels, length * sample_rate)
    device = torch.device("cpu")

    # Create a dummy model
    model = DummyModel(sources)

    # Call separate_sources
    output = separate_sources(model, mix, sample_rate, device=device)

    expected_results = np.load(
        RESOURCES / "test_enhance.test_separate_sources.npy",
        allow_pickle=True,
    )
    # Check that the output has the correct shape
    assert output.shape == expected_results.shape
    np.testing.assert_array_almost_equal(output, expected_results)


def test_get_device():
    """Test the correct device selection given the inputs"""
    # Test default case (no argument passed)
    device, device_type = get_device(None)
    assert (
        device == torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    assert device_type == "cuda" if torch.cuda.is_available() else "cpu"


def test_to_16bit():
    # Generate a random signal
    signal = np.random.uniform(low=-1.0, high=1.0, size=50)
    signal_16bit = to_16bit(signal)

    assert np.all(np.abs(signal_16bit) <= 32768)


def test_clip_signal():
    # Generate a random signal
    np.random.seed(0)
    signal = np.random.uniform(low=-2.0, high=2.0, size=50)

    # Test with soft clipping
    clipped_signal, n_clipped = clip_signal(signal, soft_clip=True)
    assert max(np.abs(clipped_signal)) <= 1.0
    assert n_clipped == 0

    # Test without soft clipping
    clipped_signal, n_clipped = clip_signal(signal, soft_clip=False)
    assert max(np.abs(clipped_signal)) <= 1.0
    assert n_clipped == 22
