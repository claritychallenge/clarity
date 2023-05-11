"""Tests for the enhance module"""
from pathlib import Path

# pylint: disable=import-error
import numpy as np
import pytest
import soundfile as sf
import torch
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.signal_processing import denormalize_signals, normalize_signal
from recipes.cad1.task1.baseline.enhance import (
    apply_baseline_ha,
    decompose_signal,
    get_device,
    map_to_dict,
    process_stems_for_listener,
    remix_signal,
    save_flac_signal,
    separate_sources,
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


def test_decompose_signal():
    """Takes a signal and decomposes it into VDBO sources using the HDEMUCS model"""
    np.random.seed(123456789)
    # Load Separation Model
    model = HDEMUCS_HIGH_MUSDB.get_model().double()
    device = torch.device("cpu")
    model.to(device)

    # Create a mock signal to decompose
    sample_rate = 8000
    duration = 1
    signal = np.random.uniform(size=(1, 2, sample_rate * duration))

    # Normalise using demucs procedure
    signal, ref = normalize_signal(signal)
    # Call the decompose_signal function and check that the output has the expected keys
    output = decompose_signal(
        model=model,
        model_sample_rate=sample_rate,
        signal=signal,
        signal_sample_rate=sample_rate,
        device=device,
        sources_list=model.sources,
        left_audiogram=np.ones(9),
        right_audiogram=np.ones(9),
        normalise=True,
    )

    for key, item in output.items():
        output[key] = denormalize_signals(item, ref)

    expected_results = np.load(
        RESOURCES / "test_enhance.test_decompose_signal.npy",
        allow_pickle=True,
    )[()]

    for key, item in output.items():
        np.testing.assert_array_almost_equal(item, expected_results[key])


def test_apply_baseline_ha():
    """Test the behaviour of the CAD1 - Task1 - baseline hearing aid"""
    np.random.seed(987654321)
    # Create mock inputs
    signal = np.random.normal(size=44100)
    listener_audiogram = np.ones(9)
    cfs = np.array([250, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000])

    # Create mock objects for enhancer and compressor
    enhancer = NALR(nfir=220, fs=44100)
    compressor = Compressor(
        threshold=0.35, attenuation=0.1, attack=50, release=1000, rms_buffer_size=0.064
    )

    # Call the apply_nalr function and check that the output is as expected
    output = apply_baseline_ha(enhancer, compressor, signal, listener_audiogram, cfs)

    expected_results = np.load(
        RESOURCES / "test_enhance.test_apply_baseline_ha.npy",
        allow_pickle=True,
    )
    np.testing.assert_array_almost_equal(output, expected_results)


def test_process_stems_for_listener():
    """Takes 2 stems and applies the baseline processing using a listeners audiograms"""
    np.random.seed(12357)
    # Create mock inputs
    stem_signals = {
        "l_source1": np.random.normal(size=16000),
        "r_source1": np.random.normal(size=16000),
    }
    audiogram_left = np.ones(9)
    audiogram_right = np.ones(9)
    cfs = np.array([250, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000])

    # Create mock objects for enhancer and compressor
    enhancer = NALR(nfir=220, fs=16000)
    compressor = Compressor(
        threshold=0.35, attenuation=0.1, attack=50, release=1000, rms_buffer_size=0.064
    )

    # Call the process_stems_for_listener function and check output is as expected
    output = process_stems_for_listener(
        stem_signals, enhancer, compressor, audiogram_left, audiogram_right, cfs
    )
    expected_results = np.load(
        RESOURCES / "test_enhance.test_process_stems_for_listener.npy",
        allow_pickle=True,
    )[()]

    for key, item in output.items():
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


def test_remix_signal():
    np.random.seed(0)
    n_samples = 1000
    stems = {
        "l1": np.random.rand(n_samples),
        "l2": np.random.rand(n_samples),
        "l3": np.random.rand(n_samples),
        "l4": np.random.rand(n_samples),
        "r1": np.random.rand(n_samples),
        "r2": np.random.rand(n_samples),
        "r3": np.random.rand(n_samples),
        "r4": np.random.rand(n_samples),
    }

    remixed = remix_signal(stems)
    assert isinstance(remixed, np.ndarray)
    assert remixed.shape[0] == stems["l1"].shape[0]
    assert remixed.shape[1] == 2
    assert np.sum(remixed[:, 0]) == pytest.approx(
        np.sum(stems["l1"] + stems["l2"] + stems["l3"] + stems["l4"]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )
    assert np.sum(remixed[:, 1]) == pytest.approx(
        np.sum(stems["r1"] + stems["r2"] + stems["r3"] + stems["r4"]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )


def test_save_signal(tmp_path):
    np.random.seed(0)
    input_signal = np.random.rand(1600)
    output_path = Path(tmp_path) / "output.flac"

    save_flac_signal(
        input_signal, output_path, signal_sample_rate=16000, output_sample_rate=16000
    )
    assert output_path.is_file()
    signal, sr = sf.read(output_path)

    assert sr == 16000
    assert np.sum(signal) == pytest.approx(
        807.2321472167969, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
