"""Tests for the enhance module"""
import numpy as np
import pytest
import torch
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from recipes.cad1.task1.baseline.enhance import (
    apply_baseline_ha,
    decompose_signal,
    denormalize_signals,
    get_device,
    map_to_dict,
    normalize_signal,
    process_stems_for_listener,
    separate_sources,
)


@pytest.mark.parametrize(
    "signal,normalised_result,row_means",
    (  # Simple input
        [
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array(
                [
                    [-4.89897949, -3.67423461, -2.44948974],
                    [-1.22474487, 0.0, 1.22474487],
                    [2.44948974, 3.67423461, 4.89897949],
                ]
            ),
            np.array([4.0, 5.0, 6.0]),
        ],
        # Zero mean input
        [
            np.array([[-1, 0, 1], [-2, 0, 2], [-3, 0, 3]]),
            np.array(
                [
                    [-0.61237244, 0.0, 0.61237244],
                    [-1.22474487, 0.0, 1.22474487],
                    [-1.83711731, 0.0, 1.83711731],
                ]
            ),
            np.array([-2.0, 0.0, 2.0]),
        ],
        # Unit variance input
        [
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            np.array(
                [
                    [-4.89897949, -3.67423461, -2.44948974],
                    [-1.22474487, 0.0, 1.22474487],
                    [2.44948974, 3.67423461, 4.89897949],
                ]
            ),
            np.array([3.0, 4.0, 5.0]),
        ],
    ),
)
def test_normalize_signal(
    signal: np.ndarray, normalised_result: np.ndarray, row_means: np.ndarray
) -> None:
    """Test normalize_signal function"""
    result = normalize_signal(signal)
    np.testing.assert_array_almost_equal(result[0], normalised_result)
    np.testing.assert_array_almost_equal(result[1], row_means)


def test_denormalize_signals():
    # Test case 1: Test the function with a simple input
    sources = np.array(
        [
            np.array(
                [
                    [-1.22474487, 0.0, 1.22474487],
                    [-1.22474487, 0.0, 1.22474487],
                    [-1.22474487, 0.0, 1.22474487],
                ]
            ),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ]
    )
    ref = np.array([4.0, 5.0, 6.0])
    expected_result = [
        np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]),
        np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]),
    ]
    result = denormalize_signals(sources, ref)
    assert len(result) == len(expected_result)
    assert np.allclose(result[0], expected_result[0])
    assert np.allclose(result[1], expected_result[1])

    # Test case 2: Test the function with an input that has a non-zero mean
    sources = np.array(
        [
            np.array(
                [
                    [-1.22474487, 0.0, 1.22474487],
                    [-1.22474487, 0.0, 1.22474487],
                    [-1.22474487, 0.0, 1.22474487],
                ]
            ),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ]
    )
    ref = np.array([1.0, 2.0, 3.0])
    expected_result = [
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
    ]
    result = denormalize_signals(sources, ref)
    assert len(result) == len(expected_result)
    assert np.allclose(result[0], expected_result[0])
    assert np.allclose(result[1], expected_result[1])

    # Test case 3: Test the function with an input that has a non-unit variance
    sources = np.array(
        [
            np.array(
                [
                    [-2.44948974, 0.0, 2.44948974],
                    [-2.44948974, 0.0, 2.44948974],
                    [-2.44948974, 0.0, 2.44948974],
                ]
            ),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ]
    )
    ref = np.array([4.0, 5.0, 6.0])
    expected_result = [
        np.array([[3.0, 5.0, 7.0], [3.0, 5.0, 7.0], [3.0, 5.0, 7.0]]),
        np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]),
    ]
    result = denormalize_signals(sources, ref)
    assert len(result) == len(expected_result)
    assert np.allclose(result[0], expected_result[0])
    assert np.allclose(result[1], expected_result[1])


def test_map_to_dict(regtest):
    sources = np.array([[1, 2], [3, 4], [5, 6]])
    sources_list = ["a", "b", "c"]
    output = map_to_dict(sources, sources_list)
    regtest.write(f"output: \n{output}\n")


def test_decompose_signal(regtest):
    np.random.seed(123456789)
    # Load Separation Model
    model = HDEMUCS_HIGH_MUSDB.get_model().double()
    device = torch.device("cpu")
    model.to(device)

    # Create a mock signal to decompose
    fs = 44100
    duration = 5
    signal = np.random.uniform(size=(1, 2, fs * duration))
    # Call the decompose_signal function and check that the output has the expected keys
    output = decompose_signal(model, signal, fs, device)
    regtest.write(f"output: \n{output}\n")


def test_apply_baseline_ha(regtest):
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
    regtest.write(f"output: \n{output}\n")


def test_process_stems_for_listener(regtest):
    np.random.seed(12357)
    # Create mock inputs
    stems = {
        "l_source1": np.random.normal(size=44100),
        "r_source1": np.random.normal(size=44100),
    }
    audiogram_left = np.ones(9)
    audiogram_right = np.ones(9)
    cfs = np.array([250, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000])

    # Create mock objects for enhancer and compressor
    enhancer = NALR(nfir=220, fs=44100)
    compressor = Compressor(
        threshold=0.35, attenuation=0.1, attack=50, release=1000, rms_buffer_size=0.064
    )

    # Call the process_stems_for_listener function and check that the output is as expected
    output = process_stems_for_listener(
        stems, enhancer, compressor, audiogram_left, audiogram_right, cfs
    )

    # Check that the processed stems are different from the input stems
    for stem_str in stems:
        regtest.write(f"{stem_str}: \n{output[stem_str]}\n")


def test_separate_sources(regtest):
    np.random.seed(123456789)

    # Create a dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self, sources):
            super().__init__()
            self.sources = sources

        def forward(self, x):
            return torch.Tensor(
                np.random.uniform(size=(x.shape[0], len(self.sources), *x.shape[1:]))
            )

    # Set up some dummy input data
    batch_size = 2
    num_channels = 2
    length = 5
    sample_rate = 16000
    sources = ["vocals", "drums", "bass", "other"]
    mix = np.random.randn(batch_size, num_channels, length * sample_rate)
    device = torch.device("cpu")

    # Create a dummy model
    model = DummyModel(sources)

    # Call separate_sources
    sources = separate_sources(model, mix, sample_rate, device=device)

    # Check that the output has the correct shape
    regtest.write(f"sources.shape: \n{sources.shape}\n")
    regtest.write(f"sources: \n{sources}\n")


def test_get_device():
    # Test default case (no argument passed)
    device, device_type = get_device(None)
    assert (
        device == torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    assert device_type == "cuda" if torch.cuda.is_available() else "cpu"
