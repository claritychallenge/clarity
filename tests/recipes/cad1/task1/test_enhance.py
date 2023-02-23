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


@pytest.mark.parametrize(
    "sources,ref,expected_result",
    (  # Test case 1: Test the function with a simple input
        [
            np.array(
                [
                    [
                        [-1.22474487, 0.0, 1.22474487],
                        [-1.22474487, 0.0, 1.22474487],
                        [-1.22474487, 0.0, 1.22474487],
                    ],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
            np.array([4.0, 5.0, 6.0]),
            [
                np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]),
                np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]),
            ],
        ],
        # Test case 2: Test the function with an input that has a non-zero mean
        [
            np.array(
                [
                    [
                        [-1.22474487, 0.0, 1.22474487],
                        [-1.22474487, 0.0, 1.22474487],
                        [-1.22474487, 0.0, 1.22474487],
                    ],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
            np.array([1.0, 2.0, 3.0]),
            [
                np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
                np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            ],
        ],
        # Test case 3: Test the function with an input that has a non-unit variance
        [
            np.array(
                [
                    [
                        [-2.44948974, 0.0, 2.44948974],
                        [-2.44948974, 0.0, 2.44948974],
                        [-2.44948974, 0.0, 2.44948974],
                    ],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
            np.array([4.0, 5.0, 6.0]),
            [
                np.array([[3.0, 5.0, 7.0], [3.0, 5.0, 7.0], [3.0, 5.0, 7.0]]),
                np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]),
            ],
        ],
    ),
)
def test_denormalize_signals(
    sources: np.ndarray, ref: np.ndarray, expected_result: np.ndarray
) -> None:
    """Test denormalize_signal function"""
    result = denormalize_signals(sources, ref)
    assert len(result) == len(expected_result)
    np.testing.assert_array_almost_equal(result[0], expected_result[0])
    np.testing.assert_array_almost_equal(result[1], expected_result[1])


def test_map_to_dict():
    sources = np.array([[1, 2], [3, 4], [5, 6]])
    sources_list = ["a", "b", "c"]
    output = map_to_dict(sources, sources_list)
    expected_output = {"l_a": 1, "r_a": 2, "l_b": 3, "r_b": 4, "l_c": 5, "r_c": 6}
    assert output == expected_output


def test_decompose_signal():
    np.random.seed(123456789)
    # Load Separation Model
    model = HDEMUCS_HIGH_MUSDB.get_model().double()
    device = torch.device("cpu")
    model.to(device)

    # Create a mock signal to decompose
    fs = 8000
    duration = 1
    signal = np.random.uniform(size=(1, 2, fs * duration))
    # Call the decompose_signal function and check that the output has the expected keys
    output = decompose_signal(model, signal, fs, device)
    expected_results = np.load(
        "../../../resources/test_enhance.test_decompose_signal.npy", allow_pickle=True
    )[()]
    for key in output.keys():
        np.testing.assert_array_almost_equal(output[key], expected_results[key])


def test_apply_baseline_ha():
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
        "../../../resources/test_enhance.test_apply_baseline_ha.npy", allow_pickle=True
    )
    np.testing.assert_array_almost_equal(output, expected_results)


def test_process_stems_for_listener():
    np.random.seed(12357)
    # Create mock inputs
    stems = {
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

    # Call the process_stems_for_listener function and check that the output is as expected
    output = process_stems_for_listener(
        stems, enhancer, compressor, audiogram_left, audiogram_right, cfs
    )
    expected_results = np.load(
        "../../../resources/test_enhance.test_process_stems_for_listener.npy",
        allow_pickle=True,
    )[()]
    for key in output.keys():
        np.testing.assert_array_almost_equal(output[key], expected_results[key])


def test_separate_sources():
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
        "../../../resources/test_enhance.test_separate_sources.npy", allow_pickle=True
    )
    # Check that the output has the correct shape
    assert output.shape == expected_results.shape
    np.testing.assert_array_almost_equal(output, expected_results)


def test_get_device():
    # Test default case (no argument passed)
    device, device_type = get_device(None)
    assert (
        device == torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    assert device_type == "cuda" if torch.cuda.is_available() else "cpu"
