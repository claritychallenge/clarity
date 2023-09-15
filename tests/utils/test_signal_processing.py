"""Test for utils.signal_processing module"""
# pylint: disable=import-error
import numpy as np
import pytest

from clarity.utils.signal_processing import (
    clip_signal,
    compute_rms,
    denormalize_signals,
    normalize_signal,
    resample,
)


def test_clip_signal_hard_clip():
    # Test hard clipping (default behavior)
    input_signal = np.array([1.5, -0.5, 2.0, -2.5, 0.75])
    clipped_signal, n_clipped = clip_signal(input_signal)
    assert np.all(clipped_signal == np.array([1.0, -0.5, 1.0, -1.0, 0.75]))
    assert n_clipped == 3


def test_clip_signal_soft_clip():
    # Test soft clipping
    input_signal = np.array([1.5, -0.5, 2.0, -2.5, 0.75])
    clipped_signal, n_clipped = clip_signal(input_signal, soft_clip=True)
    assert np.sum(clipped_signal) == pytest.approx(
        np.sum([0.90514825, -0.46211716, 0.96402758, -0.9866143, 0.63514895])
    )
    assert n_clipped == 0


def test_clip_signal_all_within_range():
    # Test when all values are already within the range
    input_signal = np.array([-0.8, 0.2, -0.6, 0.9])
    clipped_signal, n_clipped = clip_signal(input_signal)
    assert np.all(clipped_signal == input_signal)
    assert n_clipped == 0


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


def test_compute_rms():
    """Test the function compute_rms"""
    np.random.seed(0)
    sig_len = 600
    signal = 100 * np.random.random(size=sig_len)
    rms = compute_rms(signal)
    assert rms == pytest.approx(
        57.803515840, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "input_sample_rate, input_shape, output_sample_rate, output_shape",
    [
        (16000, (100, 2), 16000, (100, 2)),
        (16000, (100, 2), 8000, (50, 2)),
        (16000, (100, 2), 32000, (200, 2)),
        (16000, (100,), 8000, (50,)),
        (16000, (100, 1), 8000, (50, 1)),
        (16000, 100, 8000, (50,)),
    ],
)
def test_resample(input_sample_rate, input_shape, output_sample_rate, output_shape):
    """Test the signal resampling function"""
    input_signal = np.ones(input_shape)

    for method in ["soxr", "polyphase", "fft"]:
        output_signal = resample(
            signal=input_signal,
            sample_rate=input_sample_rate,
            new_sample_rate=output_sample_rate,
            method=method,
        )

        assert output_signal.shape == output_shape


def test_resample_default_to_soxr():
    """Test the signal resampling function"""
    input_signal = np.ones((100, 2))
    output_signal_default = resample(
        signal=input_signal, sample_rate=16000, new_sample_rate=8000
    )
    output_signal_soxr = resample(
        signal=input_signal, sample_rate=16000, new_sample_rate=8000, method="soxr"
    )

    assert output_signal_default == pytest.approx(output_signal_soxr)


def test_resample_raise_error_for_unknown_method():
    """Test the signal resampling function"""
    input_signal = np.ones((100, 2))
    with pytest.raises(ValueError):
        resample(
            signal=input_signal,
            sample_rate=16000,
            new_sample_rate=8000,
            method="unknown",
        )


def test_resample_with_3d_array():
    """Test the signal resampling function"""
    input_signal = np.ones((100, 2, 3))
    for method in ["polyphase", "fft"]:
        output_signal = resample(
            signal=input_signal, sample_rate=16000, new_sample_rate=8000, method=method
        )
        assert output_signal.shape == (50, 2, 3)


def test_resample_with_3d_array_error():
    """Resample with 3d array should raise an error for soxr and polyphase methods"""
    input_signal = np.ones((100, 2, 3))
    with pytest.raises(ValueError):
        resample(
            signal=input_signal, sample_rate=16000, new_sample_rate=8000, method="soxr"
        )
