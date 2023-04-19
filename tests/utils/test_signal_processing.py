"""Test for utils.signal_processing module"""
import numpy as np
import pytest

from clarity.utils.signal_processing import (
    compute_rms,
    denormalize_signals,
    normalize_signal,
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


def test_compute_rms():
    """Test the function compute_rms"""
    np.random.seed(0)
    sig_len = 600
    signal = 100 * np.random.random(size=sig_len)
    rms = compute_rms(signal)
    assert rms == pytest.approx(
        57.803515840, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
