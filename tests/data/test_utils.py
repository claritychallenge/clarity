"""Tests for the data.utils module."""
from __future__ import annotations

import numpy as np
import pytest

from clarity.data.utils import (
    better_ear_speechweighted_snr,
    pad,
    speechweighted_snr,
    sum_signals,
)

SEED = 864813
rng = np.random.default_rng(SEED)


@pytest.mark.parametrize(
    "target, noise, expected",
    [
        (
            np.asarray([1, 2, 3, 4]),
            np.asarray([[4, 3, 2, 1], [4, 3, 2, 1]]),
            1.5296464383117123,
        ),
        (
            np.asarray(rng.random(20)),
            np.asarray(rng.random((2, 10))),
            37.079860336786695,
        ),
        (
            np.asarray(rng.random((2, 10))),
            np.asarray(rng.random((2, 10))),
            1.3658122884632578,
        ),
        (
            np.asarray(rng.random((100, 100))),
            np.asarray(rng.random((100, 100))),
            0.9781085375091217,
        ),
    ],
)
def test_better_ear_speechweighted_snr(
    target: np.ndarray, noise: np.ndarray, expected: float
) -> None:
    """Test of better_ear_speechweighted_snr()."""
    better_ear_signal_noise_ratio = better_ear_speechweighted_snr(target, noise)
    assert isinstance(better_ear_signal_noise_ratio, float)
    assert better_ear_signal_noise_ratio == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_better_ear_speechweighted_snr_noise_wrong_shape_index_error() -> None:
    """Test of better_ear_speechweighted_snr()."""
    with pytest.raises(IndexError):
        better_ear_speechweighted_snr(
            target=np.asarray([1, 2, 3, 4]), noise=np.asarray([4, 3, 2, 1])
        )


@pytest.mark.parametrize(
    "target, noise, expected",
    [
        (np.asarray([1, 2, 3, 4]), np.asarray([4, 3, 2, 1]), 1.0),
        (
            np.asarray([1.1, 2.2, 3.3, 4.4]),
            np.asarray([11.1, 22.2, 33.3, 44.4]),
            0.0990990990990991,
        ),
        (np.asarray(rng.random(20)), np.asarray(rng.random(20)), 1.0961240900775582),
        (np.asarray(rng.random(200)), np.asarray(rng.random(200)), 0.9653767031879813),
        (
            np.asarray(rng.random(1000)),
            np.asarray(rng.random(1000)),
            1.0195987176010093,
        ),
    ],
)
def test_speechweighted_snr(
    target: np.ndarray, noise: np.ndarray, expected: float
) -> None:
    """Test of speechweighted_snr()."""
    signal_noise_ratio = speechweighted_snr(target, noise)
    assert isinstance(signal_noise_ratio, float)
    assert signal_noise_ratio == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_speechweighted_snr_unequal_arrays_value_error() -> None:
    """Test of speechweighted_snr()."""
    with pytest.raises(ValueError):
        speechweighted_snr(
            target=np.asarray([[1, 2], [3, 4]]), noise=np.asarray([3, 4, 1, 2])
        )


@pytest.mark.parametrize(
    "signals, expected",
    [
        (
            [
                np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]),
                np.asarray([[9, 10], [11, 12], [13, 14]]),
                np.asarray([[15, 16], [17, 18]]),
            ],
            np.asarray([[25, 28], [31, 34], [18, 20], [7, 8]]),
        ),
        (
            [
                np.asarray([[1, 2], [3, 4]]),
                np.asarray([[5, 6], [7, 8]]),
                np.asarray([[9, 10], [11, 12]]),
            ],
            np.asarray([[15, 18], [21, 24]]),
        ),
    ],
)
def test_sum_signals(signals: list[int], expected: np.ndarray) -> None:
    """Test of sum_signals()."""
    summed_signals = sum_signals(signals)
    np.testing.assert_array_equal(summed_signals, expected)


@pytest.mark.parametrize(
    "signal, length, expected",
    [
        (
            np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]),
            4,
            np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]),
        ),
        (
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            4,
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]]),
        ),
    ],
)
def test_pad(signal: np.ndarray, length: int, expected: np.ndarray) -> None:
    """Test of pad()."""
    padded_array = pad(signal, length)
    assert isinstance(padded_array, np.ndarray)
    np.testing.assert_array_equal(padded_array, expected)


def test_pad_invalid_length_assertion_error() -> None:
    """Test pad() raises an exception when length < signal.shape[0]."""
    with pytest.raises(ValueError):
        pad(signal=np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), length=2)
