"""Tests for hasqi module"""
import numpy as np
import pytest

from clarity.evaluator.haspi import haspi_v2, haspi_v2_be


def test_haspi_v2() -> None:
    """Test for hasqi_v2 index"""
    np.random.seed(0)
    sample_rate = 16000
    x = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    y = np.random.uniform(-1, 1, int(sample_rate * 0.5))

    hearing_loss = np.array([45, 45, 35, 45, 60, 65])
    level1 = 65

    score, _ = haspi_v2(x, sample_rate, y + x, sample_rate, hearing_loss, level1)
    assert score == pytest.approx(
        0.043808448934532965, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_haspi_v2_better_ear() -> None:
    """Test for hasqi_v2_better_ear index"""

    np.random.seed(0)
    sample_rate = 16000
    hl_left = np.array([25, 25, 25, 25, 40, 65])
    hl_right = np.array([45, 45, 35, 45, 60, 65])

    freqs = np.array([250, 500, 1000, 2000, 4000, 6000])
    ref_left = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    ref_right = np.random.uniform(-1, 1, int(sample_rate * 0.5))
    proc_left = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    proc_right = np.random.uniform(-1, 1, int(sample_rate * 0.5))

    score = haspi_v2_be(
        reference_left=ref_left,
        reference_right=ref_right,
        processed_left=proc_left + ref_left,
        processed_right=proc_right,
        sample_freq=sample_rate,
        audiogram_left=hl_left,
        audiogram_right=hl_right,
        audiogram_frequencies=freqs,
        level=100,
    )

    assert score == pytest.approx(
        0.839975335323691, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
