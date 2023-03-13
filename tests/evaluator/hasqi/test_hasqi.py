"""Tests for hasqi module"""
import numpy as np
import pytest

from clarity.evaluator.hasqi import hasqi_v2


def test_hasqi_v2() -> None:
    """Test for hasqi_v2 index"""
    np.random.seed(0)
    sample_rate = 16000
    x = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    y = np.random.uniform(-1, 1, int(sample_rate * 0.5))

    hearing_loss = np.array([45, 45, 35, 45, 60, 65])
    equalisation_mode = 1
    level1 = 65

    score, _, _, _ = hasqi_v2(
        x, sample_rate, y, sample_rate, hearing_loss, equalisation_mode, level1
    )
    assert score == pytest.approx(0.002525809, rel=1e-7)
