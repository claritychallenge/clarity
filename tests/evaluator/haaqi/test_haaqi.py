import numpy as np
import pytest

from clarity.evaluator.haaqi import haaqi_v1


def test_haaqi_v1() -> None:
    """Test for haaqi_v1 index"""
    np.random.seed(0)
    sr = 16000
    x = np.random.uniform(-1, 1, sr * 10)
    y = np.random.uniform(-1, 1, sr * 10)

    hl = np.array([45, 45, 35, 45, 60, 65])
    eq = 1
    level1 = 65

    score, _, _, _ = haaqi_v1(x, sr, y, sr, hl, eq, level1)
    print(score)
    assert score == pytest.approx(0.109534910970557, rel=1e-4)
