import numpy as np
import pytest

from clarity.evaluator.hasqi import hasqi_v2


def test_hasqi_v2() -> None:
    """Test for hasqi_v2 index"""
    np.random.seed(0)
    sr = 16000
    x = np.random.uniform(-1, 1, sr * 10)
    y = np.random.uniform(-1, 1, sr * 10)

    hl = np.array([45, 45, 35, 45, 60, 65])
    eq = 1
    level1 = 65

    score, _, _, _ = hasqi_v2(x, sr, y, sr, hl, eq, level1)
    assert score == pytest.approx(0.0012190655936307408, rel=1e-4)
