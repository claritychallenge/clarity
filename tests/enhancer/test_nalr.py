"""Tests for the enhancer.nalr module."""
import numpy as np
import pytest

from clarity.enhancer.nalr import NALR

cfg_nalr = {"nfir": 220, "fs": 44100}


@pytest.mark.parametrize(
    "audiogram, cfs, expected",
    [
        (
            np.array([45, 45, 35, 45, 60, 65]),
            np.array([250, 500, 1000, 2000, 3000, 6000]),
            5.746930975720798e-05,
        ),
        (
            np.array([45, 45, 35, 45, 60, 65]),
            None,
            5.746930975720798e-05,
        ),
    ],
)
def test_nalr(audiogram: np.ndarray, cfs: np.ndarray, expected: float) -> None:
    enhancer = NALR(**cfg_nalr)
    nalr_fir, _ = enhancer.build(audiogram, cfs)
    assert nalr_fir[0] == expected
