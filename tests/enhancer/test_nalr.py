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
            np.array([250, 500, 1000, 2000, 4000, 6000]),
            3.9861248204486366e-05,
        ),
        (
            np.array([45, 45, 35, 45, 60, 65]),
            None,
            3.9861248204486366e-05,
        ),
    ],
)
def test_nalr(audiogram: np.ndarray, cfs: np.ndarray, expected: float) -> None:
    enhancer = NALR(**cfg_nalr)
    nalr_fir, _ = enhancer.build(audiogram, cfs)
    assert nalr_fir[0] == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
