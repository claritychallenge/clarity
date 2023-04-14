"""Tests for the enhancer.nalr module."""
import numpy as np
import pytest

from clarity.enhancer.nalr import NALR

cfg_nalr = {"nfir": 220, "sample_rate": 44100}


STANDARD_CFS = np.array([250, 500, 1000, 2000, 4000, 6000])


@pytest.mark.parametrize(
    "audiogram, cfs, expected",
    [
        (
            np.array([45, 45, 35, 45, 60, 65]),
            np.array([250, 500, 1000, 1500, 4000, 6000]),  # <-- non default cfs
            29.49383298688706,
        ),
        (
            np.array([45, 45, 35, 45, 60, 65]),
            None,
            28.846583644263408,
        ),
    ],
)
def test_nalr(audiogram: np.ndarray, cfs: np.ndarray, expected: float) -> None:
    """Test that the NALR filter is built correctly."""
    enhancer = NALR(**cfg_nalr)
    nalr_fir, _ = enhancer.build(audiogram, cfs)
    assert np.sum(np.abs(nalr_fir)) == pytest.approx(expected)


def test_nalr_default_cfs() -> None:
    """Test that the NALR filter is the same for the default and standard cfs."""
    enhancer = NALR(**cfg_nalr)
    nalr_fir1, _ = enhancer.build(np.array([45, 45, 35, 45, 60, 65]))
    nalr_fir2, _ = enhancer.build(np.array([45, 45, 35, 45, 60, 65]), STANDARD_CFS)
    assert np.allclose(nalr_fir1, nalr_fir2)
