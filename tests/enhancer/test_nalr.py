"""Tests for the enhancer.nalr module."""
import numpy as np
import pytest

from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram

cfg_nalr = {"nfir": 220, "sample_rate": 44100}


STANDARD_CFS = np.array([250, 500, 1000, 2000, 4000, 6000])


@pytest.mark.parametrize(
    "levels, cfs, expected",
    [
        (
            np.array([45, 45, 35, 45, 60, 65]),
            np.array([250, 500, 1000, 1500, 4000, 6000]),  # <-- non default cfs
            29.808253441926134,  # was 29.493833... (New behaviour due to log freq axis)
        ),
        (
            np.array([45, 45, 35, 45, 60, 65]),
            np.array([250, 500, 1000, 2000, 4000, 6000]),
            28.846583644263408,
        ),
    ],
)
def test_nalr(levels: np.ndarray, cfs: np.ndarray, expected: float) -> None:
    """Test that the NALR filter is built correctly."""
    enhancer = NALR(**cfg_nalr)
    audiogram = Audiogram(levels=levels, frequencies=cfs)
    nalr_fir, _ = enhancer.build(audiogram)
    assert np.sum(np.abs(nalr_fir)) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_nalr_default_cfs() -> None:
    """Test that the NALR filter is the same for the default and standard cfs."""
    enhancer = NALR(**cfg_nalr)
    nalr_fir1, _ = enhancer.build(
        Audiogram(np.array([45, 45, 35, 45, 60, 65]), STANDARD_CFS)
    )
    assert nalr_fir1 == pytest.approx(
        nalr_fir1, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
