"""Tests for hasqi module"""
import numpy as np
import pytest

from clarity.evaluator.haspi import haspi_v2, haspi_v2_be
from clarity.utils.audiogram import Audiogram, Listener


def test_haspi_v2() -> None:
    """Test for hasqi_v2 index"""
    np.random.seed(0)
    sample_rate = 16000
    x = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    y = np.random.uniform(-1, 1, int(sample_rate * 0.5))

    hearing_loss = np.array([45, 45, 35, 45, 60, 65])
    freqs = np.array([250, 500, 1000, 2000, 4000, 6000])
    audiogram = Audiogram(levels=hearing_loss, frequencies=freqs)
    level1 = 65

    score, _ = haspi_v2(x, sample_rate, y + x, sample_rate, audiogram, level1)
    assert score == pytest.approx(
        0.043808448934532965, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "hl_left, hl_right, freqs, expected_score",
    [
        (
            np.array([25, 25, 25, 25, 40, 65]),
            np.array([45, 45, 35, 45, 60, 65]),
            np.array([250, 500, 1000, 2000, 4000, 6000]),
            0.839975335323691,
        ),
        (
            np.array([25, 25, 21, 25, 25, 40, 65]),
            np.array([45, 45, 21, 35, 45, 60, 65]),
            np.array([250, 500, 700, 1000, 2000, 4000, 6000]),  # <-- extra cf added
            0.839975335323691,
        ),
        (
            np.array([25, 25, 25, 65]),
            np.array([45, 45, 45, 65]),
            np.array([250, 500, 2000, 6000]),  # <-- missing cfs, need interp
            0.8380310721987008,  # <-- note different score, as expected
        ),
    ],
)
def test_haspi_v2_better_ear(hl_left, hl_right, freqs, expected_score) -> None:
    """Test for hasqi_v2_better_ear index"""

    sample_rate = 16000

    np.random.seed(0)
    ref_left = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    ref_right = np.random.uniform(-1, 1, int(sample_rate * 0.5))
    proc_left = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    proc_right = np.random.uniform(-1, 1, int(sample_rate * 0.5))

    audiogram_left = Audiogram(levels=hl_left, frequencies=freqs)
    audiogram_right = Audiogram(levels=hl_right, frequencies=freqs)

    listener = Listener(audiogram_left=audiogram_left, audiogram_right=audiogram_right)
    score = haspi_v2_be(
        reference_left=ref_left,
        reference_right=ref_right,
        processed_left=proc_left + ref_left,
        processed_right=proc_right,
        sample_rate=sample_rate,
        listener=listener,
        level=100,
    )

    assert score == pytest.approx(
        expected_score, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "hl_left, hl_right, freqs, expected_score",
    [
        (
            np.array([25, 25, 25, 40, 65]),
            np.array([45, 35, 45, 60, 65]),
            np.array([250, 1000, 2000, 4000, 6000]),
            0.839975335323691,
        ),
        (
            np.array([25, 25, 21, 25, 25, 65]),
            np.array([45, 45, 21, 35, 45, 65]),
            np.array([250, 500, 700, 1000, 2000, 6000]),
            0.8380310721987008,
        ),
    ],
)
def test_haspi_v2_better_ear_non_standard(
    hl_left, hl_right, freqs, expected_score
) -> None:
    """Test that haspi works with non standard audiogram frequencies"""

    sample_rate = 16000

    np.random.seed(0)
    ref_left = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    ref_right = np.random.uniform(-1, 1, int(sample_rate * 0.5))
    proc_left = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    proc_right = np.random.uniform(-1, 1, int(sample_rate * 0.5))

    audiogram_left = Audiogram(levels=hl_left, frequencies=freqs)
    audiogram_right = Audiogram(levels=hl_right, frequencies=freqs)
    listener = Listener(audiogram_left=audiogram_left, audiogram_right=audiogram_right)

    score = haspi_v2_be(
        reference_left=ref_left,
        reference_right=ref_right,
        processed_left=proc_left + ref_left,
        processed_right=proc_right,
        sample_rate=sample_rate,
        listener=listener,
        level=100,
    )
    assert score == pytest.approx(
        expected_score, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
