import numpy as np
import pytest

from clarity.evaluator.haaqi import compute_haaqi, haaqi_v1


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
    assert score == pytest.approx(0.109534910970557, rel=1e-7)


def test_compute_haaqi():
    np.random.seed(42)

    fs = 16000
    enh_signal = np.random.uniform(-1, 1, fs * 10)
    ref_signal = np.random.uniform(-1, 1, fs * 10)

    audiogram = np.array([10, 20, 30, 40, 50, 60])
    audiogram_frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])

    # Compute HAAQI score
    score = compute_haaqi(
        processed_signal=enh_signal,
        reference_signal=ref_signal,
        audiogram=audiogram,
        audiogram_frequencies=audiogram_frequencies,
        sample_rate=fs,
    )

    # Check that the score is a float between 0 and 1
    assert score == pytest.approx(0.117063418, rel=1e-7)
