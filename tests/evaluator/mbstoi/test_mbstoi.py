"""Tests for mbstoi module"""

# mbstoi

import numpy as np
import pytest

from clarity.evaluator.mbstoi import mbstoi


@pytest.mark.parametrize(
    "sr_signal, expected_score",
    [
        (10000.0, 0.9275623756362125),  # At signal at operating sample rate
        (9000.0, 0.9487599218003981),  # Signal will need resampling
    ],
)
def test_mbstoi(sr_signal, expected_score) -> None:
    """Test for mbstoi function"""
    np.random.seed(0)
    sig_len = 8000
    sample_rate = 10000.0
    left_clean = 100 * np.random.random(size=sig_len)
    right_clean = left_clean.copy()
    right_clean[4:] = right_clean[:-4]
    left_noisy = left_clean + 30 * np.random.random(size=sig_len)
    right_noisy = right_clean + 30 * np.random.random(size=sig_len)

    mbstoi_val = mbstoi(
        left_ear_clean=left_clean,
        right_ear_clean=right_clean,
        left_ear_noisy=left_noisy,
        right_ear_noisy=right_noisy,
        sr_signal=sr_signal,  # signal sample rate
        sample_rate=sample_rate,  # operating sample rate
        fft_size_in_samples=64,
        n_third_octave_bands=5,
        centre_freq_first_third_octave_hz=500,
        dyn_range=60,
    )

    assert mbstoi_val == pytest.approx(
        expected_score, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
