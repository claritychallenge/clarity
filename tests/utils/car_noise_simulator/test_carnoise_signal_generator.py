"""Tests for the Car noise signal generator"""
# pylint: disable=import-error

from pathlib import Path

import numpy as np

from clarity.utils.car_noise_simulator.carnoise_signal_generator import (
    CarNoiseGenerator,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"


def test_car_noise_generation():
    """Test that the car noise generator returns the expected signal"""
    np.random.seed(42)
    carnoise_params = {
        "speed": 86.0,
        "gear": 5,
        "rpm": 1754.4,
        "primary_filter": {"order": 1, "btype": "lowpass", "cutoff_hz": 14.9596},
        "secondary_filter": {
            "order": 2,
            "btype": "lowpass",
            "cutoff_hz": 271.75680000000006,
        },
        "bump": {"order": 2, "btype": "bandpass", "cutoff_hz": [66, 114]},
        "dip_low": {"order": 2, "btype": "lowpass", "cutoff_hz": 170},
        "dip_high": {"order": 2, "btype": "highpass", "cutoff_hz": 475},
    }

    car_noise = CarNoiseGenerator(sample_rate=16000, duration_secs=1, random_flag=True)
    car_noise_signal = car_noise.generate_car_noise(carnoise_params, 3, 0.5)

    assert car_noise_signal.shape == (4, 16000)
    expected = np.load(
        RESOURCES / "test_carnoise.signal_generator.npy", allow_pickle=True
    )
    np.testing.assert_array_almost_equal(car_noise_signal, expected)
