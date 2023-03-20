"""Tests for the Car noise signal generator"""
# pylint: disable=import-error

from pathlib import Path

import numpy as np

from clarity.utils.car_noise_simulator.carnoise_signal_generator import (
    CarNoiseSignalGenerator,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources" / "utils"


def test_car_noise_generation():
    """Test that the car noise generator returns the expected signal"""
    np.random.seed(42)
    carnoise_params = {
        "bump": {"btype": "bandpass", "cutoff_hz": [30, 60], "order": 1},
        "dip_high": {"btype": "highpass", "cutoff_hz": 300, "order": 2},
        "dip_low": {"btype": "lowpass", "cutoff_hz": 200, "order": 2},
        "engine_num_harmonics": 25,
        "gear": 6,
        "primary_filter": {
            "btype": "lowpass",
            "cutoff_hz": 16.860000000000003,
            "order": 1,
        },
        "reference_level_db": 30,
        "rpm": 1680.0000000000002,
        "secondary_filter": {
            "btype": "lowpass",
            "cutoff_hz": 280.0,
            "order": 2,
        },
        "speed": 100.0,
    }

    car_noise = CarNoiseSignalGenerator(
        sample_rate=16000, duration_secs=1, random_flag=True
    )
    car_noise_signal = car_noise.generate_car_noise(carnoise_params, 3, 0.5)

    assert car_noise_signal.shape == (4, 16000)
    expected = np.load(
        RESOURCES / "test_carnoise.signal_generator.npy", allow_pickle=True
    )
    np.testing.assert_array_almost_equal(car_noise_signal, expected)
