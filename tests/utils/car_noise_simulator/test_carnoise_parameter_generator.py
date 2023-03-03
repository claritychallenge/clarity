"""Tests for the Car noise parameters generator"""
# pylint: disable=import-error

import numpy as np
import pytest

from clarity.utils.car_noise_simulator.carnoise_parameters_generator import (
    CarNoiseParameters,
)


@pytest.mark.parametrize(
    "random_flag,speed,expected_result",
    [
        (
            False,
            100,
            {
                "bump": {"btype": "bandpass", "cutoff_hz": [30, 60], "order": 1},
                "dip_high": {"btype": "highpass", "cutoff_hz": 300, "order": 2},
                "dip_low": {"btype": "lowpass", "cutoff_hz": 200, "order": 2},
                "gear": 6,
                "primary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 16.860000000000003,
                    "order": 1,
                },
                "rpm": 1680.0000000000002,
                "secondary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 280.0,
                    "order": 2,
                },
                "speed": 100.0,
            },
        ),
        (
            True,
            100,
            {
                "bump": {"btype": "bandpass", "cutoff_hz": [91, 168], "order": 1},
                "dip_high": {"btype": "highpass", "cutoff_hz": 220, "order": 1},
                "dip_low": {"btype": "lowpass", "cutoff_hz": 90, "order": 1},
                "gear": 5,
                "primary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 18.191940000000006,
                    "order": 1,
                },
                "rpm": 2040.0,
                "secondary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 277.76000000000005,
                    "order": 2,
                },
                "speed": 100.0,
            },
        ),
        (
            True,
            70,
            {
                "bump": {"btype": "bandpass", "cutoff_hz": [34, 66], "order": 1},
                "dip_high": {"btype": "highpass", "cutoff_hz": 220, "order": 1},
                "dip_low": {"btype": "lowpass", "cutoff_hz": 90, "order": 1},
                "gear": 4,
                "primary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 12.685320000000003,
                    "order": 1,
                },
                "rpm": 1890.0,
                "secondary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 276.22400000000005,
                    "order": 2,
                },
                "speed": 70.0,
            },
        ),
    ],
)
def test_gen_parameters(random_flag, speed, expected_result):
    """Test the generation of parameters with and without randomisation."""
    np.random.seed(42)
    car_noise_parameters = CarNoiseParameters(random_flag=random_flag)
    parameters = car_noise_parameters.gen_parameters(speed_kph=speed)

    assert parameters == expected_result


@pytest.mark.parametrize("speed", [40, 130])
def test_gen_parameters_invalid_speed(speed):
    """Test speed values out of the boundaries."""
    car_noise_parameters = CarNoiseParameters(random_flag=True)
    with pytest.raises(ValueError):
        car_noise_parameters.gen_parameters(speed)
