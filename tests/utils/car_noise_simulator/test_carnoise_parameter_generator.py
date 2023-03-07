"""Tests for the Car noise parameters generator"""
# pylint: disable=import-error

import numpy as np
import pytest

from clarity.utils.car_noise_simulator.carnoise_parameters_generator import (
    CarNoiseParametersGenerator,
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
            },
        ),
        (
            True,
            100,
            {
                "bump": {"btype": "bandpass", "cutoff_hz": [80, 116], "order": 2},
                "dip_high": {"btype": "highpass", "cutoff_hz": 319, "order": 1},
                "dip_low": {"btype": "lowpass", "cutoff_hz": 130, "order": 1},
                "engine_num_harmonics": 38,
                "gear": 5,
                "primary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 15.410040000000004,
                    "order": 1,
                },
                "reference_level_db": 31.9,
                "rpm": 2040.0,
                "secondary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 281.68000000000006,
                    "order": 2,
                },
                "speed": 100.0,
            },
        ),
        (
            True,
            70,
            {
                "bump": {"btype": "bandpass", "cutoff_hz": [91, 168], "order": 1},
                "dip_high": {"btype": "highpass", "cutoff_hz": 220, "order": 1},
                "dip_low": {"btype": "lowpass", "cutoff_hz": 90, "order": 1},
                "engine_num_harmonics": 29,
                "gear": 4,
                "primary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 12.558720000000001,
                    "order": 1,
                },
                "reference_level_db": 30.6,
                "rpm": 1890.0,
                "secondary_filter": {
                    "btype": "lowpass",
                    "cutoff_hz": 233.984,
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
    car_noise_parameters = CarNoiseParametersGenerator(random_flag=random_flag)
    parameters = car_noise_parameters.gen_parameters(speed_kph=speed)

    assert parameters == expected_result


@pytest.mark.parametrize("speed", [40, 130])
def test_gen_parameters_invalid_speed(speed):
    """Test speed values out of the boundaries."""
    car_noise_parameters = CarNoiseParametersGenerator(random_flag=True)
    with pytest.raises(ValueError):
        car_noise_parameters.gen_parameters(speed)
