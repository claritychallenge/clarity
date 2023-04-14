"""Tests for the build_scene_metadata.py script"""
from pathlib import Path

import numpy as np
import pytest

from recipes.cad1.task2.data_preparation.build_scene_metadata import (
    get_random_car_params,
    get_random_head_rotation,
    read_json,
    set_seed,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources" / "recipes" / "cad1" / "task2"


def test_set_seed():
    """Test that the seed is set correctly"""
    seed = 123
    set_seed(seed)
    np.testing.assert_array_almost_equal(np.random.rand(), 0.6964691855978616)


def test_get_random_head_rotation():
    """Test that the function returns a random item from the dictionary"""
    input_dict = {"1": "value1", "2": "value2", "3": "value3"}
    np.random.seed(42)
    random_item = get_random_head_rotation(input_dict)
    assert random_item == 1


def test_get_random_car_params():
    """Test that the function returns a dictionary with the expected keys"""
    np.random.seed(42)
    car_params = get_random_car_params()
    assert car_params == {
        "bump": {"btype": "bandpass", "cutoff_hz": [94, 187], "order": 1},
        "dip_high": {"btype": "highpass", "cutoff_hz": 128, "order": 2},
        "dip_low": {"btype": "lowpass", "cutoff_hz": 60, "order": 2},
        "engine_num_harmonics": 20,
        "gear": 5,
        "primary_filter": {"btype": "lowpass", "cutoff_hz": 16.507, "order": 1},
        "reference_level_db": 31.4,
        "rpm": 2060.4,
        "secondary_filter": {
            "btype": "lowpass",
            "cutoff_hz": 305.51040000000006,
            "order": 2,
        },
        "speed": 101.0,
    }


@pytest.mark.parametrize(
    "json_file, expected",
    [
        (
            RESOURCES / "test_build_scene_metadata.json_sample.json",
            {"key1": "value1", "key2": "value2"},
        ),
    ],
)
def test_read_json(json_file, expected):
    """Test that the function reads the json file correctly"""
    output = read_json(json_file.as_posix())

    assert output == expected


@pytest.mark.skip(reason="Not implemented yet")
def test_get_random_snr():
    """Test get_random_snr"""
    # get_random_snr()


@pytest.mark.skip(reason="Not implemented yet")
def test_run():
    """Test run"""
    # run()
