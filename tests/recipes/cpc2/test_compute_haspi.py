"""Tests for the haspi computation functions."""

import numpy as np
import pytest

from recipes.cpc2.baseline.compute_haspi import (
    parse_cec2_signal_name,
    set_seed_with_string,
)


@pytest.mark.parametrize(
    "signal_name, expected",
    [("S1_L1_E1_hr", ("S1", "L1", "E1_hr")), ("S1_L1_E2", ("S1", "L1", "E2"))],
)
def test_parse_cec2_signal_name_ok(signal_name, expected):
    """Test the parse_CEC2_signal_name function."""
    assert parse_cec2_signal_name(signal_name) == expected


@pytest.mark.parametrize(
    "signal_name, expected",
    [
        ("S1", ValueError),
        ("S1_L1", ValueError),
        ("___", ValueError),
        ("_X_X", ValueError),
    ],
)
def test_parse_cec2_signal_name_error(signal_name, expected):
    """Test the parse_CEC2_signal_name function for invalid inputs."""
    with pytest.raises(expected):
        parse_cec2_signal_name(signal_name)


@pytest.mark.parametrize("string_value", ["", "abc", "123", "abc123"])
def test_set_seed_with_string_ok(string_value):
    """Test the set_seed_with_string function."""
    set_seed_with_string(string_value)
    x = np.random.randint(0, 1000)
    set_seed_with_string(string_value)
    assert np.random.randint(0, 1000) == x
