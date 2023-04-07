"""Tests for the haspi computation functions."""


from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pytest
from omegaconf import DictConfig

import recipes
from clarity.utils.file_io import read_jsonl
from recipes.cpc2.baseline.compute_haspi import (
    parse_cec2_signal_name,
    run_calculate_haspi,
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


@pytest.fixture()
def hydra_cfg():
    """Fixture for hydra config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/cpc2/baseline",
        job_name="test_cpc2",
    )
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.clarity_data_dir=tests/test_data/recipes/cpc2",
            "dataset=CEC1.train.sample",
        ],
    )
    return cfg


def not_tqdm(iterable):
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


@patch("recipes.cpc2.baseline.compute_haspi.tqdm", not_tqdm)
def test_run_calculate_haspi(hydra_cfg: DictConfig):
    # Mocking the slow haspi calculation

    expected_scores = [
        {"signal": "S08547_L0001_E001", "haspi": 0.8},
        {"signal": "S08564_L0001_E001", "haspi": 0.8},
        {"signal": "S08564_L0002_E002", "haspi": 0.8},
        {"signal": "S08564_L0003_E003", "haspi": 0.8},
    ]
    expected_output_file = "CEC1.train.sample.haspi.jsonl"

    with patch.object(
        recipes.cpc2.baseline.compute_haspi,
        "haspi_v2_be",
        return_value=0.8,
    ) as mock_haspi:
        run_calculate_haspi(hydra_cfg)
        assert mock_haspi.call_count == 4

    # Check that the output scores are correct
    assert Path(expected_output_file).exists()
    haspi_scores = read_jsonl(expected_output_file)
    assert haspi_scores == expected_scores

    # Clean up
    Path(expected_output_file).unlink()
