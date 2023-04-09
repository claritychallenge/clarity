"""Tests for cpc1 e032_sheffield prepare_data module."""

# pylint: disable=all
# This file was causing pylint to crash, so I disabled it for now. !!

from pathlib import Path
from unittest.mock import patch

import hydra

from recipes.cpc1.e032_sheffield.prepare_data import run


def not_tqdm(iterable):
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


@patch("recipes.cpc1.e032_sheffield.prepare_data.tqdm", not_tqdm)
def test_run(tmp_path):
    """Test for the run function."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/cpc1/e032_sheffield", job_name="test_cpc1_e032"
    )
    hydra_cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.root=tests/test_data/recipes/cpc1/e032_sheffield",
            f"path.exp_folder={tmp_path}",
        ],
    )

    root = Path("tests/test_data/recipes/cpc1/e032_sheffield/")
    expected_files = [
        "clarity_CPC1_data_test/clarity_data/HA_outputs/test/"
        "S08520_L0216_E001_HL-output.wav",
        "clarity_CPC1_data_train/clarity_data/HA_outputs/train/"
        "S08510_L0239_E001_HL-output.wav",
    ]
    run(hydra_cfg)

    for expected_file in expected_files:
        assert (root / expected_file).exists()
        (root / expected_file).unlink()
