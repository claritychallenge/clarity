"""Tests for icassp_2023 cec2 enhance module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pytest
from omegaconf import DictConfig

from clarity.utils.file_io import read_signal
from recipes.icassp_2023.baseline.enhance import enhance


@pytest.fixture()
def hydra_cfg(tmp_path: Path):
    """Fixture for hydra config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/icassp_2023/baseline",
        job_name="test_icassp_2023",
    )
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.root=tests/test_data",
            f"path.exp_folder={tmp_path}",
            "path.metadata_dir=tests/test_data/metadata",
            "path.scenes_listeners_file=${path.metadata_dir}/scenes_listeners.1.json",
            "path.scenes_folder=${path.root}/scenes",
        ],
    )
    return cfg


def not_tqdm(iterable):
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


@patch("recipes.icassp_2023.baseline.enhance.tqdm", not_tqdm)
def test_enhance(hydra_cfg: DictConfig) -> None:
    """Test run_HL_processing function."""
    np.random.seed(0)

    # Run the enhance function
    enhance(hydra_cfg)

    # Check that the output signal is correct
    filename = Path("enhanced_signals/S06001_L0064_enhanced.wav")
    assert filename.exists()
    signal = read_signal(filename)
    assert np.sum(np.abs(signal)) == pytest.approx(
        125253.92190551758, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    # Note, enhance.py writes results to where this test is run from,
    # so we need to clean up.
    filename.unlink()
    filename.parent.rmdir()
