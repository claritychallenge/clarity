"""Tests for icassp_2023 cec2 evaluate module"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pytest
from omegaconf import DictConfig

import recipes
from clarity.utils.file_io import read_signal
from recipes.icassp_2023.baseline.evaluate import run_calculate_si


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


@patch("recipes.icassp_2023.baseline.evaluate.tqdm", not_tqdm)
def test_evaluate(hydra_cfg: DictConfig):
    """Test evaluate function."""
    np.random.seed(0)

    Path("enhanced_signals").mkdir(parents=True, exist_ok=True)

    # set up test data
    from_file = Path(
        "tests/test_data/recipes/cec2/baseline/eval_signals/S06001_L0064_HA-output.wav"
    )
    to_file = Path("enhanced_signals/S06001_L0064_enhanced.wav")
    to_file.write_bytes(from_file.read_bytes())

    # Mocking the slow hasqi and haspi calculations
    with patch.object(
        recipes.icassp_2023.baseline.evaluate,
        "hasqi_v2_better_ear",
        return_value=0.5,
    ) as mock_hasqi:
        with patch.object(
            recipes.icassp_2023.baseline.evaluate,
            "haspi_v2_be",
            return_value=0.8,
        ) as mock_haspi:
            run_calculate_si(hydra_cfg)
            assert mock_haspi.call_count == 1
            assert mock_hasqi.call_count == 1

    # Check that the output scores are correct
    with open("scores.csv", encoding="utf-8") as f:
        results = next(csv.DictReader(f))
        assert results == {
            "scene": "S06001",
            "listener": "L0064",
            "combined": "0.65",
            "haspi": "0.8",
            "hasqi": "0.5",
        }

    # Check that the output signal is correct
    expected_signals = [
        ("amplified_signals/S06001_L0064_HA-output.wav", 518635.2062121812),
        ("enhanced_signals/S06001_L0064_enhanced.wav", 78939.73132324219),
    ]

    for filename, expected_sum in expected_signals:
        assert Path(filename).exists()
        # Check that the output signal is correct
        signal = read_signal(filename)
        Path(filename).unlink()
        assert np.sum(np.abs(signal)) == pytest.approx(
            expected_sum, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
        )

    # Clean up
    Path("scores.csv").unlink()
    Path("amplified_signals").rmdir()
    Path("enhanced_signals").rmdir()
