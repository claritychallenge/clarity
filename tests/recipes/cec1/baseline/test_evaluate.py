from __future__ import annotations

import pathlib
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pytest
from numpy import ndarray
from omegaconf import DictConfig

import recipes
from clarity.utils.file_io import read_signal

# pylint: disable=import-error, no-name-in-module, no-member
from recipes.cec1.baseline.evaluate import run_calculate_SI, run_HL_processing

# listen, run_calculate_SI,


def not_tqdm(iterable):
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


def truncated_read_signal(
    filename: str | Path,
    offset: int = 0,
    n_samples: int = -1,
    n_channels: int = 0,
    offset_is_samples: bool = False,
) -> ndarray:
    """Replacement for read signal function.

    Returns first 1 second of the signal
    """
    n_samples = 44100  # <-- take just the first 1 second of the signal
    return read_signal(
        filename,
        offset=offset,
        n_channels=n_channels,
        n_samples=n_samples,
        offset_is_samples=offset_is_samples,
    )


@pytest.fixture()
def hydra_cfg(tmp_path: Path):
    """Fixture for hydra config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path=".../../../../../../recipes/cec1/baseline", job_name="test_cec1"
    )
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.root=.",
            f"path.exp_folder={tmp_path}",
            "path.scenes_listeners_file="
            "tests/test_data/metadata/scenes_listeners.1.json",
            "path.listeners_file=tests/test_data/metadata/listeners.json",
            "path.scenes_folder=tests/test_data/scenes",
            "path.enhanced_signals=enhanced_signals",
        ],
    )
    return cfg


@patch("recipes.cec1.baseline.evaluate.tqdm", not_tqdm)
def test_run_HL_processing(tmp_path: Path, hydra_cfg: DictConfig) -> None:
    """Test run_HL_processing function."""
    np.random.seed(0)

    with patch.object(
        recipes.cec1.baseline.evaluate,
        "read_signal",
        side_effect=truncated_read_signal,
    ) as mock_read_signal:
        run_HL_processing(hydra_cfg)
        assert mock_read_signal.call_count == 2

    # check if output files exist and contents are correct
    expected_files = [
        ("S06001_L0064_HL-mixoutput.wav", 2692.4437607042),
        ("S06001_L0064_HL-output.wav", 2692.4437607042),
        ("S06001_L0064_HLddf-output.wav", 0.0439577816261102),
        ("S06001_flat0dB_HL-output.wav", 1.308636470194449),
    ]

    for filename, sig_sum in expected_files:
        assert (pathlib.Path(f"{tmp_path}/eval_signals/{filename}")).exists()
        x = read_signal(f"{tmp_path}/eval_signals/{filename}")
        assert np.sum(np.abs(x)) == pytest.approx(
            sig_sum, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
        )


@patch("recipes.cec1.baseline.evaluate.tqdm", not_tqdm)
def test_run_calculate_SI(tmp_path: Path, hydra_cfg: DictConfig):
    """Test run_calculate_SI function."""

    np.random.seed(0)

    Path(f"{tmp_path}/eval_signals").mkdir(parents=True, exist_ok=True)

    test_data = [
        "S06001_L0064_HLddf-output.wav",
        "S06001_L0064_HL-output.wav",
    ]

    # set up test data
    for filename in test_data:
        from_file = (
            Path("tests/test_data/recipes/cec1/baseline/eval_signals") / filename
        )
        to_file = Path(f"{tmp_path}/eval_signals") / filename
        to_file.write_bytes(from_file.read_bytes())

    with patch.object(
        recipes.cec1.baseline.evaluate,
        "read_signal",
        side_effect=truncated_read_signal,
    ) as mock_read_signal:
        run_calculate_SI(hydra_cfg)
        assert mock_read_signal.call_count == 3

    with open(f"{tmp_path}/sii.csv", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert lines[1][:24] == "S06001,L0064,-0.03769851"
