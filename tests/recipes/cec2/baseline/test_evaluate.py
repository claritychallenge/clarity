"""Tests for cec2 baseline evaluate module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pytest
from omegaconf import DictConfig
from scipy.io.wavfile import read

import recipes
from recipes.cec2.baseline.evaluate import read_csv_scores, run_calculate_SI


def truncated_read_signal(
    filename: str | Path,
) -> tuple[int, np.ndarray]:
    """Replacement for wavfile read signal function.

    Returns first 1 second of the signal
    """
    sample_rate, signal = read(filename)
    # Take 2 second sample from the start of the signal
    return sample_rate, signal[0:88200, :]


@pytest.fixture()
def hydra_cfg(tmp_path: Path):
    """Fixture for hydra config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path=".../../../../../../recipes/cec2/baseline", job_name="test_cec2"
    )
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.root=.",
            f"path.exp_folder={tmp_path}",
            "path.metadata_dir=tests/test_data/metadata",
            "path.scenes_listeners_file="
            "tests/test_data/metadata/scenes_listeners.1.json",
            "path.listeners_file=tests/test_data/metadata/listeners.json",
            "path.scenes_folder=tests/test_data/scenes",
        ],
    )
    return cfg


def test_read_csv_scores():
    """Test read_csv_scores function."""

    score_dict = read_csv_scores(
        Path("tests/test_data/recipes/cec2/baseline/cec2_si.csv")
    )

    assert score_dict["S06001_L0064"] == 0.29
    assert score_dict["S06002_L0064"] == 0.49


def not_tqdm(iterable):
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


@patch("recipes.cec2.baseline.evaluate.tqdm", not_tqdm)
def test_calulate_SI(tmp_path: Path, hydra_cfg: DictConfig):
    """Test evaluate function."""

    np.random.seed(0)

    Path(f"{tmp_path}/enhanced_signals").mkdir(parents=True, exist_ok=True)

    test_data = [
        "S06001_L0064_HA-output.wav",
    ]

    # set up test data
    for filename in test_data:
        from_file = (
            Path("tests/test_data/recipes/cec2/baseline/eval_signals") / filename
        )
        to_file = Path(f"{tmp_path}/enhanced_signals") / filename
        to_file.write_bytes(from_file.read_bytes())

    with patch.object(
        recipes.cec2.baseline.evaluate.wavfile,
        "read",
        side_effect=truncated_read_signal,
    ) as mock_read_signal:
        run_calculate_SI(hydra_cfg)
        assert mock_read_signal.call_count == 4

    si_dict = read_csv_scores(Path(f"{tmp_path}/si.csv"))
    si_unproc_dict = read_csv_scores(Path(f"{tmp_path}/si_unproc.csv"))

    assert si_dict["S06001_L0064"] == pytest.approx(
        0.745459815729705, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert si_unproc_dict["S06001_L0064"] == pytest.approx(
        0.981990966133383, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
