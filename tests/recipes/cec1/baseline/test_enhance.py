from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pytest
from omegaconf import DictConfig

from clarity.utils.file_io import read_signal, write_signal

# pylint: disable=import-error, no-name-in-module, no-member
from recipes.cec1.baseline.enhance import enhance


def not_tqdm(iterable):
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


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
def test_enhance(tmp_path: Path, hydra_cfg: DictConfig, mocker) -> None:
    """Test run_HL_processing function."""
    np.random.seed(0)

    # Mock the subprocess.run function as OpenMHA is not installed
    m = mocker.patch("clarity.enhancer.gha.gha_interface.subprocess.run")

    # Write a dummy output for the openMHA so that enhance can run
    out_dir = tmp_path / "enhanced_signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile_name = out_dir / "S06001_L0064_HA-output.wav"

    write_signal(
        filename=outfile_name,
        signal=np.array([[-0.1, 0.1, -0.1, 0.1], [-0.2, 0.2, -0.2, 0.2]]).T,
        sample_rate=44100,
        floating_point=False,
    )

    # Run the enhance function
    enhance(hydra_cfg)

    # Check that the output signal is correct
    signal = read_signal(outfile_name)
    assert np.sum(np.abs(signal)) == pytest.approx(
        1.1998291015625, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    assert m.call_count == 1
