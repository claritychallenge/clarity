from __future__ import annotations

import pathlib
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pytest
from numpy import ndarray

import recipes
from clarity.evaluator.msbg.msbg_utils import read_signal

# pylint: disable=import-error, no-name-in-module
from recipes.cec1.baseline.evaluate import run_HL_processing

# listen, run_calculate_SI,


@pytest.mark.skip(reason="Not implemented yet")
def test_listen():
    """Test listen function."""


def not_tqdm(iterable):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


def truncated_read_signal(
    filename: str | Path,
    offset: int = 0,
    nsamples: int = -1,
    nchannels: int = 0,
    offset_is_samples: bool = False,
) -> ndarray:
    # print(f"In mocked read - truncating signal {filename} to 1 second")
    signal = read_signal(filename, offset, nsamples, nchannels, offset_is_samples)
    # Take 1 second sample from the middle of the signal
    middle = signal.shape[0] // 2
    return signal[middle - 22050 : middle + 22050, :]


@patch("recipes.cec1.baseline.evaluate.tqdm", not_tqdm)
def test_run_HL_processing(tmp_path):
    """Test run_HL_processing function."""
    hydra.initialize(config_path=".", job_name="test_cec1")
    np.random.seed(0)
    cfg = hydra.compose(
        config_name="config", overrides=["path.root=.", f"path.exp_folder={tmp_path}"]
    )

    with patch.object(
        # clarity.evaluator.msbg.msbg_utils,
        recipes.cec1.baseline.evaluate,
        "read_signal",
        side_effect=truncated_read_signal,
    ) as mock_read_signal:
        run_HL_processing(cfg)
        assert mock_read_signal.call_count == 2

    # check if output files exist and contents are correct
    expected_files = [
        ("S06001_L0064_HL-mixoutput.wav", 2267.102214803097),
        ("S06001_L0064_HL-output.wav", 2267.102214803097),
        ("S06001_L0064_HLddf-output.wav", 0.0439577816261102),
        ("S06001_flat0dB_HL-output.wav", 1.308636470194449),
    ]

    for filename, sig_sum in expected_files:
        assert (pathlib.Path(f"{tmp_path}/eval_signals/{filename}")).exists()
        x = read_signal(f"{tmp_path}/eval_signals/{filename}")
        assert np.sum(np.abs(x)) == pytest.approx(sig_sum)


@pytest.mark.skip(reason="Not implemented yet")
def test_run_calculate_SI():
    """Test run_calculate_SI function."""


# Mock the subprocess.run function as OpenMHA is not installed
# m = mocker.patch("clarity.enhancer.gha.gha_interface.subprocess.run")
