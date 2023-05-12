"""Tests for cec1 e009 test module"""

from unittest.mock import patch

import hydra
import numpy as np
import pytest
import torch

from clarity.enhancer.dnn.mc_conv_tasnet import ConvTasNet
from clarity.enhancer.dsp.filter import AudiometricFIR
from clarity.utils.file_io import read_signal
from recipes.cec1.e009_sheffield.test import run


def not_tqdm(iterable, desc):  # pylint: disable=unused-argument
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


@patch("recipes.cec1.e009_sheffield.test.tqdm", not_tqdm)
def test_run(tmp_path):
    """Test for the run function."""
    np.random.seed(0)
    torch.manual_seed(0)

    expected_output_file = "enhanced_L0001/S06001_L0001_HA-output.wav"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/cec1/e009_sheffield", job_name="test_cec1_e009"
    )
    hydra_cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.cec1_root=tests/test_data/recipes/cec1/e009_sheffield",
            f"path.exp_folder={tmp_path}",
            # Disable multiprocessing for testing (faster)
            "test_loader.num_workers=0",
        ],
    )

    # Call run with all the torch processing mocked out
    with patch.object(ConvTasNet, "load_state_dict") as mock_1:
        with patch.object(torch, "load") as mock_2:
            with patch.object(torch.nn.parallel, "DataParallel") as mock_3:
                with patch.object(AudiometricFIR, "load_state_dict") as mock_4:
                    run(hydra_cfg)

    # Check the call counts are as expected
    assert mock_1.call_count == 2
    assert mock_2.call_count == 4
    assert mock_3.call_count == 4
    assert mock_4.call_count == 2

    # Check out
    assert (tmp_path / expected_output_file).exists()
    signal = read_signal(tmp_path / expected_output_file)
    assert signal.shape == (259200, 2)
    # Tolerances below slightly relaxed from the 1e-7 default
    assert np.sum(np.abs(signal)) == pytest.approx(
        4331.347137451172, rel=1e-6, abs=1e-6
    )
