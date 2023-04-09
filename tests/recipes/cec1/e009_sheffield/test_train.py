"""Tests for cec1 e009 train module"""

import logging
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch

from recipes.cec1.e009_sheffield.train import train_amp, train_den


@pytest.mark.slow
def test_run(tmp_path):
    """Test for the run function."""
    np.random.seed(0)
    torch.manual_seed(0)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/cec1/e009_sheffield", job_name="test_cec1_e009"
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    hydra_cfg = hydra.compose(
        config_name="config",
        # Override settings to make a fast training test
        overrides=[
            "path.cec1_root=tests/test_data/recipes/cec1/e009_sheffield",
            f"path.exp_folder={tmp_path}",
            # Disable multiprocessing for testing (faster)
            "train_loader.num_workers=0",
            "dev_loader.num_workers=0",
            "test_loader.num_workers=0",
            "train_loader.batch_size=1",
            "train_dataset.wav_sample_len=1.0",
            "den_trainer.epochs=1",
            "amp_trainer.epochs=1",
            "fir.nfir=32",
            "mc_conv_tasnet.H=64",
            "mc_conv_tasnet.B=32",
            # The validation sanity check step is slow, so disable it
            "amp_trainer.num_sanity_val_steps=0",
        ],
    )

    train_den(hydra_cfg, ear="left")
    hydra_cfg.downsample_factor = 40
    train_amp(hydra_cfg, ear="left")

    expected_files = [
        "left_amp/checkpoints/epoch=0-step=1.ckpt",
        "left_amp/best_k_models.json",
        "left_amp/best_model.pth",
        "left_den/checkpoints/epoch=0-step=1.ckpt",
        "left_den/best_k_models.json",
        "left_den/best_model.pth",
    ]
    for filename in expected_files:
        assert (Path(tmp_path) / filename).exists()
