"""Module to test the ConvTasNet recipe."""

import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.io import wavfile

from recipes.cad2.task1.ConvTasNet.local.tasnet import ConvTasNet
from recipes.cad2.task1.ConvTasNet.train import (
    create_callbacks,
    create_datasets_and_loaders,
    create_model_and_optimizer,
    create_trainer,
    get_loss_func,
    save_best_model,
)


@pytest.fixture
def mock_conf():
    return {
        "data": {
            "root": "data/root",
            "mix_background": True,
            "sample_rate": 16000,
            "segment": 0.5,
            "samples_per_track": 1,
        },
        "convtasnet": {
            "N": 256,
            "L": 20,
            "B": 256,
            "H": 512,
            "P": 3,
            "X": 10,
            "R": 4,
            "C": 2,
            "audio_channels": 2,
            "mask_nonlinear": "relu",
            "norm_type": "gLN",
            "causal": False,
        },
        "training": {
            "batch_size": 4,
            "num_workers": 2,
            "epochs": 1,
            "half_lr": True,
            "early_stop": True,
            "aggregate": 1,
        },
        "optim": {"lr": 0.001},
        "main_args": {"exp_dir": "exp/tmp"},
    }


def create_mock_dataset(
    root_dir,
    duration=1,
    sample_rate=16000,
):
    track_names = [
        "Actions - One Minute Smile",
        "Actions - South Of The Water",
        "A Classic Education - NightOwl",
    ]
    for track_name in track_names:
        length = duration * sample_rate
        waveform = np.random.randn(int(length))

        track_dir = os.path.join(root_dir, "train", track_name)
        os.makedirs(track_dir, exist_ok=True)
        for source in ["vocals", "drums", "bass", "other", "mixture"]:
            source_path = os.path.join(track_dir, f"{source}.wav")
            wavfile.write(source_path, sample_rate, waveform)


def test_create_datasets_and_loaders(mock_conf):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock dataset with files
        create_mock_dataset(temp_dir)
        # Update the data root directory in the configuration
        mock_conf["data"]["root"] = temp_dir

        train_loader, train_set, val_loader, val_set = create_datasets_and_loaders(
            mock_conf
        )

        # Assertions can be made based on the created loaders or datasets
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        assert len(train_set) > 0
        assert len(val_set) > 0


def test_create_model_and_optimizer(mock_conf):
    model, optimizer, scheduler = create_model_and_optimizer(mock_conf)
    assert isinstance(model, ConvTasNet)
    assert isinstance(optimizer, torch.optim.Adam)
    if mock_conf["training"]["half_lr"]:
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_create_trainer(mock_conf):
    callbacks = []
    trainer = create_trainer(mock_conf, callbacks)
    assert isinstance(trainer, pl.Trainer)


def test_get_loss_func():
    loss_func = get_loss_func()
    assert isinstance(loss_func, torch.nn.L1Loss)


def test_create_callbacks(mock_conf):
    exp_dir = mock_conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    callbacks, checkpoint = create_callbacks(mock_conf, exp_dir)
    assert isinstance(callbacks, list)
    assert isinstance(checkpoint, ModelCheckpoint)


def test_save_best_model(mock_conf):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock system, checkpoint, and train_set
        mock_system = MagicMock()
        mock_system.model.serialize.return_value = {}

        mock_checkpoint = MagicMock()
        mock_checkpoint.best_k_models = {
            "model_1": torch.Tensor([0.50]),
            "model_2": torch.Tensor([0.3]),
        }
        mock_checkpoint.best_model_path = os.path.join(temp_dir, "best_model.ckpt")

        # Create a dummy state_dict to save
        torch.save({"state_dict": {}}, mock_checkpoint.best_model_path)

        mock_train_set = MagicMock()
        mock_train_set.get_infos.return_value = {}

        # Call the function under test
        save_best_model(mock_system, mock_checkpoint, temp_dir, mock_train_set)

        # Verify that the best_k_models.json file is created
        best_k_models_path = os.path.join(temp_dir, "best_k_models.json")
        assert os.path.exists(best_k_models_path)

        # Verify that the best_model.pth file is created
        best_model_path = os.path.join(temp_dir, "best_model.pth")
        assert os.path.exists(best_model_path)

        # Load and check the contents of best_k_models.json
        with open(best_k_models_path, encoding="utf-8") as f:
            best_k_models = json.load(f)
            assert best_k_models == {
                "model_1": torch.Tensor([0.5]),
                "model_2": torch.Tensor([0.3]),
            }
