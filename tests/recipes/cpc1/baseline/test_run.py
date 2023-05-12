"""Tests for cpc1 run module"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pytest
from scipy.io.wavfile import read

import recipes
from clarity.evaluator.msbg.msbg import Ear
from clarity.utils.audiogram import Audiogram, Listener
from clarity.utils.file_io import read_signal
from recipes.cpc1.baseline.run import listen, run, run_calculate_SI, run_HL_processing


def truncated_read_signal(
    filename: str | Path,
) -> np.ndarray:
    """Replacement for wavfile read signal function.

    Returns first 1 second of the signal
    """
    signal = read_signal(filename)
    return signal[0:44100, :]


def not_tqdm(iterable):
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


@pytest.fixture()
def hydra_cfg():
    """Fixture for hydra config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/cpc1/baseline/", job_name="test_cpc1"
    )
    cfg = hydra.compose(
        config_name="config", overrides=["train_path.root=tests/test_data/recipes/cpc1"]
    )
    return cfg


def test_listen(hydra_cfg):
    """Test listen function."""
    ear = Ear(**hydra_cfg.MSBGEar)

    cfs = np.array([250, 500, 1000, 2000, 4000, 8000])
    levels_1 = np.array([5, 5, 5, 5, 5, 5])
    levels_2 = np.array([10, 25, 80, 80, 80, 85])
    signal = np.random.rand(2000, 2)
    audiogram_mild = Audiogram(frequencies=cfs, levels=levels_1)
    audiogram_severe = Audiogram(frequencies=cfs, levels=levels_2)

    # Test asymmetric hearing loss in both orientations
    processed = listen(ear, signal, Listener(audiogram_mild, audiogram_severe))
    assert processed.shape == (2240, 2)
    processed = listen(ear, signal, Listener(audiogram_severe, audiogram_mild))
    assert processed.shape == (2240, 2)


@patch("recipes.cpc1.baseline.run.tqdm", not_tqdm)
def test_run_HL_processing(hydra_cfg, tmp_path):
    """Test run_HL_processing function."""
    np.random.seed(0)
    hydra_cfg.train_path.exp_folder = str(tmp_path / "exps/train")
    # Test the MSBG hearing loss stage
    # Use just 2 seconds of the signal to speed up the test
    with patch.object(
        recipes.cpc1.baseline.run,
        "read_signal",
        side_effect=truncated_read_signal,
    ) as mock_read_signal:
        run_HL_processing(hydra_cfg, hydra_cfg.train_path)
        assert mock_read_signal.call_count == 2

    expected_signals = [
        ("S08510_L0239_E001_HL-mixoutput.wav", 2799.13671875),
        ("S08510_L0239_E001_HL-output.wav", 734.46435546875),
        ("S08510_L0239_E001_HLddf-output.wav", 0.04365737363696098),
        ("S08510_flat0dB_HL-output.wav", 1.3086366653442383),
    ]
    for signal, expected_value in expected_signals:
        _fs, signal = read(tmp_path / "exps/train/eval_signals" / signal)
        assert np.sum(np.abs(signal)) == pytest.approx(
            expected_value, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
        )


@patch("recipes.cpc1.baseline.run.tqdm", not_tqdm)
def test_calculate_SI(hydra_cfg, tmp_path):
    """Test calculate_SI function."""
    np.random.seed(0)
    # copy test data to the working exp folder
    src_dir = "tests/test_data/recipes/cpc1/exps/train"
    dest_dir = tmp_path / "exps/train"
    hydra_cfg.train_path.exp_folder = str(dest_dir)
    shutil.copytree(src_dir, dest_dir)

    # run the MBSTOI-based SI model
    run_calculate_SI(hydra_cfg, hydra_cfg.train_path)

    # check the results that appear in sii.csv
    with open(dest_dir / "sii.csv", encoding="utf-8") as f:
        results = next(csv.DictReader(f))
        assert results["signal_ID"] == "S08510_L0239_E001"
        assert float(results["intelligibility_score"]) == pytest.approx(
            -0.01626095527838348
        )


@patch("recipes.cpc1.baseline.run.run_HL_processing")
@patch("recipes.cpc1.baseline.run.run_calculate_SI")
def test_run(mock_calculate_SI, mock_run_HL_processing, hydra_cfg):
    """Test run function."""
    run(hydra_cfg)
    # Just testing the number of calls to the functions is as expected
    assert mock_calculate_SI.call_count == 4
    assert mock_run_HL_processing.call_count == 4
