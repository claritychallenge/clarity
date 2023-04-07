"""Tests for the CPC2 evaluation functions."""

import math
import warnings
from csv import DictWriter
from pathlib import Path

import hydra
import numpy as np
import pytest

from clarity.utils.file_io import read_jsonl
from recipes.cpc2.baseline.evaluate import (
    compute_scores,
    evaluate,
    kt_score,
    ncc_score,
    rmse_score,
    std_err,
)


@pytest.mark.parametrize(
    "x, y, expected", [([1, 2, 3], [1, 2, 3], 0), ([0], [1], 1), ([1, 1, 1], [1], 0)]
)
def test_rmse_score_ok(x, y, expected):
    """Test the function rmse_score valid inputs"""
    assert rmse_score(np.array(x), np.array(y)) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "x, y, expected", [([1, 2, 3], [2, 3], ValueError), ([], [], ValueError)]
)
def test_rmse_score_error(x, y, expected):
    """Test the function rmse_score for invalid inputs"""
    with pytest.raises(expected):
        np.seterr(all="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # <--- suppress mean of empty slice warning
            result = rmse_score(np.array(x), np.array(y))
        if np.isnan(result):
            raise ValueError


@pytest.mark.parametrize(
    "x, y, expected", [([1, 2, 3], [1, 2, 3], 1), ([1, -1], [-1, 1], -1)]
)
def test_ncc_score_ok(x, y, expected):
    """Test the function ncc_score valid inputs"""
    assert ncc_score(np.array(x), np.array(y)) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "x, y, expected",
    [([1], [2], ValueError), ([], [], ValueError), ([1, 2, 3], [2, 3], ValueError)],
)
def test_ncc_score_error(x, y, expected):
    """Test the function ncc_score for invalid inputs"""
    with pytest.raises(expected):
        ncc_score(np.array(x), np.array(y))


@pytest.mark.parametrize(
    "x, y, expected", [([1, 2, 3], [1, 2, 3], 1), ([1, -1], [-1, 1], -1)]
)
def test_kt_score_ok(x, y, expected):
    """Test the function kt_score valid inputs"""
    assert kt_score(np.array(x), np.array(y)) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "x, y, expected",
    [([1], [2], ValueError), ([], [], ValueError), ([1, 2, 3], [2, 3], ValueError)],
)
def test_kt_score_error(x, y, expected):
    """Test the function kt_score for invalid inputs"""
    with pytest.raises(expected):
        result = kt_score(np.array(x), np.array(y))
        if np.isnan(result):
            raise ValueError


@pytest.mark.parametrize(
    "x, y, expected",
    [
        ([1, 2, 3], [1, 2, 3], 0),
        ([1, -1], [-1, 1], math.sqrt(2.0)),
        ([1], [2], 0),
        ([1, 2, 3], [11, 12, 13], 0),
    ],
)
def test_std_err_ok(x, y, expected):
    """Test the function std_err valid inputs"""
    assert std_err(np.array(x), np.array(y)) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "x, y, expected", [([], [], ValueError), ([1, 2, 3], [2, 3], ValueError)]
)
def test_std_err_error(x, y, expected):
    """Test the function std_err for invalid inputs"""
    with pytest.raises(expected):
        warnings.simplefilter("ignore")
        result = std_err(np.array(x), np.array(y))
        if np.isnan(result):
            raise ValueError


@pytest.mark.parametrize(
    "x, y, expected",
    [([], [], ValueError), ([1, 2, 3], [2, 3], ValueError), ([1], [2], ValueError)],
)
def test_compute_scores_error(x, y, expected):
    """Test the function compute_scores for invalid inputs"""
    with pytest.raises(expected):
        warnings.simplefilter("ignore")  # <--- suppress mean of empty slice warning
        compute_scores(np.array(x), np.array(y))


@pytest.mark.parametrize("x, y", [([1, 2, 3], [1, 2, 3]), ([1, 2], [1, 3])])
def test_compute_scores_ok(x, y):
    """Test the function compute_scores for valid inputs"""
    x = np.array(x)
    y = np.array(y)
    result = compute_scores(x, y)
    assert len(result) == 4  # RMSE, NCC, KT, Std and no others
    assert result["RMSE"] == rmse_score(x, y)
    assert result["NCC"] == ncc_score(x, y)
    assert result["KT"] == kt_score(x, y)
    assert result["Std"] == std_err(x, y)


@pytest.fixture()
def hydra_cfg():
    """Fixture for hydra config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/cpc2/baseline",
        job_name="test_cpc2",
    )
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.clarity_data_dir=tests/test_data/recipes/cpc2",
            "dataset=CEC1.train.sample",
        ],
    )
    return cfg


def test_evaluate(hydra_cfg, capsys):
    """Test evaluate function."""

    prediction_file = "CEC1.train.sample.predict.csv"
    score_file = "CEC1.train.sample.evaluate.jsonl"
    expected_output = (
        "{'RMSE': 30.256228825071368, 'Std': 4.209845712831399, "
        "'NCC': nan, 'KT': nan}\n"
    )
    test_data = [
        {"signal": "S08547_L0001_E001", "predicted": 0.8},
        {"signal": "S08564_L0001_E001", "predicted": 0.8},
        {"signal": "S08564_L0002_E002", "predicted": 0.8},
        {"signal": "S08564_L0003_E003", "predicted": 0.8},
    ]
    dict_keys = test_data[0].keys()

    with open(prediction_file, "w", encoding="utf-8") as fp:
        dict_writer = DictWriter(fp, fieldnames=dict_keys)
        dict_writer.writeheader()
        dict_writer.writerows(test_data)

    # Run evaluate, suppress warnings due to unrealist data
    warnings.simplefilter("ignore", category=RuntimeWarning)
    evaluate(hydra_cfg)

    captured = capsys.readouterr()
    assert captured.out == expected_output

    # Check scores
    scores = read_jsonl(score_file)
    assert scores[0]["RMSE"] == pytest.approx(30.2562, abs=1e-4)
    assert scores[0]["Std"] == pytest.approx(4.2098, abs=1e-4)
    assert np.isnan(scores[0]["NCC"])
    assert np.isnan(scores[0]["KT"])

    # Clean up
    Path(prediction_file).unlink()
    Path(score_file).unlink()
