"""Tests for the CPC2 evaluation functions."""

import math

import numpy as np
import pytest

from clarity.recipes.cpc2.baseline.evaluate import (
    compute_scores,
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


@pytest.mark.skip(reason="Not implemented yet")
def test_evaluate():
    """Test evaluate function."""
