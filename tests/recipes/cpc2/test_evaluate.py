"""Tests for the CPC2 evaluation functions."""

import math

import numpy as np
import pytest

from recipes.cpc2.baseline.evaluate import (
    compute_scores,
    kt_score,
    ncc_score,
    rmse_score,
    std_err,
)


def test_rmse_score():
    """Test the function rmse_score"""
    some_array = np.array([1, 2, 3])
    empty_array = np.array([])

    assert rmse_score(some_array, some_array) == 0.0
    assert rmse_score(np.array([0]), np.array([1])) == 1.0
    assert np.isnan(rmse_score(np.array([]), np.array([])))

    # Should raise a ValueError exception is the arrays are not the same size
    with pytest.raises(Exception):
        rmse_score(empty_array, some_array)


def test_ncc_score():
    """Test the function ncc_score"""

    # Correlation with self should be 1
    assert ncc_score(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)

    # Correlation with self should be 1
    assert ncc_score(np.array([1, -1]), np.array([-1, 1])) == pytest.approx(-1.0)

    with pytest.raises(Exception):
        # correlation requires at least 2 data points
        ncc_score(np.array([0]), np.array([1]))

    with pytest.raises(Exception):
        # correlation requires at least 2 data points
        ncc_score(np.array([]), np.array([]))

    with pytest.raises(Exception):
        # Requires same size arrays
        ncc_score(np.array([1, 2]), np.array([1, 2, 3]))


def test_kt_score():
    """Test the function kt_score"""

    # Correlation with self should be 1
    assert kt_score(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)

    # Correlation with self should be 1
    assert kt_score(np.array([1, -1]), np.array([-1, 1])) == pytest.approx(-1.0)

    # Requires at least 2 data points
    assert np.isnan(kt_score(np.array([0]), np.array([1])))
    assert np.isnan(kt_score(np.array([]), np.array([])))

    with pytest.raises(Exception):
        # Requires same size arrays
        kt_score(np.array([1, 2]), np.array([1, 2, 3]))


def test_std_err():
    """Test the function std_err"""

    # Correlation with self should be 1
    assert std_err(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0

    # Correlation with self should be 1
    assert std_err(np.array([1, -1]), np.array([-1, 1])) == pytest.approx(
        math.sqrt(2.0)
    )

    # Should be 0 for sequences of length 1
    assert std_err(np.array([0]), np.array([1])) == pytest.approx(0)

    # Should be 0 for sequences with constant differences
    assert std_err(np.array([1, 2, 3]), np.array([11, 12, 13])) == pytest.approx(0)

    # Requires at least 1 data points
    assert np.isnan(std_err(np.array([]), np.array([])))

    with pytest.raises(Exception):
        # Requires same size arrays
        std_err(np.array([1, 2]), np.array([1, 2, 3]))


def test_compute_scores():
    """Test the function compute_scores"""

    with pytest.raises(Exception):
        # Requires same size arrays
        compute_scores(np.array([1, 2]), np.array([1, 2, 3]))

    with pytest.raises(Exception):
        # Requires at least 2 data points
        compute_scores(np.array([1]), np.array([1]))

    x = np.array([1, 2, 3])
    result = compute_scores(x, x)
    assert result["RMSE"] == rmse_score(x, x)
    assert result["NCC"] == ncc_score(x, x)
    assert result["KT"] == kt_score(x, x)
    assert result["Std"] == std_err(x, x)
