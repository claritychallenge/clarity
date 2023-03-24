"""Tests for the CPC2 predict functions."""

import numpy as np
import pandas as pd
import pytest

from recipes.cpc2.baseline.predict import LogisticModel, make_disjoint_train_set


# pylint: disable=redefined-outer-name
@pytest.fixture
def model():
    """Return a LogisticModel instance."""
    model = LogisticModel()
    model.fit(np.array([0, 1, 2, 3, 4]), np.array([0, 25, 50, 75, 100]))
    return model


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize(
    "model, value", [(model, 0.0), (model, 1.0), (model, 2.0)], indirect=["model"]
)
def test_logistic_model_symmetry(model: LogisticModel, value):
    """Test the LogisticModel is symmetric."""
    symmetric_value = 4 - value
    assert model.predict(value) + model.predict(symmetric_value) == pytest.approx(
        100.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize(
    "model, value",
    [(model, -100.0), (model, -200.0), (model, 100.0), (model, 200.0)],
    indirect=["model"],
)
def test_logistic_model_extremes(model, value):
    """Test the LogisticModel class ."""
    # logistic_model must asymptote to 0 and 100 for extreme values
    if value > 10:
        assert model.predict(value) == pytest.approx(
            100, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
        )
    elif value < -10:
        assert model.predict(value) == pytest.approx(
            0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
        )


@pytest.mark.parametrize(
    "data_1, data_2, expected",
    [
        (
            {"signal": "S100", "system": "E100", "listener": "L100"},
            {"signal": "S100", "system": "E100", "listener": "L100"},
            0,
        ),
        (
            {"signal": "S100", "system": "E100", "listener": "L100"},
            {"signal": "S100", "system": "E101", "listener": "L100"},
            0,
        ),
        (
            {"signal": "S100", "system": "E100", "listener": "L100"},
            {"signal": "S101", "system": "E101", "listener": "L101"},
            1,
        ),
    ],
)
def test_make_disjoint_train_set_empty(data_1, data_2, expected):
    """Test the make_disjoint_train_set function."""
    test_df1 = pd.DataFrame([data_1])
    test_df2 = pd.DataFrame([data_2])
    disjoint = make_disjoint_train_set(test_df1, test_df2)
    assert disjoint.shape[0] == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
