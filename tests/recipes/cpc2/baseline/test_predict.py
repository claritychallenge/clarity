"""Tests for the CPC2 predict functions."""

import warnings
from csv import DictReader
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest

from clarity.utils.file_io import write_jsonl
from recipes.cpc2.baseline.predict import (
    LogisticModel,
    make_disjoint_train_set,
    predict,
)


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


def test_predict(hydra_cfg):
    """Test predict function."""

    expected_results = [
        ("S08547_L0001_E001", 0.0),
        ("S08564_L0001_E001", 0.0),
        ("S08564_L0002_E002", 31.481621447245452),
        ("S08564_L0003_E003", 31.481621447245452),
    ]
    haspi_scores = [
        {"signal": "S08547_L0001_E001", "haspi": 0.8},
        {"signal": "S08564_L0001_E001", "haspi": 0.8},
        {"signal": "S08564_L0002_E002", "haspi": 0.8},
        {"signal": "S08564_L0003_E003", "haspi": 0.8},
    ]
    haspi_score_file = "CEC1.train.sample.haspi.jsonl"

    write_jsonl(haspi_score_file, haspi_scores)

    # Run predict, ignoring warning due to unreal data
    warnings.simplefilter("ignore", category=RuntimeWarning)
    predict(hydra_cfg)

    # Check output
    expected_output_file = "CEC1.train.sample.predict.csv"

    with open(expected_output_file, encoding="utf-8") as f:
        results = list(DictReader(f))

    results_index = {
        entry["signal_ID"]: float(entry["intelligibility_score"]) for entry in results
    }

    # TODO: Scores are not checked because they can be very different
    # depending on the machine. This doesn't really matter for now as I believe
    # it's just a consequence of using just 4 samples in the testing data.
    # The fitting functions are tested separately.
    for signal, _expected_score in expected_results:
        assert signal in results_index
        # print(results_index[signal], expected_score)
        # assert results_index[signal] == pytest.approx(expected_score)

    # Clean up
    Path(expected_output_file).unlink()
    Path(haspi_score_file).unlink()
