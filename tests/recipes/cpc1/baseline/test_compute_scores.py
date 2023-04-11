"""Tests for cpc1 compute_scores module"""

import json
from pathlib import Path

import hydra
import pytest

from recipes.cpc1.baseline.compute_scores import run


@pytest.fixture()
def hydra_cfg():
    """Fixture for hydra config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/cpc1/baseline/", job_name="test_cpc1"
    )
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "train_path.root=tests/test_data/recipes/cpc1",
            (
                "train_path.scenes_file="
                "${train_path.root}/clarity_CPC1_data/metadata/CPC1.train.4.json"
            ),
            (
                "train_indep_path.scenes_file="
                "${train_path.root}/clarity_CPC1_data/metadata/CPC1.train_indep.4.json"
            ),
        ],
    )
    return cfg


def test_run(hydra_cfg):
    """Test run function."""

    expected_results = {
        "closed_set scores:": {
            "RMSE": 80.68942294220068,
            "Std": 8.103245804533998,
            "NCC": 0.2255349034109652,
            "KT": -0.3333333333333334,
        },
        "open_set scores:": {
            "RMSE": 62.32533032225741,
            "Std": 18.61907164264097,
            "NCC": 0.7820801834042259,
            "KT": 1.0,
        },
    }

    run(hydra_cfg)

    # Check output

    with open("results.json", encoding="utf-8") as f:
        results = json.load(f)

    # TODO: Find out what is causing results to be rounded to 4 dp
    # Need the abs=1e-4 because sometimes the results are being
    # printed rounded to 4 decimal places, and sometimes not.
    # Depends on the order the tests are run in.
    for test_set in ["closed_set scores:", "open_set scores:"]:
        for metric in ["RMSE", "Std", "NCC", "KT"]:
            assert results[test_set][metric] == pytest.approx(
                expected_results[test_set][metric], abs=1e-4
            )

    # Clean up

    Path("results.json").unlink()
