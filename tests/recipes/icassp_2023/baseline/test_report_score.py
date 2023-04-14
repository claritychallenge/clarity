"""Tests for icassp_2023 cec2 report score module"""
from __future__ import annotations

from csv import DictWriter
from pathlib import Path

import hydra
import pytest
from omegaconf import DictConfig

from recipes.icassp_2023.baseline.report_score import report_score


@pytest.fixture()
def hydra_cfg(tmp_path: Path):
    """Fixture for hydra config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/icassp_2023/baseline",
        job_name="test_icassp_2023",
    )
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.root=tests/test_data",
            "path.scenes_listeners_file=${path.metadata_dir}/scenes_listeners.1.json",
        ],
    )
    return cfg


@pytest.mark.parametrize(
    "test_data, expected_output",
    [
        (  # test case 1 - bad data
            [
                {
                    "scene": "S06001",
                    "listener": "L0065",
                    "haspi": 0.8,
                    "hasqi": 0.5,
                    "combined": 0.65,
                }
            ],
            (
                "The following results were not found:\n"
                "    scene listener\n0  S06001    L0064\n"
            ),
        ),
        (  # test case 2 - good data
            [
                {
                    "scene": "S06001",
                    "listener": "L0064",
                    "haspi": 0.8,
                    "hasqi": 0.5,
                    "combined": 0.65,
                }
            ],
            (
                "Scores based on 1 scenes.\nhaspi       0.80\n"
                "hasqi       0.50\ncombined    0.65\ndtype: float64\n"
            ),
        ),
    ],
)
def test_report_score(hydra_cfg: DictConfig, capsys, test_data, expected_output):
    """Test report_score function."""

    dict_keys = test_data[0].keys()
    score_file = Path("scores.csv")
    with score_file.open("w", encoding="utf-8") as fp:
        dict_writer = DictWriter(fp, fieldnames=dict_keys)
        dict_writer.writeheader()
        dict_writer.writerows(test_data)

    report_score(hydra_cfg)
    captured = capsys.readouterr()
    assert captured.out == expected_output

    score_file.unlink()
