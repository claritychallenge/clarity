# Tests for prepare_cec1_data module

from unittest.mock import patch

import hydra
import numpy as np
import pytest

from clarity.utils.file_io import read_signal
from recipes.cec1.data_preparation import prepare_cec1_data


def not_tqdm(iterable):
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


@patch("recipes.cec1.data_preparation.prepare_cec1_data.tqdm", not_tqdm)
def test_prepare_data(tmp_path):
    """Test prepare_data function."""

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path=".../../../../../../recipes/cec1/data_preparation",
        job_name="test_cec1",
    )
    cfg = hydra.compose(
        config_name="data_config",
        overrides=[
            "root=.",
            "input_path=tests/test_data",
            "+datasets.test.metafile_path="
            "tests/test_data/metadata/scenes.cec1.test.json",
            f"+datasets.test.scene_folder={tmp_path}",
        ],
    )

    del cfg.datasets.train
    del cfg.datasets.dev

    prepare_cec1_data.run(cfg)

    expected_files = [
        ("interferer", 28371.805869146872),
        ("interferer_CH0", 117331.99145541647),
        ("interferer_CH1", 117331.98102872199),
        ("interferer_CH2", 114417.14049813793),
        ("interferer_CH3", 114282.84357450972),
        ("mixed_CH0", 119765.13373339969),
        ("mixed_CH1", 117495.8815120442),
        ("mixed_CH2", 114563.55586907637),
        ("mixed_CH3", 114434.21600541842),
        ("target", 3669.940668940544),
        ("target_CH0", 10415.23247436894),
        ("target_CH1", 3355.1884318235343),
        ("target_CH2", 3231.3030493172855),
        ("target_CH3", 3271.638333302505),
        ("target_anechoic", 3430.6408055864526),
    ]

    for stem, expected_sum in expected_files:
        filename = tmp_path / f"S06001_{stem}.wav"
        assert filename.exists()
        # Check that the output signal is correct
        signal = read_signal(filename)
        assert np.sum(np.abs(signal)) == pytest.approx(
            expected_sum, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
        )
