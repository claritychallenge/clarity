# Tests for render_additional_scenes module
#
# Very slow test - takes about a minute to render a scene

import hydra
import numpy as np
import pytest

from clarity.evaluator.msbg.msbg_utils import read_signal
from clarity.recipes.cec2.baseline.data_generation.render_additional_scenes import (
    render_scenes,
)


def test_render_scenes(tmp_path):
    """Test render_scenes function."""

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../../clarity/recipes/cec2/baseline/data_generation/",
        job_name="test_cec2",
    )
    hydra_cfg = hydra.compose(
        config_name="additional_data_config",
        overrides=[
            "path.root=tests/test_data",
            "path.metadata_dir=tests/test_data/metadata",
            (
                "scene_renderer.train.metadata.scene_definitions="
                "tests/test_data/metadata/scenes.test.json"
            ),
            f"scene_renderer.train.paths.scenes={tmp_path}",
        ],
    )

    render_scenes(hydra_cfg)

    expected_files = [
        ("hr", 271413.70622756885),
        ("interferer_CH0", 1304.3892517089844),
        ("interferer_CH1", 4294.033477783203),
        ("interferer_CH2", 4294.033477783203),
        ("interferer_CH3", 4294.033477783203),
        ("mix_CH0", 1812.4237365722656),
        ("mix_CH1", 5105.1387939453125),
        ("mix_CH2", 5105.1387939453125),
        ("mix_CH3", 5105.1387939453125),
        ("target_CH0", 703.9017333984375),
        ("target_CH1", 1316.4816284179688),
        ("target_CH2", 1316.4816284179688),
        ("target_CH3", 1316.4816284179688),
        ("target_anechoic_CH1", 8006.998596191406),
    ]

    for stem, expected_sum in expected_files:
        filename = tmp_path / f"S06001_{stem}.wav"
        assert filename.exists()
        # Check that the output signal is correct
        signal = read_signal(filename)
        assert np.sum(np.abs(signal)) == pytest.approx(expected_sum)
