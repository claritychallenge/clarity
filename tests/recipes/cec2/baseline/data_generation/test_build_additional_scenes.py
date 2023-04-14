# Tests for build_additional_scenes module

import json
from unittest.mock import patch

import hydra

from recipes.cec2.baseline.data_generation.build_additional_scenes import (
    instantiate_scenes,
)


def not_tqdm(iterable):
    """
    Replacement for tqdm that just passes back the iterable.

    Useful for silencing `tqdm` in tests.
    """
    return iterable


@patch(
    "clarity.data.scene_builder_cec2.tqdm",
    not_tqdm,
)
def test_instantiate_scenes(tmp_path):
    """Test instantiate_scenes function."""

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../../recipes/cec2/baseline/data_generation/",
        job_name="test_cec2",
    )
    hydra_cfg = hydra.compose(
        config_name="additional_data_config",
        overrides=[
            "path.root=.",
            "path.metadata_dir=tests/test_data/metadata",
            f"path.additional_data_file={tmp_path}/scenes.test.json",
            "scene_datasets.train.n_scenes=100",
        ],
    )

    instantiate_scenes(hydra_cfg)

    # Check that the output file exists...
    filename = tmp_path / "scenes.test.json"
    assert filename.exists()

    with open(filename, encoding="utf-8") as fp:
        scenes = json.load(fp)

    # ... then check there are the correct number of scenes
    # and that all scenes have the correct keys
    assert len(scenes) == 100
    scene_keys = {
        "dataset",
        "room",
        "scene",
        "target",
        "duration",
        "interferers",
        "SNR",
        "listener",
    }
    for scene in scenes:
        assert scene.keys() == scene_keys
