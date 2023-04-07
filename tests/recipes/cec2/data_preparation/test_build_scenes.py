"""Tests for cec2 build_scenes module."""
import json
from pathlib import Path
from unittest.mock import patch

import hydra

from recipes.cec2.data_preparation.build_scenes import (
    build_rooms_from_rpf,
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
        config_path="../../../../recipes/cec2/data_preparation/",
        job_name="test_cec2",
    )
    hydra_cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.root=.",
            "path.metadata_dir=tests/test_data/metadata",
            "scene_datasets.train.n_scenes=100",
        ],
    )

    del hydra_cfg.scene_datasets.dev
    del hydra_cfg.scene_datasets.demo

    instantiate_scenes(hydra_cfg)

    # Check that the output file exists...
    filename = Path("tests/test_data/metadata/scenes.train.json")
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
    filename.unlink()


@patch(
    "clarity.data.scene_builder_cec2.tqdm",
    not_tqdm,
)
def test_build_rooms_from_rpf(tmp_path):
    """Test build_rooms_from_rpf function."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(
        config_path="../../../../recipes/cec2/data_preparation/",
        job_name="test_cec2",
    )
    hydra_cfg = hydra.compose(
        config_name="config",
        overrides=[
            "path.root=tests/test_data",
            f"path.metadata_dir={tmp_path}",
            "room_datasets.train.n_rooms=1",
        ],
    )

    del hydra_cfg.room_datasets.dev
    del hydra_cfg.room_datasets.demo

    build_rooms_from_rpf(hydra_cfg)

    with open(tmp_path / "rooms.train.json", encoding="utf-8") as fp:
        rooms = json.load(fp)

    expected_room = [
        {
            "name": "R00001",
            "dimensions": "6.9933x3x3",
            "target": {
                "position": [-0.3, 2.4, 1.2],
                "view_vector": [0.071, 0.997, 0.0],
            },
            "listener": {
                "position": [-0.1, 5.2, 1.2],
                "view_vector": [-0.071, -0.997, 0.0],
            },
            "interferers": [
                {"position": [0.4, 4.0, 1.2]},
                {"position": [0.4, 3.1, 1.2]},
                {"position": [0.2, 3.7, 1.2]},
            ],
        }
    ]

    assert rooms == expected_room
