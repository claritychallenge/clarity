# Regression test
# Use scene builder to build random scene

import json
from pathlib import Path

from omegaconf import OmegaConf

from clarity.data.scene_builder_cec2 import RoomBuilder, SceneBuilder, set_random_seed


def test_CEC2_scene_builder(regtest):
    """Regression test for CEC2 scene builder"""
    cfg = OmegaConf.load("tests/test_data/configs/test_CEC2_scene_builder.yaml")
    cfg.path.root = "tests"
    set_random_seed(cfg.random_seed)
    dataset = "train"
    room_builder = RoomBuilder()

    room_file = Path(cfg.path.metadata_dir) / f"rooms.{dataset}.json"
    room_builder.load(room_file)
    scene_builder = SceneBuilder(
        rb=room_builder,
        scene_datasets=cfg.scene_datasets[dataset],
        target=cfg.target,
        interferer=cfg.interferer,
        snr_range=cfg.snr_range[dataset],
        listener=cfg.listener,
        shuffle_rooms=cfg.shuffle_rooms,
    )
    scene_builder.instantiate_scenes(dataset=dataset)

    print(len(scene_builder.scenes))

    with regtest:
        print(json.dumps(scene_builder.scenes, indent=4))
