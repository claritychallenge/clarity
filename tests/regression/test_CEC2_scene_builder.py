# Regression test
# Use scene builder to build random scene

import json
import os

from omegaconf import OmegaConf

from clarity.data.scene_builder_cec2 import RoomBuilder, SceneBuilder, set_random_seed


def test_CEC2_scene_builder(regtest):
    cfg = OmegaConf.load("tests/test_data/configs/test_CEC2_scene_builder.yaml")
    cfg.path.root = "tests"
    set_random_seed(cfg.random_seed)
    dataset = "train"
    rb = RoomBuilder()

    room_file = os.path.join(cfg.path.metadata_dir, f"rooms.{dataset}.json")
    rb.load(room_file)
    sb = SceneBuilder(
        rb=rb,
        scene_datasets=cfg.scene_datasets[dataset],
        target=cfg.target,
        interferer=cfg.interferer,
        snr_range=cfg.snr_range[dataset],
        listener=cfg.listener,
        shuffle_rooms=cfg.shuffle_rooms,
    )
    sb.instantiate_scenes(dataset=dataset)

    print(len(sb.scenes))

    with regtest:
        print(json.dumps(sb.scenes, indent=4))
