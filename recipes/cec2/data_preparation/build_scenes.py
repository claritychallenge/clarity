import logging
import os

import hydra
from omegaconf import DictConfig

from clarity.data.scene_builder_cec2 import RoomBuilder, SceneBuilder, set_random_seed

logger = logging.getLogger(__name__)


def build_rooms_from_rpf(cfg):
    rb = RoomBuilder()
    for dataset in cfg.room_datasets:
        room_file = os.path.join(cfg.path.metadata_dir, f"rooms.{dataset}.json")
        if not os.path.exists(room_file):
            rb.build_from_rpf(**cfg.room_datasets[dataset])
            rb.save_rooms(room_file)
        else:
            logger.info(f"rooms.{dataset}.json exists, skip")


def instantiate_scenes(cfg):
    rb = RoomBuilder()
    set_random_seed(cfg.random_seed)
    for dataset in cfg.scene_datasets:
        scene_file = os.path.join(cfg.path.metadata_dir, f"scenes.{dataset}.json")
        if not os.path.exists(scene_file):
            logger.info(f"instantiate scenes for {dataset} set")
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
            sb.save_scenes(scene_file)
        else:
            logger.info(f"scenes.{dataset}.json exists, skip")


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    logger.info("Building rooms")
    build_rooms_from_rpf(cfg)
    logger.info("Instantiating scenes")
    instantiate_scenes(cfg)


if __name__ == "__main__":
    run()
