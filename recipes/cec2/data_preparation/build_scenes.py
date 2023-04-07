import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from clarity.data.scene_builder_cec2 import RoomBuilder, SceneBuilder, set_random_seed

logger = logging.getLogger(__name__)


def build_rooms_from_rpf(cfg):
    room_builder = RoomBuilder()
    for dataset in cfg.room_datasets:
        room_file = Path(cfg.path.metadata_dir) / f"rooms.{dataset}.json"
        if not room_file.exists():
            room_builder.build_from_rpf(**cfg.room_datasets[dataset])
            room_builder.save_rooms(str(room_file))
        else:
            logger.info(f"rooms.{dataset}.json exists, skip")


def instantiate_scenes(cfg):
    room_builder = RoomBuilder()
    set_random_seed(cfg.random_seed)
    for dataset in cfg.scene_datasets:
        scene_file = Path(cfg.path.metadata_dir) / f"scenes.{dataset}.json"
        if not scene_file.exists():
            logger.info(f"instantiate scenes for {dataset} set")
            room_file = Path(cfg.path.metadata_dir) / f"rooms.{dataset}.json"
            room_builder.load(str(room_file))
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
            scene_builder.save_scenes(str(scene_file))
        else:
            logger.info(f"scenes.{dataset}.json exists, skip")


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    logger.info("Building rooms")
    build_rooms_from_rpf(cfg)
    logger.info("Instantiating scenes")
    instantiate_scenes(cfg)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()
