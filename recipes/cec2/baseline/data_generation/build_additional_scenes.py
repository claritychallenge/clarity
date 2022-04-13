import os
import logging

import hydra
from omegaconf import DictConfig

from clarity.data.scene_builder_cec2 import RoomBuilder, SceneBuilder, set_random_seed

logger = logging.getLogger(__name__)


def instantiate_scenes(cfg):
    rb = RoomBuilder()
    set_random_seed(cfg.random_seed)
    room_file = os.path.join(cfg.path.metadata_dir, f"rooms.train.json")
    for dataset in cfg.scene_datasets:
        scene_file = os.path.join(cfg.path.metadata_dir, f"scenes.{dataset}.json")
        if not os.path.exists(cfg.path.additional_data_file):
            logger.info(f"instantiate scenes for {dataset} set")
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
            sb.save_scenes(cfg.path.additional_data_file)
        else:
            logger.info(f"scenes.{dataset}.json has existed, skip")


@hydra.main(config_path=".", config_name="additional_data_config")
def run(cfg: DictConfig) -> None:
    logger.info("Instantiating scenes for additional training data")
    instantiate_scenes(cfg)


if __name__ == "__main__":
    run()
