import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from clarity.data.scene_builder_cec2 import RoomBuilder, SceneBuilder, set_random_seed

logger = logging.getLogger(__name__)


def instantiate_scenes(cfg):
    room_builder = RoomBuilder()
    set_random_seed(cfg.random_seed)
    room_file = Path(cfg.path.metadata_dir) / "rooms.train.json"
    for dataset in cfg.scene_datasets:
        if not Path(cfg.path.additional_data_file).exists():
            logger.info(f"instantiate scenes for {dataset} set")
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
            scene_builder.save_scenes(cfg.path.additional_data_file)
        else:
            logger.info(f"scenes.{dataset}.json has existed, skip")


@hydra.main(config_path=".", config_name="additional_data_config")
def run(cfg: DictConfig) -> None:
    logger.info("Instantiating scenes for additional training data")
    instantiate_scenes(cfg)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()
