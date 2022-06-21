import json
import logging

import hydra
from omegaconf import DictConfig

from clarity.data.scene_renderer_cec2 import SceneRenderer

logger = logging.getLogger(__name__)


def render_scenes(cfg):
    for dataset in cfg.scene_renderer:
        logger.info(f"Beginning scene generation for {dataset} set...")
        with open(cfg.scene_renderer[dataset].metadata.scene_definitions, "r") as f:
            scenes = json.load(f)

        starting_scene = (
            cfg.scene_renderer[dataset].chunk_size * cfg.render_starting_chunk
        )
        n_scenes = (
            cfg.scene_renderer[dataset].chunk_size * cfg.render_n_chunk_to_process
        )
        scenes = scenes[starting_scene : starting_scene + n_scenes]

        scene_renderer = SceneRenderer(
            cfg.scene_renderer[dataset].paths,
            cfg.scene_renderer[dataset].metadata,
            **cfg.render_params,
        )
        scene_renderer.render_scenes(scenes)


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    logger.info("Rendering scenes")
    render_scenes(cfg)


if __name__ == "__main__":
    run()
