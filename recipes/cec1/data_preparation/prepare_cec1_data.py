import json
import logging
import os

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.data.scene_renderer_cec1 import Renderer, check_scene_exists


def prepare_data(root_path, metafile_path, scene_folder, num_channels):
    """
    Generate scene data given dataset (train or dev)
    Args:
        root_path: Clarity root path
        metafile_path: scene metafile path
        scene_folder: folder containing generated scenes
        num_channels: number of channels
    """
    with open(metafile_path, "r") as f:
        scenes = json.load(f)

    os.makedirs(scene_folder, exist_ok=True)

    renderer = Renderer(input_path=root_path, output_path=scene_folder, num_channels=3)
    for scene in tqdm(scenes):
        if check_scene_exists(scene, scene_folder, num_channels):
            logging.info(f"Skipping processed scene {scene['scene']}.")
        else:
            renderer.render(
                pre_samples=scene["pre_samples"],
                post_samples=scene["post_samples"],
                dataset=scene["dataset"],
                target=scene["target"]["name"],
                noise_type=scene["interferer"]["type"],
                interferer=scene["interferer"]["name"],
                room=scene["room"]["name"],
                scene=scene["scene"],
                offset=scene["interferer"]["offset"],
                snr_dB=scene["SNR"],
            )


@hydra.main(config_path=".", config_name="data_config")
def run(cfg: DictConfig) -> None:
    for dataset in cfg["datasets"]:
        prepare_data(
            cfg["input_path"],
            cfg["datasets"][dataset]["metafile_path"],
            cfg["datasets"][dataset]["scene_folder"],
            cfg["num_channels"],
        )


if __name__ == "__main__":
    run()
