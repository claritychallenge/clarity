"""Module to generate the scenes and scene-listeners metadata files for training"""
from __future__ import annotations

# pylint: disable=import-error
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from recipes.cad_icassp_2024.baseline.evaluate import set_song_seed


def choose_samples(source: list, size: int) -> list:
    """Choose a number of samples from a list
    Args:
        source: List of samples to choose from
        size: Number of samples to choose
    Returns:
        List of chosen samples
    """
    if len(source) == 0:
        return []
    if len(source) <= size:
        return source
    return np.random.choice(source, size, replace=False)


def generate_scenes(cfg: DictConfig) -> None:
    """
    Generate the scenes metadata file for training
    Args:
        cfg: Hydra config
    """

    with open(cfg.path.gains_file, encoding="utf-8") as f:
        gains = list(json.load(f).keys())

    with open(cfg.path.head_loudspeaker_positions_file, encoding="utf-8") as f:
        head_loudspeaker_positions = list(json.load(f).keys())

    scenes_metadata = {}
    scene_id = 10001

    music_path = Path(cfg.path.metadata_dir) / "musdb18.train.json"
    with open(music_path, encoding="utf-8") as f:
        tracks_list = [item["Track Name"] for item in json.load(f)]

    for track_name in tracks_list:
        set_song_seed(track_name)

        gains_to_choose = gains.copy()
        hlp_to_choose = head_loudspeaker_positions.copy()

        for _ in range(cfg.scene.number_scenes_per_song):
            g = choose_samples(gains_to_choose, 1)[0]
            hlp = choose_samples(hlp_to_choose, 1)[0]

            # remove the chosen gain and head loudspeaker position from the list
            # so we choose a different one next time
            gains_to_choose.remove(g)
            hlp_to_choose.remove(hlp)

            scenes_metadata[f"scene_{scene_id:04d}"] = {
                "music": track_name,
                "gain": g,
                "head_loudspeaker_positions": hlp,
            }
            scene_id += 1

    output_file = Path(cfg.path.metadata_dir) / "scenes.train.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scenes_metadata, f, indent=4)


def generate_scene_listener(cfg: DictConfig) -> None:
    """Generates the scene-listeners metadata file for training

    Args:
        cfg: Hydra config
    """
    scenes_path = Path(cfg.path.metadata_dir) / "scenes.train.json"
    with open(scenes_path, encoding="utf-8") as f:
        scenes = list(json.load(f).keys())

    listeners_path = Path(cfg.path.metadata_dir) / "listeners.train.json"
    with open(listeners_path, encoding="utf-8") as f:
        listeners = list(json.load(f).keys())

    scene_listeners = {}

    set_song_seed("icassp2024")

    for scene in scenes:
        scene_listeners[scene] = list(
            choose_samples(listeners, cfg.scene_listener.number_listeners_per_scene)
        )

    output_file = Path(cfg.path.metadata_dir) / "scene_listeners.train.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scene_listeners, f, indent=4)


@hydra.main(config_path="", config_name="config")
def run(cfg: DictConfig) -> None:
    """Module generates the scenes and scene-listeners metadata files for training."""
    generate_scenes(cfg)
    generate_scene_listener(cfg)


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    run()
