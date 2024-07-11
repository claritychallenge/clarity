"""
Module to generate new train scenes"""

from __future__ import annotations

import hashlib
# pylint: disable=import-error
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


def set_song_seed(song: str) -> None:
    """Set a seed that is unique for the given song"""
    song_encoded = hashlib.md5(song.encode("utf-8")).hexdigest()
    song_md5 = int(song_encoded, 16) % (10**8)
    np.random.seed(song_md5)


def choose_samples(source: list, number: int) -> list:
    """Choose a number of samples from a source list"""
    if number == 0:
        return source
    return np.random.choice(source, number, replace=False)


def generate_scenes(cfg: DictConfig) -> dict:
    """Generate scenes for the training set"""
    scene_metadata = {}

    scene_id = 10000
    music_metadata = Path(cfg.path.music_metadata)

    gains_metadata = Path(cfg.path.gain_metadata)
    with open(gains_metadata) as f:
        gains = json.load(f)

    with open(music_metadata) as f:
        music_meta = json.load(f)

    for track, track_info in music_meta.items():
        gains_per_track = int(cfg.gains_per_track)
        instr_track = len(track_info) - 1
        instr_track = min(5, instr_track)

        for _ in range(gains_per_track):
            scene_id += 1
            scene_metadata[f"S{scene_id}"] = {
                "music": track,
                "gain": np.random.choice(list(gains[str(instr_track)].keys())),
            }
    return scene_metadata


def generate_scene_listener(cfg: DictConfig) -> None:
    """Generate listeners for the scenes"""
    scene_metadata_path = Path(cfg.path.scene_metadata_file)

    with open(scene_metadata_path) as f:
        scenes = json.load(f)

    with open(f"{cfg.path.metadata_dir}/listeners.train.json") as f:
        listeners = list(json.load(f).keys())

    scene_listeners = {}

    set_song_seed("Cadenza2-scene")
    for scene in scenes:
        scene_listeners[scene] = list(
            np.random.choice(listeners, cfg.listener_per_scene)
        )

    output_file = Path(cfg.path.metadata_dir) / "scene_listeners.train.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(scene_listeners, f, indent=4)


@hydra.main(config_path="", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    set_song_seed(cfg.seed_scene_generation)

    scene_metadata = generate_scenes(cfg)
    scene_metadata_path = Path(cfg.path.scene_metadata_file)

    with open(scene_metadata_path, "w") as f:
        json.dump(scene_metadata, f, indent=4)

    generate_scene_listener(cfg)


if __name__ == "__main__":
    run()
    print("Done!")
