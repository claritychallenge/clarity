import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.enhancer.gha.audiogram import Audiogram
from clarity.enhancer.gha.gha_interface import GHAHearingAid


@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    enhanced_folder = Path(cfg.path.exp_folder) / "enhanced_signals"
    enhanced_folder.mkdir(parents=True, exist_ok=True)
    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
    with open(cfg.path.listeners_file, encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)

    enhancer = GHAHearingAid(**cfg["GHAHearingAid"])

    for scene in tqdm(scenes_listeners):
        for listener in scenes_listeners[scene]:
            # retrieve audiograms
            cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
            audiogram_left = np.array(
                listener_audiograms[listener]["audiogram_levels_l"]
            )
            audiogram_right = np.array(
                listener_audiograms[listener]["audiogram_levels_r"]
            )
            audiogram = Audiogram(
                cfs=cfs, levels_l=audiogram_left, levels_r=audiogram_right
            )

            infile_names = [
                f"{cfg.path.scenes_folder}/{scene}_mixed_CH{ch}.wav"
                for ch in range(1, cfg["num_channels"] + 1)
            ]

            enhancer.process_files(
                infile_names=infile_names,
                outfile_name=f"{enhanced_folder}/{scene}_{listener}_HA-output.wav",
                audiogram=audiogram,
                listener=listener,
            )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    enhance()
