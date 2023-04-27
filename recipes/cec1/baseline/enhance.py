import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.enhancer.gha.gha_interface import GHAHearingAid
from clarity.utils.audiogram import Listener


@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    enhanced_folder = Path(cfg.path.exp_folder) / "enhanced_signals"
    enhanced_folder.mkdir(parents=True, exist_ok=True)
    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
    listener_dict = Listener.load_listener_dict(cfg.path.listeners_file)
    enhancer = GHAHearingAid(**cfg["GHAHearingAid"])

    for scene in tqdm(scenes_listeners):
        for listener_id in scenes_listeners[scene]:
            listener = listener_dict[listener_id]

            infile_names = [
                f"{cfg.path.scenes_folder}/{scene}_mixed_CH{ch}.wav"
                for ch in range(1, cfg["num_channels"] + 1)
            ]

            enhancer.process_files(
                infile_names=infile_names,
                outfile_name=f"{enhanced_folder}/{scene}_{listener_id}_HA-output.wav",
                listener=listener,
            )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    enhance()
