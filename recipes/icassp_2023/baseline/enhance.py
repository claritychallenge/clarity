import json
import logging
import os
import pathlib

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    enhanced_folder = pathlib.Path(cfg.path.exp_folder) / "enhanced_signals"
    os.makedirs(enhanced_folder, exist_ok=True)
    scenes_listeners = json.load(open(cfg.path.scenes_listeners_file))
    listener_audiograms = json.load(open(cfg.path.listeners_file))  # noqa: F841

    for scene in tqdm(scenes_listeners):
        fs, signal = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
        )

        # Convert to 32-bit floating point scaled between -1 and 1
        signal = (signal / 32768.0).astype(np.float32)

        for listener in scenes_listeners[scene]:

            # # Audiograms can read like this, but they are not needed for the baseline
            #
            # cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
            #
            # audiogram_left = np.array(
            #    listener_audiograms[listener]["audiogram_levels_l"]
            # )
            # audiogram_right = np.array(
            #    listener_audiograms[listener]["audiogram_levels_r"]
            # )

            # Baseline just reads the signal from the front microphone pair
            # and write it out as the enhanced signal
            #

            filename = f"{scene}_{listener}_enhanced.wav"

            wavfile.write(os.path.join(enhanced_folder, filename), fs, signal)


if __name__ == "__main__":
    enhance()
