""" Run the dummy enhancement. """
import json
import logging
import pathlib

import hydra
import numpy as np
from evaluate import make_scene_listener_list
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    """Run the dummy enhancement."""

    enhanced_folder = pathlib.Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    with open(cfg.path.scenes_listeners_file, "r", encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    with open(cfg.path.listeners_file, "r", encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)  # noqa: F841

    # Make list of all scene listener pairs that will be run
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )

    for scene, listener in tqdm(scene_listener_pairs):
        sample_freq, signal = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
        )

        # Convert to 32-bit floating point scaled between -1 and 1
        signal = (signal / 32768.0).astype(np.float32)

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

        wavfile.write(
            enhanced_folder / f"{scene}_{listener}_enhanced.wav", sample_freq, signal
        )


if __name__ == "__main__":
    enhance()
