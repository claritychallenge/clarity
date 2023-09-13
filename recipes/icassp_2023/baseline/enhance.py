""" Run the dummy enhancement. """
import json
import logging
import pathlib

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from clarity.utils.audiogram import Listener
from recipes.icassp_2023.baseline.evaluate import make_scene_listener_list

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    """Run the dummy enhancement."""

    enhanced_folder = pathlib.Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    listener_dict = Listener.load_listener_dict(cfg.path.listeners_file)

    # Make list of all scene listener pairs that will be run
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )

    for scene, listener_id in tqdm(scene_listener_pairs):
        sample_rate, signal = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
        )

        # Convert to 32-bit floating point scaled between -1 and 1
        signal = (signal / 32768.0).astype(np.float32)

        # pylint: disable=unused-variable
        listener = listener_dict[listener_id]  # noqa: F841

        # Note: The audiograms are stored in the listener object,
        # but they are not needed for the baseline

        # Baseline just reads the signal from the front microphone pair
        # and write it out as the enhanced signal

        wavfile.write(
            enhanced_folder / f"{scene}_{listener_id}_enhanced.wav", sample_rate, signal
        )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    enhance()
