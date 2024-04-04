""" Run the dummy enhancement. """

import json
import logging
import pathlib

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram, Listener
from recipes.icassp_2023.baseline.evaluate import make_scene_listener_list

logger = logging.getLogger(__name__)


def amplify_signal(signal, audiogram: Audiogram, enhancer, compressor):
    """Amplify signal for a given audiogram"""
    nalr_fir, _ = enhancer.build(audiogram)
    out = enhancer.apply(nalr_fir, signal)
    out, _, _ = compressor.process(out)
    return out


@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    """Run the dummy enhancement."""

    enhanced_folder = pathlib.Path(cfg.path.exp) / "enhanced_signals"
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    listener_dict = Listener.load_listener_dict(cfg.path.listeners_file)
    enhancer = NALR(**cfg.nalr)
    compressor = Compressor(**cfg.compressor)
    amplified_folder = pathlib.Path(cfg.path.exp) / "amplified_signals"
    amplified_folder.mkdir(parents=True, exist_ok=True)

    # Make list of all scene listener pairs that will be run
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )

    for scene, listener_id in tqdm(scene_listener_pairs):
        sample_rate, signal_ch1 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
        )

        _, signal_ch2 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH2.wav"
        )

        _, signal_ch3 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH3.wav"
        )

        # Convert to 32-bit floating point scaled between -1 and 1
        signal_ch1 = (signal_ch1 / 32768.0).astype(np.float32)
        signal_ch2 = (signal_ch2 / 32768.0).astype(np.float32)
        signal_ch3 = (signal_ch3 / 32768.0).astype(np.float32)

        signal = (signal_ch1 + signal_ch2 + signal_ch3) / 3

        # pylint: disable=unused-variable
        listener = listener_dict[listener_id]  # noqa: F841

        wavfile.write(
            enhanced_folder / f"{scene}_{listener_id}_enhanced.wav", sample_rate, signal
        )

        # Apply the baseline NALR amplification

        out_l = amplify_signal(
            signal[:, 0], listener.audiogram_left, enhancer, compressor
        )
        out_r = amplify_signal(
            signal[:, 1], listener.audiogram_right, enhancer, compressor
        )
        amplified = np.stack([out_l, out_r], axis=1)

        if cfg.soft_clip:
            amplified = np.tanh(amplified)

        wavfile.write(
            amplified_folder / f"{scene}_{listener_id}_HA-output.wav",
            sample_rate,
            amplified.astype(np.float32),
        )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    enhance()
