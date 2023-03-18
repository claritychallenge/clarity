import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    enhanced_folder = Path(cfg.path.exp_folder) / "enhanced_signals"
    enhanced_folder.mkdir(parents=True, exist_ok=True)
    with Path(cfg.path.scenes_listeners_file).open("r", encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
    with Path(cfg.path.listeners_file).open("r", encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)

    enhancer = NALR(**cfg.nalr)
    compressor = Compressor(**cfg.compressor)

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

            fs, signal = wavfile.read(
                Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
            )
            signal = signal / 32768.0
            assert fs == cfg.nalr.fs
            nalr_fir, _ = enhancer.build(audiogram_left, cfs)
            out_l = enhancer.apply(nalr_fir, signal[:, 0])

            nalr_fir, _ = enhancer.build(audiogram_right, cfs)
            out_r = enhancer.apply(nalr_fir, signal[:, 1])

            out_l, _, _ = compressor.process(out_l)
            out_r, _, _ = compressor.process(out_r)
            enhanced = np.stack([out_l, out_r], axis=1)
            filename = f"{scene}_{listener}_HA-output.wav"

            if cfg.soft_clip:
                enhanced = np.tanh(enhanced)
            n_clipped = np.sum(np.abs(enhanced) > 1.0)
            if n_clipped > 0:
                logger.warning(f"Writing {filename}: {n_clipped} samples clipped")
            np.clip(enhanced, -1.0, 1.0, out=enhanced)
            signal_16 = (32768.0 * enhanced).astype(np.int16)
            wavfile.write(enhanced_folder / filename, fs, signal_16)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    enhance()
