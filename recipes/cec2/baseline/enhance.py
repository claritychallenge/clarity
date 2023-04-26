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
from clarity.utils.audiogram import Listener

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    enhanced_folder = Path(cfg.path.exp_folder) / "enhanced_signals"
    enhanced_folder.mkdir(parents=True, exist_ok=True)
    with Path(cfg.path.scenes_listeners_file).open("r", encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
    listener_dict = Listener.load_listener_dict(cfg.path.listeners_file)

    enhancer = NALR(**cfg.nalr)
    compressor = Compressor(**cfg.compressor)

    for scene in tqdm(scenes_listeners):
        for listener_id in scenes_listeners[scene]:
            listener = listener_dict[listener_id]

            sample_rate, signal = wavfile.read(
                Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
            )
            signal = signal / 32768.0
            assert sample_rate == cfg.nalr.sample_rate
            nalr_fir, _ = enhancer.build(listener.audiogram_left)
            out_l = enhancer.apply(nalr_fir, signal[:, 0])

            nalr_fir, _ = enhancer.build(listener.audiogram_right)
            out_r = enhancer.apply(nalr_fir, signal[:, 1])

            out_l, _, _ = compressor.process(out_l)
            out_r, _, _ = compressor.process(out_r)
            enhanced = np.stack([out_l, out_r], axis=1)
            filename = f"{scene}_{listener_id}_HA-output.wav"

            if cfg.soft_clip:
                enhanced = np.tanh(enhanced)
            n_clipped = np.sum(np.abs(enhanced) > 1.0)
            if n_clipped > 0:
                logger.warning(f"Writing {filename}: {n_clipped} samples clipped")
            np.clip(enhanced, -1.0, 1.0, out=enhanced)
            signal_16 = (32768.0 * enhanced).astype(np.int16)
            wavfile.write(enhanced_folder / filename, sample_rate, signal_16)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    enhance()
