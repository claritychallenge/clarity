import csv
import hashlib
import json
import logging
import os
import pathlib

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.evaluator.haspi import haspi_v2_be
from clarity.evaluator.hasqi import hasqi_v2_be

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def run_calculate_SI(cfg: DictConfig) -> None:
    scenes_listeners = json.load(open(cfg.path.scenes_listeners_file))
    listener_audiograms = json.load(open(cfg.path.listeners_file))
    os.makedirs(cfg.path.exp_folder, exist_ok=True)

    enhancer = NALR(**cfg.nalr)
    compressor = Compressor(**cfg.compressor)

    enhanced_folder = pathlib.Path(cfg.path.exp_folder) / "enhanced_signals"
    amplified_folder = pathlib.Path(cfg.path.exp_folder) / "amplified_signals"
    os.makedirs(amplified_folder, exist_ok=True)

    si_file = os.path.join(cfg.path.exp_folder, "si.csv")
    csv_lines = [["scene", "listener", "combined_metrics", "haspi", "hasqi"]]

    for scene in tqdm(scenes_listeners):
        for listener in scenes_listeners[scene]:
            logger.info(f"Running evaluation: scene {scene}, listener {listener}")
            if cfg.evaluate.set_random_seed:
                scene_md5 = int(hashlib.md5(scene.encode("utf-8")).hexdigest(), 16) % (
                    10**8
                )
                np.random.seed(scene_md5)

            # retrieve audiograms
            cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
            audiogram_left = np.array(
                listener_audiograms[listener]["audiogram_levels_l"]
            )
            audiogram_right = np.array(
                listener_audiograms[listener]["audiogram_levels_r"]
            )

            filename = f"{scene}_{listener}_enhanced.wav"

            fs, signal = wavfile.read(os.path.join(enhanced_folder, filename))

            assert fs == cfg.nalr.fs
            nalr_fir, _ = enhancer.build(audiogram_left, cfs)
            out_l = enhancer.apply(nalr_fir, signal[:, 0])

            nalr_fir, _ = enhancer.build(audiogram_right, cfs)
            out_r = enhancer.apply(nalr_fir, signal[:, 1])

            out_l, _, _ = compressor.process(out_l)
            out_r, _, _ = compressor.process(out_r)
            amplified = np.stack([out_l, out_r], axis=1)
            filename = f"{scene}_{listener}_HA-output.wav"

            if cfg.soft_clip:
                amplified = np.tanh(amplified)

            # Output the amplified signals
            wavfile.write(
                pathlib.Path(amplified_folder) / filename,
                fs,
                amplified.astype(np.float32),
            )

            fs_ref_anechoic, ref_anechoic = wavfile.read(
                pathlib.Path(cfg.path.scenes_folder)
                / f"{scene}_target_anechoic_CH1.wav"
            )

            fs_ref_target, ref_target = wavfile.read(
                pathlib.Path(cfg.path.scenes_folder) / f"{scene}_target_CH1.wav"
            )

            # assert fs_ref_anechoic == fs_ref_target == fs_proc

            rms_target = np.mean(ref_target**2, axis=0) ** 0.5
            rms_anechoic = np.mean(ref_anechoic**2, axis=0) ** 0.5
            ref = ref_anechoic * rms_target / rms_anechoic

            si_haspi = haspi_v2_be(
                xl=ref[:, 0],
                xr=ref[:, 1],
                yl=amplified[:, 0],
                yr=amplified[:, 1],
                fs_signal=fs_ref_anechoic,
                audiogram_l=audiogram_left,
                audiogram_r=audiogram_right,
                audiogram_cfs=cfs,
            )

            si_hasqi = hasqi_v2_be(
                xl=ref[:, 0],
                xr=ref[:, 1],
                yl=amplified[:, 0],
                yr=amplified[:, 1],
                fs_signal=fs_ref_anechoic,
                audiogram_l=audiogram_left,
                audiogram_r=audiogram_right,
                audiogram_cfs=cfs,
            )

            si = np.mean([si_hasqi, si_haspi])
            logger.info(
                f"The combined score is {si} (haspi {si_haspi}, hasqi {si_hasqi})"
            )
            csv_lines.append([scene, listener, str(si), str(si_haspi), str(si_hasqi)])

    with open(si_file, "w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines:
            csv_writer.writerow(line)


if __name__ == "__main__":
    run_calculate_SI()
