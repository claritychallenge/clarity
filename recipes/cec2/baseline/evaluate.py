import os
import csv
import json
import logging
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from clarity.evaluator.haspi import haspi_v2_be

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def run_calculate_SI(cfg: DictConfig) -> None:
    scenes_listeners = json.load(open(cfg.path.scenes_listeners_file))
    listener_audiograms = json.load(open(cfg.path.listeners_file))

    sii_file = os.path.join(cfg.path.exp_folder, "sii.csv")
    csv_lines = [["scene", "listener", "sii"]]

    for scene in tqdm(scenes_listeners):
        for listener in scenes_listeners[scene]:
            logger.info(f"Running SI calculation: scene {scene}, listener {listener}")

            # retrieve audiograms
            cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
            audiogram_left = np.array(
                listener_audiograms[listener]["audiogram_levels_l"]
            )
            audiogram_right = np.array(
                listener_audiograms[listener]["audiogram_levels_r"]
            )

            # retrieve signals
            fs_proc, proc = wavfile.read(
                os.path.join(cfg.path.scenes_folder, f"{scene}_mix_CH1.wav")
            )

            fs_ref, ref = wavfile.read(
                os.path.join(cfg.path.scenes_folder, f"{scene}_target_anechoic_CH1.wav")
            )
            assert fs_ref == fs_proc
            proc = proc / 32768.0
            ref = ref / 32768.0
            sii = haspi_v2_be(
                xl=ref[:, 0],
                xr=ref[:, 1],
                yl=proc[:, 0],
                yr=proc[:, 1],
                fs_signal=fs_ref,
                audiogram_l=audiogram_left,
                audiogram_r=audiogram_right,
                audiogram_cfs=cfs,
            )
            logger.info(f"The HASPI score is {sii}")
            csv_lines.append([scene, listener, sii])

    with open(sii_file, "w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines:
            csv_writer.writerow(line)


if __name__ == "__main__":
    run_calculate_SI()
