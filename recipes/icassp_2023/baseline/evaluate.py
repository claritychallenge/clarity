"""Evaluate the enhanced signals using a combination of HASPI and HASQI metrics"""
import csv
import hashlib
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
from clarity.evaluator.haspi import haspi_v2_be
from clarity.evaluator.hasqi import hasqi_v2_be

logger = logging.getLogger(__name__)


def amplify_signal(signal, bin_audiogram, ear, enhancer, compressor):
    """Amplify signal for either left (l) or right (r) ear"""
    cfs = np.array(bin_audiogram["audiogram_cfs"])
    audiogram = np.array(bin_audiogram[f"audiogram_levels_{ear}"])
    nalr_fir, _ = enhancer.build(audiogram, cfs)
    out = enhancer.apply(nalr_fir, signal)
    out, _, _ = compressor.process(out)
    return out


def set_scene_seed(scene):
    """Set a seed that is unique for the given scene"""
    scene_encoded = hashlib.md5(scene.encode("utf-8")).hexdigest()
    scene_md5 = int(scene_encoded, 16) % (10**8)
    np.random.seed(scene_md5)


def compute_metric(metric, signal, ref, audiogram, fs_signal):
    """Compute HASPI or HASQI metric"""
    score = metric(
        xl=ref[:, 0],
        xr=ref[:, 1],
        yl=signal[:, 0],
        yr=signal[:, 1],
        fs_signal=fs_signal,
        audiogram_l=audiogram["audiogram_levels_l"],
        audiogram_r=audiogram["audiogram_levels_r"],
        audiogram_cfs=audiogram["audiogram_cfs"],
    )
    return score


class ResultsFile:
    """Class to write results to a CSV file"""

    def __init__(self, file_name):
        self.file_name = file_name

    def write_header(self):
        with open(self.file_name, "w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(["scene", "listener", "combined", "haspi", "hasqi"])

    def add_result(self, scene, listener, score, haspi, hasqi):
        """Add a result to the CSV file"""

        logger.info(f"The combined score is {score} (haspi {haspi}, hasqi {hasqi})")

        with open(self.file_name, "a", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow([scene, listener, str(score), str(haspi), str(hasqi)])


def make_scene_listener_list(scenes_listeners, small_test=False):
    """Make the list of scene-listener pairing to process"""
    scene_listener_pairs = [
        (scene, listener)
        for scene in scenes_listeners
        for listener in scenes_listeners[scene]
    ]

    # Can define a standard 'small_test' with just 1/15 of the data
    if small_test:
        scene_listener_pairs = scene_listener_pairs[::15]

    return scene_listener_pairs


@hydra.main(config_path=".", config_name="config")
def run_calculate_si(cfg: DictConfig) -> None:
    """Evaluate the enhanced signals using a combination of HASPI and HASQI metrics"""

    # Load listener data
    with open(cfg.path.scenes_listeners_file, "r", encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    with open(cfg.path.listeners_file, "r", encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)

    enhancer = NALR(**cfg.nalr)
    compressor = Compressor(**cfg.compressor)

    enhanced_folder = pathlib.Path("enhanced_signals")
    amplified_folder = pathlib.Path("amplified_signals")
    scenes_folder = pathlib.Path(cfg.path.scenes_folder)
    amplified_folder.mkdir(parents=True, exist_ok=True)

    # Make list of all scene listener pairs that will be run

    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )

    results_file = ResultsFile("scores.csv")
    results_file.write_header()

    for scene, listener in tqdm(scene_listener_pairs):
        logger.info(f"Running evaluation: scene {scene}, listener {listener}")

        if cfg.evaluate.set_random_seed:
            set_scene_seed(scene)

        # Read signals

        fs_signal, signal = wavfile.read(
            enhanced_folder / f"{scene}_{listener}_enhanced.wav"
        )
        fs_ref_anechoic, ref_anechoic = wavfile.read(
            scenes_folder / f"{scene}_target_anechoic_CH1.wav"
        )
        fs_ref_target, ref_target = wavfile.read(
            scenes_folder / f"{scene}_target_CH1.wav"
        )
        ref_anechoic = ref_anechoic / 32768.0
        ref_target = ref_target / 32768.0

        assert fs_ref_anechoic == fs_ref_target == fs_signal == cfg.nalr.fs

        # amplify left and right ear signals
        audiogram = listener_audiograms[listener]

        out_l = amplify_signal(signal[:, 0], audiogram, "l", enhancer, compressor)
        out_r = amplify_signal(signal[:, 1], audiogram, "r", enhancer, compressor)
        amplified = np.stack([out_l, out_r], axis=1)

        if cfg.soft_clip:
            amplified = np.tanh(amplified)

        wavfile.write(
            amplified_folder / f"{scene}_{listener}_HA-output.wav",
            fs_signal,
            amplified.astype(np.float32),
        )

        # Evaluate the amplified signal

        rms_target = np.mean(ref_target**2, axis=0) ** 0.5
        rms_anechoic = np.mean(ref_anechoic**2, axis=0) ** 0.5
        ref = ref_anechoic * rms_target / rms_anechoic

        haspi_score = compute_metric(haspi_v2_be, amplified, ref, audiogram, fs_signal)
        hasqi_score = compute_metric(hasqi_v2_be, amplified, ref, audiogram, fs_signal)
        score = 0.5 * (hasqi_score + haspi_score)

        results_file.add_result(
            scene, listener, score=score, haspi=haspi_score, hasqi=hasqi_score
        )


if __name__ == "__main__":
    run_calculate_si()  # noqa
