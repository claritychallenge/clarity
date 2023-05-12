"""Evaluate the enhanced signals using a combination of HASPI and HASQI metrics"""
import csv
import hashlib
import json
import logging
import pathlib

import hydra
import numpy as np
from numpy import ndarray
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.evaluator.haspi import haspi_v2_be
from clarity.evaluator.hasqi import hasqi_v2_better_ear
from clarity.utils.audiogram import Audiogram, Listener

logger = logging.getLogger(__name__)


def amplify_signal(signal, audiogram: Audiogram, enhancer, compressor):
    """Amplify signal for a given audiogram"""
    nalr_fir, _ = enhancer.build(audiogram)
    out = enhancer.apply(nalr_fir, signal)
    out, _, _ = compressor.process(out)
    return out


def set_scene_seed(scene):
    """Set a seed that is unique for the given scene"""
    scene_encoded = hashlib.md5(scene.encode("utf-8")).hexdigest()
    scene_md5 = int(scene_encoded, 16) % (10**8)
    np.random.seed(scene_md5)


def compute_metric(
    metric, signal: ndarray, ref: ndarray, listener: Listener, sample_rate: float
):
    """Compute HASPI or HASQI metric"""
    score = metric(
        reference_left=ref[:, 0],
        reference_right=ref[:, 1],
        processed_left=signal[:, 0],
        processed_right=signal[:, 1],
        sample_rate=sample_rate,
        listener=listener,
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

    def add_result(
        self, scene: str, listener: str, score: float, haspi: float, hasqi: float
    ):
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
    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    listeners_dict = Listener.load_listener_dict(cfg.path.listeners_file)
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

    for scene, listener_id in tqdm(scene_listener_pairs):
        logger.info(f"Running evaluation: scene {scene}, listener {listener_id}")

        if cfg.evaluate.set_random_seed:
            set_scene_seed(scene)

        # Read signals

        sr_signal, signal = wavfile.read(
            enhanced_folder / f"{scene}_{listener_id}_enhanced.wav"
        )
        sr_ref_anechoic, ref_anechoic = wavfile.read(
            scenes_folder / f"{scene}_target_anechoic_CH1.wav"
        )
        sr_ref_target, ref_target = wavfile.read(
            scenes_folder / f"{scene}_target_CH1.wav"
        )
        ref_anechoic = ref_anechoic / 32768.0
        ref_target = ref_target / 32768.0

        assert sr_ref_anechoic == sr_ref_target == sr_signal == cfg.nalr.sample_rate

        # amplify left and right ear signals
        listener = listeners_dict[listener_id]

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
            sr_signal,
            amplified.astype(np.float32),
        )

        # Evaluate the amplified signal

        rms_target = np.mean(ref_target**2, axis=0) ** 0.5
        rms_anechoic = np.mean(ref_anechoic**2, axis=0) ** 0.5
        ref = ref_anechoic * rms_target / rms_anechoic

        haspi_score = compute_metric(haspi_v2_be, amplified, ref, listener, sr_signal)
        hasqi_score = compute_metric(
            hasqi_v2_better_ear, amplified, ref, listener, sr_signal
        )
        score = 0.5 * (hasqi_score + haspi_score)

        results_file.add_result(
            scene, listener_id, score=score, haspi=haspi_score, hasqi=hasqi_score
        )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run_calculate_si()
