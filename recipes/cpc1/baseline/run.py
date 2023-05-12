import csv  # pylint: disable=ungrouped-imports
import json  # pylint: disable=ungrouped-imports
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.signal import unit_impulse
from tqdm import tqdm

from clarity.evaluator.mbstoi.mbstoi import mbstoi
from clarity.evaluator.mbstoi.mbstoi_utils import find_delay_impulse
from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import MSBG_FS, pad
from clarity.utils.audiogram import AUDIOGRAM_REF_CLARITY, Listener
from clarity.utils.file_io import read_signal, write_signal

logger = logging.getLogger(__name__)


def listen(ear, signal, listener: Listener):
    """
    Generate MSBG processed signal
    :param ear: MSBG ear
    :param signal: binaural signal
    :param listener: listener - the listener characteristics to simulate
    :return: binaural signal
    """
    ear.set_audiogram(listener.audiogram_left)
    out_l = ear.process(signal[:, 0])

    ear.set_audiogram(listener.audiogram_right)
    out_r = ear.process(signal[:, 1])

    if len(out_l[0]) != len(out_r[0]):
        diff = len(out_l[0]) - len(out_r[0])
        if diff > 0:
            out_r[0] = np.flipud(pad(np.flipud(out_r[0]), len(out_l[0])))
        else:
            out_l[0] = np.flipud(pad(np.flipud(out_l[0]), len(out_r[0])))
    return np.concatenate([out_l, out_r]).T


def run_HL_processing(cfg, path):
    output_path = Path(path.exp_folder) / "eval_signals"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(path.scenes_file, encoding="utf-8") as fp:
        scenes = json.load(fp)
    listener_dict = Listener.load_listener_dict(path.listeners_file)

    enhanced_folder = Path(path.scenes_folder)

    # initialize ear
    ear = Ear(**cfg["MSBGEar"])

    for scene_dict in tqdm(scenes):
        scene = scene_dict["scene"]
        listener_id = scene_dict["listener"]
        system = scene_dict["system"]
        signal_file = enhanced_folder / f"{scene}_{listener_id}_{system}.wav"

        # signals to write
        outfile_stem = f"{output_path}/{scene}_{listener_id}_{system}"
        signal_files_to_write = [
            f"{output_path}/{scene}_flat0dB_HL-output.wav",
            f"{outfile_stem}_HL-output.wav",
            f"{outfile_stem}_HLddf-output.wav",
            f"{outfile_stem}_HL-mixoutput.wav",
        ]
        # if all signals to write exist, pass
        if all(Path(f).exists() for f in signal_files_to_write):
            continue

        signal = read_signal(signal_file)
        mixture_signal = read_signal(Path(path.ref_folder) / f"{scene}_mixed_CH0.wav")

        # retrieve audiograms
        listener = listener_dict[listener_id]

        # Create discrete delta function (DDF) signal for time alignment
        ddf_signal = np.zeros(np.shape(signal))
        ddf_signal[:, 0] = unit_impulse(len(signal), int(MSBG_FS / 2))
        ddf_signal[:, 1] = unit_impulse(len(signal), int(MSBG_FS / 2))
        reference_listener = Listener(AUDIOGRAM_REF_CLARITY, AUDIOGRAM_REF_CLARITY)
        signals_to_write = [
            listen(ear, ddf_signal, reference_listener),
            listen(ear, signal, listener),
            listen(ear, ddf_signal, listener),
            listen(ear, mixture_signal, listener),
        ]
        for signal, signal_file in zip(signals_to_write, signal_files_to_write):
            write_signal(
                signal_file,
                signal,
                MSBG_FS,
                floating_point=True,
            )


def run_calculate_SI(cfg, path) -> None:
    with open(path.scenes_file, encoding="utf-8") as fp:
        scenes = json.load(fp)
    proc_folder = Path(path.exp_folder) / "eval_signals"
    ref_folder = Path(path.ref_folder)
    sii_file = Path(path.exp_folder) / "sii.csv"
    csv_lines = [["signal_ID", "intelligibility_score"]]

    for scene_dict in tqdm(scenes):
        scene = scene_dict["scene"]
        listener = scene_dict["listener"]
        system = scene_dict["system"]
        proc = read_signal(proc_folder / f"{scene}_{listener}_{system}_HL-output.wav")
        clean = read_signal(ref_folder / f"{scene}_target_anechoic.wav")
        ddf = read_signal(proc_folder / f"{scene}_{listener}_{system}_HLddf-output.wav")

        # Calculate channel-specific unit impulse delay due to HL model and audiograms
        delay = find_delay_impulse(ddf, initial_value=int(MSBG_FS / 2))
        if delay[0] != delay[1]:
            logging.info(f"Difference in delay of {delay[0] - delay[1]}.")
        maxdelay = int(np.max(delay))

        # Allow for value lower than 1000 samples in case of unimpaired hearing
        if maxdelay > 2000:
            logging.error("Error in delay calculation for signal time-alignment.")

        # Correct for delays by padding clean signals
        cleanpad = np.zeros((len(clean) + maxdelay, 2))
        procpad = np.zeros((len(clean) + maxdelay, 2))

        if len(procpad) < len(proc):
            raise ValueError("Padded processed signal is too short.")

        cleanpad[int(delay[0]) : int(len(clean) + int(delay[0])), 0] = clean[:, 0]
        cleanpad[int(delay[1]) : int(len(clean) + int(delay[1])), 1] = clean[:, 1]
        procpad[: len(proc)] = proc

        sii = mbstoi(
            cleanpad[:, 0],
            cleanpad[:, 1],
            procpad[:, 0],
            procpad[:, 1],
            cfg["mbstoi"]["sample_rate"],
            cfg["mbstoi"]["grid_coarseness"],
        )
        csv_lines.append([f"{scene}_{listener}_{system}", sii])  # type: ignore

    with open(sii_file, "w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines:
            csv_writer.writerow(line)


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    logger.info("Prediction with MSGB + MBSTOI for train set")
    run_HL_processing(cfg, cfg.train_path)
    run_calculate_SI(cfg, cfg.train_path)

    logger.info("Prediction with MSGB + MBSTOI for train_indep set")
    run_HL_processing(cfg, cfg.train_indep_path)
    run_calculate_SI(cfg, cfg.train_indep_path)

    logger.info("Prediction with MSGB + MBSTOI for test set")
    run_HL_processing(cfg, cfg.test_path)
    run_calculate_SI(cfg, cfg.test_path)

    logger.info("Prediction with MSGB + MBSTOI for test_indep set")
    run_HL_processing(cfg, cfg.test_indep_path)
    run_calculate_SI(cfg, cfg.test_indep_path)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()
