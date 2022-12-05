import csv
import json
import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.signal import unit_impulse
from tqdm import tqdm

from clarity.evaluator.mbstoi.mbstoi import mbstoi
from clarity.evaluator.mbstoi.mbstoi_utils import find_delay_impulse
from clarity.evaluator.msbg.audiogram import Audiogram
from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import MSBG_FS, pad, read_signal, write_signal


def listen(ear, signal, audiogram_l, audiogram_r):
    """
    Generate MSBG processed signal
    :param ear: MSBG ear
    :param wav: binaural signal
    :return: binaural signal
    """
    ear.set_audiogram(audiogram_l)
    out_l = ear.process(signal[:, 0])

    ear.set_audiogram(audiogram_r)
    out_r = ear.process(signal[:, 1])
    if len(out_l[0]) != len(out_r[0]):
        diff = len(out_l[0]) - len(out_r[0])
        if diff > 0:
            out_r[0] = np.flipud(pad(np.flipud(out_r[0]), len(out_l[0])))
        else:
            out_l[0] = np.flipud(pad(np.flipud(out_l[0]), len(out_r[0])))
    return np.concatenate([out_l, out_r]).T


@hydra.main(config_path=".", config_name="config")
def run_HL_processing(cfg: DictConfig) -> None:
    output_path = os.path.join(cfg["path"]["exp_folder"], "eval_signals")
    os.makedirs(output_path, exist_ok=True)
    scenes_listeners = json.load(open(cfg["path"]["scenes_listeners_file"]))
    listener_audiograms = json.load(open(cfg["path"]["listeners_file"]))
    enhanced_folder = cfg["path"]["enhanced_signals"]

    # initialize ear
    ear = Ear(**cfg["MSBGEar"])

    for scene in tqdm(scenes_listeners):
        for listener in scenes_listeners[scene]:
            if os.path.exists(enhanced_folder):
                signal_file = os.path.join(
                    enhanced_folder, f"{scene}_{listener}_HA-output.wav"
                )
            # if no enhanced signals, use the unprocessed signal for si calculation
            else:
                signal_file = os.path.join(
                    cfg["path"]["scenes_folder"], f"{scene}_mixed_CH0.wav"
                )

            # signals to write
            outfile_stem = f"{output_path}/{scene}_{listener}"
            signal_files_to_write = [
                f"{output_path}/{scene}_flat0dB_HL-output.wav",
                f"{outfile_stem}_HL-output.wav",
                f"{outfile_stem}_HLddf-output.wav",
                f"{outfile_stem}_HL-mixoutput.wav",
            ]
            # if all signals to write exist, pass
            if all([os.path.isfile(f) for f in signal_files_to_write]):
                continue

            signal = read_signal(signal_file)
            mixture_signal = read_signal(
                os.path.join(cfg["path"]["scenes_folder"], f"{scene}_mixed_CH0.wav")
            )

            # retrieve audiograms
            cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
            audiogram_left = np.array(
                listener_audiograms[listener]["audiogram_levels_l"]
            )
            left_audiogram = Audiogram(cfs=cfs, levels=audiogram_left)
            audiogram_right = np.array(
                listener_audiograms[listener]["audiogram_levels_r"]
            )
            right_audiogram = Audiogram(cfs=cfs, levels=audiogram_right)

            # Create discrete delta function (DDF) signal for time alignment
            ddf_signal = np.zeros((np.shape(signal)))
            ddf_signal[:, 0] = unit_impulse(len(signal), int(MSBG_FS / 2))
            ddf_signal[:, 1] = unit_impulse(len(signal), int(MSBG_FS / 2))

            # Get flat-0dB ear audiograms
            flat0dB_audiogram = Audiogram(cfs=cfs, levels=np.zeros((np.shape(cfs))))

            signals_to_write = [
                listen(ear, ddf_signal, flat0dB_audiogram, flat0dB_audiogram),
                listen(ear, signal, left_audiogram, right_audiogram),
                listen(ear, ddf_signal, left_audiogram, right_audiogram),
                listen(ear, mixture_signal, left_audiogram, right_audiogram),
            ]
            for i in range(len(signals_to_write)):
                write_signal(
                    signal_files_to_write[i],
                    signals_to_write[i],
                    MSBG_FS,
                    floating_point=True,
                )


@hydra.main(config_path=".", config_name="config")
def run_calculate_SI(cfg: DictConfig) -> None:
    scenes_listeners = json.load(open(cfg["path"]["scenes_listeners_file"]))
    proc_folder = os.path.join(cfg["path"]["exp_folder"], "eval_signals")
    sii_file = os.path.join(cfg["path"]["exp_folder"], "sii.csv")
    csv_lines = [["scene", "listener", "sii"]]

    for scene in tqdm(scenes_listeners):
        for listener in scenes_listeners[scene]:
            logging.info(f"Running SI calculation: scene {scene}, listener {listener}")
            proc = read_signal(
                os.path.join(proc_folder, f"{scene}_{listener}_HL-output.wav")
            )
            clean = read_signal(
                os.path.join(
                    cfg["path"]["scenes_folder"], f"{scene}_target_anechoic.wav"
                )
            )
            ddf = read_signal(
                os.path.join(proc_folder, f"{scene}_{listener}_HLddf-output.wav")
            )

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
                cfg["mbstoi"]["fs"],
                cfg["mbstoi"]["gridcoarseness"],
            )
            csv_lines.append([scene, listener, sii])  # type: ignore

    with open(sii_file, "w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines:
            csv_writer.writerow(line)


if __name__ == "__main__":
    run_HL_processing()
    run_calculate_SI()
