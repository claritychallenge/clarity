import csv
import json
import logging
from pathlib import Path

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
    output_path = Path(cfg.path.exp_folder) / "eval_signals"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
    with open(cfg.path.listeners_file, encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)
    enhanced_folder = Path(cfg.path.enhanced_signals)

    # initialize ear
    ear = Ear(
        src_pos=cfg.MSBGEar.src_pos,
        sample_frequency=cfg.MSBGEar.fs,
        equiv_0db_spl=cfg.MSBGEar.equiv0dBSPL,
        ahr=cfg.MSBGEar.ahr,
    )

    for scene in tqdm(scenes_listeners):
        for listener in scenes_listeners[scene]:
            if enhanced_folder.exists():
                signal_file = enhanced_folder / f"{scene}_{listener}_HA-output.wav"
            # if no enhanced signals, use the unprocessed signal for si calculation
            else:
                signal_file = Path(cfg.path.scenes_folder) / f"{scene}_mixed_CH0.wav"

            # signals to write
            outfile_stem = f"{output_path}/{scene}_{listener}"
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
            mixture_signal = read_signal(
                Path(cfg.path.scenes_folder) / f"{scene}_mixed_CH0.wav"
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
            ddf_signal = np.zeros(np.shape(signal))
            ddf_signal[:, 0] = unit_impulse(len(signal), int(MSBG_FS / 2))
            ddf_signal[:, 1] = unit_impulse(len(signal), int(MSBG_FS / 2))

            # Get flat-0dB ear audiograms
            flat0dB_audiogram = Audiogram(cfs=cfs, levels=np.zeros(np.shape(cfs)))

            signals_to_write = [
                listen(ear, ddf_signal, flat0dB_audiogram, flat0dB_audiogram),
                listen(ear, signal, left_audiogram, right_audiogram),
                listen(ear, ddf_signal, left_audiogram, right_audiogram),
                listen(ear, mixture_signal, left_audiogram, right_audiogram),
            ]

            for signal, out_signal_file in zip(signals_to_write, signal_files_to_write):
                write_signal(
                    out_signal_file,
                    signal,
                    MSBG_FS,
                    floating_point=True,
                )


@hydra.main(config_path=".", config_name="config")
def run_calculate_SI(cfg: DictConfig) -> None:
    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
    proc_folder = Path(cfg.path.exp_folder) / "eval_signals"
    sii_file = Path(cfg.path.exp_folder) / "sii.csv"
    csv_lines = [["scene", "listener", "sii"]]

    for scene in tqdm(scenes_listeners):
        for listener in scenes_listeners[scene]:
            logging.info(f"Running SI calculation: scene {scene}, listener {listener}")
            proc = read_signal(proc_folder / f"{scene}_{listener}_HL-output.wav")
            clean = read_signal(
                Path(cfg.path.scenes_folder) / f"{scene}_target_anechoic.wav"
            )
            ddf = read_signal(proc_folder / f"{scene}_{listener}_HLddf-output.wav")

            # Calculate channel-specific unit impulse delay due to HL model
            # and audiograms
            delay = find_delay_impulse(ddf, initial_value=int(MSBG_FS / 2))
            if delay[0] != delay[1]:
                logging.info(f"Difference in delay of {delay[0] - delay[1]}.")
            max_delay = int(np.max(delay))

            # Allow for value lower than 1000 samples in case of unimpaired hearing
            if max_delay > 2000:
                logging.error("Error in delay calculation for signal time-alignment.")

            # Correct for delays by padding clean signals
            clean_pad = np.zeros((len(clean) + max_delay, 2))
            proc_pad = np.zeros((len(clean) + max_delay, 2))

            if len(proc_pad) < len(proc):
                raise ValueError("Padded processed signal is too short.")

            clean_pad[int(delay[0]) : int(len(clean) + int(delay[0])), 0] = clean[:, 0]
            clean_pad[int(delay[1]) : int(len(clean) + int(delay[1])), 1] = clean[:, 1]
            proc_pad[: len(proc)] = proc

            sii = mbstoi(
                clean_pad[:, 0],
                clean_pad[:, 1],
                proc_pad[:, 0],
                proc_pad[:, 1],
                cfg.mbstoi.fs,
                cfg.mbstoi.gridcoarseness,
            )
            csv_lines.append([scene, listener, sii])  # type: ignore

    with sii_file.open("w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines:
            csv_writer.writerow(line)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run_HL_processing()
    run_calculate_SI()