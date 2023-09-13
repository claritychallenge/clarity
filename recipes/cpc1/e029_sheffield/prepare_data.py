"""Data preparation for the E029 Sheffield CPC1 recipe."""
import csv
import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
from librosa import resample  # pylint: disable=no-name-in-module
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import MSBG_FS, pad
from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal, write_signal

logger = logging.getLogger(__name__)

target_sample_rate = 16000


def run_data_split(cfg, track):
    data_split_dir = Path(cfg.path.exp_folder) / "data_split"
    data_split_dir.mkdir(parents=True, exist_ok=True)
    scene_train_json = data_split_dir / f"scene_train_list{track}.json"
    scene_dev_json = data_split_dir / f"scene_dev_list{track}.json"

    if scene_train_json.exists() and scene_dev_json.exists():
        logger.info("Train set and dev set lists exist...")
        return

    file_path = (
        Path(cfg.path.root)
        / f"clarity_CPC1_data_train/metadata/CPC1.{'train'+track}.json"
    )

    with file_path.open("r", encoding="utf-8") as fp:
        scenes_dict = json.load(fp)
    scene_list = []
    for item in scenes_dict:
        scene_list.append(item["scene"])
    scene_list = list(set(scene_list))
    scene_dev_list = random.sample(
        scene_list, int(np.floor(len(scene_list) * cfg.dev_percent))
    )
    scene_train_list = list(set(scene_list) - set(scene_dev_list))

    with scene_train_json.open("w", encoding="utf-8") as fp:
        json.dump(scene_train_list, fp)
    with scene_dev_json.open("w", encoding="utf-8") as fp:
        json.dump(scene_dev_list, fp)


def listen(ear, signal, listener: Listener):
    """
    Generate MSBG processed signal
    :param ear: MSBG ear
    :param wav: binaural signal
    :param listener: listener object
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


def run_msbg_simulation(cfg, track):
    for split in ["train", "test"]:
        dataset = split + track
        dataset_folder = Path(cfg.path.root) / f"clarity_CPC1_data_{split}"
        output_path = dataset_folder / "clarity_data/HA_outputs" / dataset
        file_path = dataset_folder / f"metadata/CPC1.{dataset}.json"
        with file_path.open("r", encoding="utf-8") as fp:
            scenes = json.load(fp)
        if split == "train":
            file_path = Path(dataset_folder) / "metadata/listeners.CPC1_train.json"
        else:
            file_path = Path(dataset_folder) / "metadata/listeners.CPC1_all.json"
        listener_dict = Listener.load_listener_dict(file_path)

        # initialize ear
        ear = Ear(**cfg["MSBGEar"])

        for scene_dict in tqdm(scenes):
            scene = scene_dict["scene"]
            listener_id = scene_dict["listener"]
            system = scene_dict["system"]
            signal_file = output_path / f"{scene}_{listener_id}_{system}.wav"

            # signals to write
            outfile_stem = output_path / f"{scene}_{listener_id}_{system}"
            signal_files_to_write = [Path(f"{outfile_stem}_HL-output.wav")]
            # if all signals to write exist, pass
            if all(f.exists() for f in signal_files_to_write):
                continue
            signal = read_signal(signal_file)

            listener = listener_dict[listener_id]
            signals_to_write = [listen(ear, signal, listener)]

            for signal, signal_file in zip(signals_to_write, signal_files_to_write):
                write_signal(
                    signal_file,
                    signal,
                    MSBG_FS,
                    floating_point=True,
                )


def generate_data_split(
    orig_data_json: Path,
    orig_signal_folder: Path,
    target_data_folder: Path,
    data_split,
    split_data_list,
    if_msbg=False,
    if_ref=False,
):
    with orig_data_json.open("r", encoding="utf-8") as fp:
        all_data_list = json.load(fp)

    ext = ""
    if if_msbg:
        ext = "_msbg"
    elif if_ref:
        ext = "_ref"

    left_tgt_signal_folder = target_data_folder / (
        str(orig_signal_folder).split("/")[-2] + "_left" + ext
    )

    right_tgt_signal_folder = target_data_folder / (
        str(orig_signal_folder).split("/")[-2] + "_right" + ext
    )

    left_tgt_signal_folder.mkdir(parents=True, exist_ok=True)
    right_tgt_signal_folder.mkdir(parents=True, exist_ok=True)

    csv_lines_left = [["ID", "duration", "wav", "spk_id", "wrd"]]
    csv_lines_right = [["ID", "duration", "wav", "spk_id", "wrd"]]
    csv_lines_binaural = [["ID", "duration", "wav", "spk_id", "wrd"]]

    left_csvfile = target_data_folder / f"left_{data_split}{ext}.csv"
    right_csvfile = target_data_folder / f"right_{data_split}{ext}.csv"
    binaural_csvfile = target_data_folder / f"binaural_{data_split}{ext}.csv"

    for item in tqdm(all_data_list):
        snt_id = item["signal"]
        if snt_id.split("_")[0] in split_data_list or data_split == "test":
            if if_msbg:
                wav_file = orig_signal_folder / f"{snt_id}_HL-output.wav"
            elif if_ref:
                wav_file = orig_signal_folder / (
                    f"../../scenes/{snt_id.split('_')[0]}_target_anechoic.wav"
                )
            else:
                wav_file = orig_signal_folder / f"{snt_id}.wav"
            wav_file_left = left_tgt_signal_folder / f"{snt_id}.wav"
            wav_file_right = right_tgt_signal_folder / f"{snt_id}.wav"
            if wav_file_left.exists() and wav_file_right.exists():
                continue

            spk_id = item["listener"]  # should be listener_id
            wrds = item["prompt"].upper()

            utt, orig_fs = sf.read(wav_file)
            utt_16k = resample(
                np.array(utt).transpose(), orig_sr=orig_fs, target_sr=target_sample_rate
            ).transpose()
            duration = (
                len(utt_16k[2 * target_sample_rate :, 0]) / target_sample_rate
            )  # Get rid of the first two seconds, as there is no speech
            sf.write(
                wav_file_left, utt_16k[2 * target_sample_rate :, 0], target_sample_rate
            )
            sf.write(
                wav_file_right, utt_16k[2 * target_sample_rate :, 1], target_sample_rate
            )

            csv_lines_left.append(
                [snt_id, str(duration), str(wav_file_left), spk_id, wrds]
            )
            csv_lines_right.append(
                [snt_id, str(duration), str(wav_file_right), spk_id, wrds]
            )
            csv_lines_binaural.append(
                ["left_" + snt_id, str(duration), str(wav_file_left), spk_id, wrds]
            )
            csv_lines_binaural.append(
                ["right_" + snt_id, str(duration), str(wav_file_right), spk_id, wrds]
            )

    with left_csvfile.open(mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines_left:
            csv_writer.writerow(line)

    with right_csvfile.open(mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines_right:
            csv_writer.writerow(line)

    with binaural_csvfile.open(mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines_binaural:
            csv_writer.writerow(line)


def run_signal_generation_train(cfg, track):
    train_json_path = (
        Path(cfg.path.cpc1_train_data) / f"metadata/CPC1.{'train'+track}.json"
    )

    train_signal_folder = (
        Path(cfg.path.cpc1_train_data) / f"clarity_data/HA_outputs/{'train'+track}/"
    )

    target_folder = Path(cfg.path.exp_folder) / f"cpc1_asr_data{track}"
    lists_to_generate = [
        Path(cfg.path.exp_folder) / f"data_split/scene_train_list{track}.json",
        Path(cfg.path.exp_folder) / f"data_split/scene_dev_list{track}.json",
    ]

    datasets_to_generate = []
    for list_to_generate in lists_to_generate:
        with list_to_generate.open("r", encoding="utf-8") as fp:
            datasets_to_generate.append(json.load(fp))

    for list_to_generate, dataset in zip(lists_to_generate, datasets_to_generate):
        # generate_data_split(
        #     train_json_path,
        #     train_signal_folder,
        #     target_folder,
        #     list_to_generate.name.split("_")[1],
        #     datasets_to_generate[i],
        # )

        generate_data_split(
            train_json_path,
            train_signal_folder,
            target_folder,
            list_to_generate.name.split("_")[1],
            dataset,
            if_msbg=True,
        )

        generate_data_split(
            train_json_path,
            train_signal_folder,
            target_folder,
            list_to_generate.name.split("_")[1],
            dataset,
            if_ref=True,
        )


def run_signal_generation_test(cfg, track):
    test_json_path = (
        Path(cfg.path.cpc1_test_data) / f"metadata/CPC1.{'test'+track}.json"
    )
    test_signal_folder = (
        Path(cfg.path.cpc1_test_data) / f"clarity_data/HA_outputs/{'test'+track}/"
    )

    target_folder = Path(cfg.path.exp_folder) / f"cpc1_asr_data{track}"

    # generate_data_split(
    #     test_json_path, test_signal_folder, target_folder, "test", [],
    # )

    generate_data_split(
        test_json_path, test_signal_folder, target_folder, "test", [], if_msbg=True
    )

    generate_data_split(
        test_json_path, test_signal_folder, target_folder, "test", [], if_ref=True
    )


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    if cfg.cpc1_track == "open":
        track = "_indep"
    elif cfg.cpc1_track == "closed":
        track = ""
    else:
        logger.error("cpc1_track has to be closed or open")
        raise ValueError("cpc1_track has to be closed or open")

    logger.info("Split all training data into train set and dev set.")
    run_data_split(cfg, track)
    logger.info("Running MSGB simulation.")
    run_msbg_simulation(cfg, track)
    logger.info("Begin generating signals for ASR for cpc1_train_data.")
    run_signal_generation_train(cfg, track)
    logger.info("Begin generating signals for ASR for cpc1_test_data.")
    run_signal_generation_test(cfg, track)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()
