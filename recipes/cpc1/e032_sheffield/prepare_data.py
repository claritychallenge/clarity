import csv
import json
import logging
import os
import random

import hydra
import numpy as np
import soundfile as sf
from librosa import resample
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.evaluator.msbg.audiogram import Audiogram
from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import MSBG_FS, pad, read_signal, write_signal

logger = logging.getLogger(__name__)

targ_fs = 16000


def run_data_split(cfg, track):
    os.makedirs(os.path.join(cfg.path.exp_folder, "data_split"), exist_ok=True)
    scene_train_json = os.path.join(
        cfg.path.exp_folder, "data_split", f"scene_train_list{track}.json"
    )
    scene_dev_json = os.path.join(
        cfg.path.exp_folder, "data_split", f"scene_dev_list{track}.json"
    )
    if os.path.isfile(scene_train_json) and os.path.isfile(scene_dev_json):
        logger.info("Train set and dev set lists exist...")
        return

    scenes_dict = json.load(
        open(
            os.path.join(
                cfg.path.root,
                "clarity_CPC1_data_train/metadata",
                f"CPC1.{'train'+track}.json",
            )
        )
    )
    scene_list = []
    for item in scenes_dict:
        scene_list.append(item["scene"])
    scene_list = list(set(scene_list))
    scene_dev_list = random.sample(
        scene_list, int(np.floor(len(scene_list) * cfg.dev_percent))
    )
    scene_train_list = list(set(scene_list) - set(scene_dev_list))

    with open(scene_train_json, "w") as f:
        json.dump(scene_train_list, f)
    with open(scene_dev_json, "w") as f:
        json.dump(scene_dev_list, f)


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


def run_msbg_simulation(cfg, track):
    for split in ["train", "test"]:
        dataset = split + track
        dataset_folder = os.path.join(cfg.path.root, "clarity_CPC1_data_" + split)
        output_path = os.path.join(dataset_folder, "clarity_data/HA_outputs", dataset)
        scenes = json.load(
            open(os.path.join(dataset_folder, "metadata", f"CPC1.{dataset}.json"))
        )
        if split == "train":
            listener_audiograms = json.load(
                open(
                    os.path.join(
                        dataset_folder, "metadata", "listeners.CPC1_train.json"
                    )
                )
            )
        else:
            listener_audiograms = json.load(
                open(
                    os.path.join(dataset_folder, "metadata", "listeners.CPC1_all.json")
                )
            )

        # initialize ear
        ear = Ear(**cfg["MSBGEar"])

        for scene_dict in tqdm(scenes):
            scene = scene_dict["scene"]
            listener = scene_dict["listener"]
            system = scene_dict["system"]
            signal_file = os.path.join(output_path, f"{scene}_{listener}_{system}.wav")

            # signals to write
            outfile_stem = f"{output_path}/{scene}_{listener}_{system}"
            signal_files_to_write = [
                f"{outfile_stem}_HL-output.wav",
            ]
            # if all signals to write exist, pass
            if all([os.path.isfile(f) for f in signal_files_to_write]):
                continue
            signal = read_signal(signal_file)

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

            signals_to_write = [
                listen(ear, signal, left_audiogram, right_audiogram),
            ]
            for i in range(len(signals_to_write)):
                write_signal(
                    signal_files_to_write[i],
                    signals_to_write[i],
                    MSBG_FS,
                    floating_point=True,
                )


def generate_data_split(
    orig_data_json,
    orig_signal_folder,
    target_data_folder,
    data_split,
    split_data_list,
    if_msbg=False,
    if_ref=False,
):

    with open(orig_data_json, "r") as f:
        all_data_list = json.load(f)

    left_tgt_signal_folder = os.path.join(
        target_data_folder, orig_signal_folder.split("/")[-2] + "_left"
    )
    right_tgt_signal_folder = os.path.join(
        target_data_folder, orig_signal_folder.split("/")[-2] + "_right"
    )
    if if_msbg:
        left_tgt_signal_folder = left_tgt_signal_folder + "_msbg"
        right_tgt_signal_folder = right_tgt_signal_folder + "_msbg"
    if if_ref:
        left_tgt_signal_folder = left_tgt_signal_folder + "_ref"
        right_tgt_signal_folder = right_tgt_signal_folder + "_ref"
    os.makedirs(left_tgt_signal_folder, exist_ok=True)
    os.makedirs(right_tgt_signal_folder, exist_ok=True)

    csv_lines_left = [["ID", "duration", "wav", "spk_id", "wrd"]]
    csv_lines_right = [["ID", "duration", "wav", "spk_id", "wrd"]]
    csv_lines_binaural = [["ID", "duration", "wav", "spk_id", "wrd"]]

    left_csvfile = "left_" + data_split
    right_csvfile = "right_" + data_split
    binaural_csvfile = "binaural_" + data_split
    if if_msbg:
        left_csvfile += "_msbg"
        right_csvfile += "_msbg"
        binaural_csvfile += "_msbg"
    if if_ref:
        left_csvfile += "_ref"
        right_csvfile += "_ref"
        binaural_csvfile += "_ref"
    left_csvfile = os.path.join(target_data_folder, left_csvfile + ".csv")
    right_csvfile = os.path.join(target_data_folder, right_csvfile + ".csv")
    binaural_csvfile = os.path.join(target_data_folder, binaural_csvfile + ".csv")

    for item in tqdm(all_data_list):
        snt_id = item["signal"]
        if snt_id.split("_")[0] in split_data_list or data_split == "test":
            if if_msbg:
                wav_file = os.path.join(orig_signal_folder, snt_id + "_HL-output.wav")
            elif if_ref:
                wav_file = os.path.join(
                    orig_signal_folder,
                    "../..",
                    "scenes",
                    snt_id.split("_")[0] + "_target_anechoic.wav",
                )
            else:
                wav_file = os.path.join(orig_signal_folder, snt_id + ".wav")
            wav_file_left = os.path.join(left_tgt_signal_folder, snt_id + ".wav")
            wav_file_right = os.path.join(right_tgt_signal_folder, snt_id + ".wav")
            if os.path.isfile(wav_file_left) and os.path.isfile(wav_file_right):
                continue

            spk_id = item["listener"]  # should be listener_id
            wrds = item["prompt"].upper()

            utt, orig_fs = sf.read(wav_file)
            utt_16k = resample(np.array(utt).transpose(), orig_fs, targ_fs).transpose()
            duration = (
                len(utt_16k[2 * targ_fs :, 0]) / targ_fs
            )  # Get rid of the first two seconds, as there is no speech
            sf.write(wav_file_left, utt_16k[2 * targ_fs :, 0], targ_fs)
            sf.write(wav_file_right, utt_16k[2 * targ_fs :, 1], targ_fs)

            csv_lines_left.append([snt_id, str(duration), wav_file_left, spk_id, wrds])
            csv_lines_right.append(
                [snt_id, str(duration), wav_file_right, spk_id, wrds]
            )
            csv_lines_binaural.append(
                ["left_" + snt_id, str(duration), wav_file_left, spk_id, wrds]
            )
            csv_lines_binaural.append(
                ["right_" + snt_id, str(duration), wav_file_right, spk_id, wrds]
            )

    with open(left_csvfile, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines_left:
            csv_writer.writerow(line)

    with open(right_csvfile, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines_right:
            csv_writer.writerow(line)

    with open(binaural_csvfile, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines_binaural:
            csv_writer.writerow(line)


def run_signal_generation_train(cfg, track):
    train_json_path = os.path.join(
        cfg.path.cpc1_train_data, f"metadata/CPC1.{'train'+track}.json"
    )
    train_signal_folder = os.path.join(
        cfg.path.cpc1_train_data, f"clarity_data/HA_outputs/{'train'+track}/"
    )
    target_folder = os.path.join(cfg.path.exp_folder, "cpc1_asr_data" + track)
    lists_to_generate = [
        os.path.join(
            cfg.path.exp_folder, "data_split", f"scene_train_list{track}.json"
        ),
        os.path.join(cfg.path.exp_folder, "data_split", f"scene_dev_list{track}.json"),
    ]

    datasets_to_generate = []
    for i in range(len(lists_to_generate)):
        with open(lists_to_generate[i], "r") as f:
            datasets_to_generate.append(json.load(f))
            f.close()

    for i in range(len(lists_to_generate)):
        # generate_data_split(
        #     train_json_path,
        #     train_signal_folder,
        #     target_folder,
        #     os.path.basename(lists_to_generate[i]).split("_")[1],
        #     datasets_to_generate[i],
        # )

        generate_data_split(
            train_json_path,
            train_signal_folder,
            target_folder,
            os.path.basename(lists_to_generate[i]).split("_")[1],
            datasets_to_generate[i],
            if_msbg=True,
        )

        generate_data_split(
            train_json_path,
            train_signal_folder,
            target_folder,
            os.path.basename(lists_to_generate[i]).split("_")[1],
            datasets_to_generate[i],
            if_ref=True,
        )


def run_signal_generation_test(cfg, track):
    test_json_path = os.path.join(
        cfg.path.cpc1_test_data, f"metadata/CPC1.{'test'+track}.json"
    )
    test_signal_folder = os.path.join(
        cfg.path.cpc1_test_data, f"clarity_data/HA_outputs/{'test'+track}/"
    )
    target_folder = os.path.join(cfg.path.exp_folder, "cpc1_asr_data" + track)

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

    logger.info("Split all training data into train set and dev set.")
    run_data_split(cfg, track)
    logger.info("Running MSGB simulation.")
    run_msbg_simulation(cfg, track)
    logger.info("Begin generating signals for ASR for cpc1_train_data.")
    run_signal_generation_train(cfg, track)
    logger.info("Begin generating signals for ASR for cpc1_test_data.")
    run_signal_generation_test(cfg, track)


if __name__ == "__main__":
    run()
