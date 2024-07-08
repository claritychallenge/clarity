"""
Module take the data downloaded from Zenodo and rearrange the data for the challenge.

The module expects the data to be downloaded from Zenodo and stored in the
following structure:

- zenodo_download_path
    - EnsembleSet.zip
    - CadenzaWoodwind.zip
    - metadata.zip
    - URMP.zip
    - BACH10.zip
    - CadenzaWoodwind.json
    - EnsembleSet.json
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import hydra
import librosa
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.utils.flac_encoder import save_flac_signal


def create_audio_train(cfg):
    """Create the audio training dataset."""
    audio_path = Path(cfg.path.music_dir)
    audio_path.mkdir(parents=True, exist_ok=True)

    zenodo_download_path = Path(cfg.path.zenodo_download_path)
    train_metadata_path = zenodo_download_path / "metadata" / "music_tracks.train.json"

    with open(train_metadata_path, "r") as f:
        train_metadata = json.load(f)

    for track, track_info in tqdm(
        train_metadata.items(), desc="Copying training audio"
    ):
        dataset = track_info["dataset"]
        for source in track_info["stems_path"]:
            target = audio_path / "train" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy(
                    zenodo_download_path / dataset / source,
                    audio_path / "train" / source,
                )
        if not (audio_path / "train" / track_info["mixture_path"]).exists():
            shutil.copy(
                zenodo_download_path / dataset / track_info["mixture_path"],
                audio_path / "train" / track_info["mixture_path"],
            )


def create_audio_valid(cfg):
    """Create the audio validation dataset"""
    audio_path = Path(cfg.path.music_dir)
    audio_path.mkdir(parents=True, exist_ok=True)

    zenodo_download_path = Path(cfg.path.zenodo_download_path)
    valid_metadata_path = zenodo_download_path / "metadata" / "music_tracks.valid.json"
    valid_samples_path = (
        zenodo_download_path / "metadata" / "music.valid.to_generate.json"
    )

    with open(valid_metadata_path, "r") as f:
        valid_metadata = json.load(f)

    with open(valid_samples_path, "r") as f:
        valid_samples = json.load(f)

    for track, track_info in tqdm(
        valid_samples.items(), desc="Creating validation audio"
    ):
        mixture = []
        for source, source_info in track_info.items():
            if source == "mixture":
                mixture_target = audio_path / source_info["track"]
                continue

            track_path = (
                zenodo_download_path
                / source_info["source_dataset"]
                / source_info["track"]
            )
            target = (
                audio_path
                / "valid"
                / track
                / "/".join(source_info["track"].replace("/Mix_1", "").split("/")[1:])
            )
            target.parent.mkdir(parents=True, exist_ok=True)

            start = source_info["start"]
            duration = source_info["duration"]
            signal, sr = librosa.load(
                track_path, sr=None, offset=start, duration=duration, mono=False
            )
            mixture.append(signal)

            save_flac_signal(
                signal.T,
                target,
                sr,
                sr,
            )

        mixture = np.stack(mixture)
        mixture = mixture.sum(axis=0)
        save_flac_signal(
            mixture.T,
            mixture_target,
            sr,
            sr,
        )


@hydra.main(config_path="", config_name="config", version_base=None)
def prepare_cad2_dataset(config: DictConfig) -> None:
    """
    Prepare the dataset for the challenge.
    """
    # Zenodo data path
    zenodo_path = Path(config.path.zenodo_download_path)

    for directory in [
        "CadenzaWoodwind",
        "EnsembleSet_Mix_1",
        "metadata",
        "URMP",
        "BACH10",
    ]:
        if not (zenodo_path / directory.replace("_Mix_1", "")).exists():
            if (zenodo_path / f"{directory}.zip").exists():
                # Unzip the file
                with zipfile.ZipFile(zenodo_path / f"{directory}.zip", "r") as zip_ref:
                    for member in tqdm(
                        zip_ref.infolist(), desc=f"Extracting {directory}"
                    ):
                        zip_ref.extract(member, zenodo_path)
            else:
                print(f"Skipping {directory} file. {directory}.zip not found")
        else:
            print(f"Directory {directory} already exists")

    # Create the root path if it doesn't exist
    root_path = Path(config.path.root)
    root_path.mkdir(parents=True, exist_ok=True)

    # Create train audio directory
    create_audio_train(config)

    # Create valid audio directory
    create_audio_valid(config)


if __name__ == "__main__":
    prepare_cad2_dataset()
