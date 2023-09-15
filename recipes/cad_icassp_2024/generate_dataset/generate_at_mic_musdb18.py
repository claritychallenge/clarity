"""
Script creates the dataset for the ICASSP 2024 Grand Challenge.

It takes the music from the MUSDB18 dataset and applies the HRTF signals
to simulate the music at the microphone position.
The output is saved in the same format as the MUSDB18 dataset.

The script takes as input:
    - The metadata of the scenes.
    - The metadata of the music.
    - The metadata of the head positions.
    - The HRTF signals.
    - The music signals.

The script outputs:
    - The metadata of the music at the hearing aids microphone.
    - The music signals at the hearing aids microphone.
"""
from __future__ import annotations

# pylint: disable=import-error
import json
import logging
import warnings
from pathlib import Path

import hydra
import numpy as np
import pyloudnorm as pyln
from numpy import ndarray
from omegaconf import DictConfig
from scipy.signal import lfilter

from clarity.utils.file_io import read_signal, write_signal

logger = logging.getLogger(__name__)


def apply_hrtf(signal: ndarray, hrtf_left: ndarray, hrtf_right) -> ndarray:
    """Applies the Left and Right HRTF to a signal.

    Args:
        signal (ndarray): Signal.
        hrtf_left (ndarray): Left HRTF.
        hrtf_right (ndarray): Right HRTF.

    Returns:
        ndarray: Signal with applied HRTF.
    """
    output_left_ear = lfilter(hrtf_left[:, 0], 1, signal[:, 0])
    output_right_ear = lfilter(hrtf_left[:, 1], 1, signal[:, 0])

    output_left_ear += lfilter(hrtf_right[:, 0], 1, signal[:, 1])
    output_right_ear += lfilter(hrtf_right[:, 1], 1, signal[:, 1])

    return np.stack([output_left_ear, output_right_ear], axis=1)


def load_hrtf_signals(hrtf_path: str, hp: dict) -> tuple[ndarray, ndarray]:
    """Loads the HRTF signals for a given head position.

    Args:
        hrtf_path (str): Path to the HRTF signals.
        hp (dict): Head position.

    Returns:
        tuple(ndarray, ndarray): Left and right HRTF signals.
    """

    hp_left_path = (
        Path(hrtf_path) / f"{hp['mic']}-{hp['subject']}-n{abs(hp['left_angle'])}.wav"
    )
    hp_right_path = (
        Path(hrtf_path) / f"{hp['mic']}-{hp['subject']}-p{abs(hp['right_angle'])}.wav"
    )

    hp_left_signal = read_signal(hp_left_path)
    hp_right_signal = read_signal(hp_right_path)

    return hp_left_signal, hp_right_signal


def normalise_lufs_level(
    signal: ndarray, reference_signal: ndarray, sample_rate: float
) -> ndarray:
    """Normalises the signal to the LUFS level of the reference signal.

    Args:
        signal (ndarray): Signal to normalise.
        reference_signal (ndarray): Reference signal.
        sample_rate (float): Sample rate of the signal.

    Returns:
        ndarray: Normalised signal.
    """
    loudness_meter = pyln.Meter(int(sample_rate))

    signal_lufs = loudness_meter.integrated_loudness(signal)
    reference_signal_lufs = loudness_meter.integrated_loudness(reference_signal)

    gain = reference_signal_lufs - signal_lufs
    return pyln.normalize.loudness(signal, signal_lufs, signal_lufs + gain)


def find_precreated_samples(source_dir: str | Path) -> list[str]:
    """Finds music tracks created in a previous run.
    This avoids reprocessing them.

    Args:
        source_dir (str| Path): Source directory.

    Returns:
        list[str]: List of precreated samples.
    """
    if isinstance(source_dir, str):
        source_dir = Path(source_dir)

    if not source_dir.exists():
        return []

    return [f.name for f in source_dir.glob("*/*")]


@hydra.main(config_path="", config_name="config")
def run(cfg: DictConfig) -> None:
    """Main function of the script."""

    logger.info("Generating dataset for the ICASSP 2024 Grand Challenge.\n")
    logger.info(f"Processing music for scenes: {cfg.path.scene_file}")
    logger.info(f"Transforming music signals from: {cfg.path.music_dir}")
    logger.info(f"and save them to {cfg.path.output_music_dir}")

    # Load precraeted samples to avoid reprocessing them
    precreated_samples = find_precreated_samples(cfg.path.output_music_dir)
    if len(precreated_samples) > 0:
        logger.warning(f"Found {len(precreated_samples)} precreated samples.\n")

    # Load the scenes metadata
    with open(cfg.path.scene_file, encoding="utf-8") as f:
        scenes_metadata = json.load(f)

    # Load the music metadata
    with open(cfg.path.music_file, encoding="utf-8") as f:
        music_metadata = json.load(f)
        music_metadata = {m["Track Name"]: m for m in music_metadata}

    # Load the head positions metadata
    with open(cfg.path.head_positions_file, encoding="utf-8") as f:
        head_positions_metadata = json.load(f)

    # From the scenes, get the samples names and parameters
    toprocess_samples = {
        f"{v['music']}-{v['head_position']}": {
            "music": v["music"],
            "head_position": v["head_position"],
        }
        for _, v in scenes_metadata.items()
    }

    # create output metadata content
    out_music = {}
    for idx, sample in enumerate(toprocess_samples.items(), 1):
        sample_name, sample_detail = sample
        music = music_metadata[sample_detail["music"]]
        head_position = sample_detail["head_position"]

        out_music[sample_name] = {
            "Track Name": sample_name,
            "Split": music["Split"],
            "Path": f"{music['Split']}/{sample_name}",
            "Original Track Name": music["Track Name"],
            "Head Position": head_position,
        }

        if sample_name in precreated_samples:
            logger.info(
                f"[{idx}/{len(toprocess_samples)}] Skipping sample: {sample_name}"
            )
            continue

        scene_path = Path(cfg.path.output_music_dir) / music["Split"] / sample_name
        scene_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{idx}/{len(toprocess_samples)}] Creating sample: {sample_name}")

        hrtf_left, hrtf_rigth = load_hrtf_signals(
            cfg.path.hrtf_dir, head_positions_metadata[head_position]
        )

        for stem_name in ["mixture", "vocals", "drums", "bass", "other"]:
            music_signal = read_signal(
                Path(cfg.path.music_dir) / music["Path"] / f"{stem_name}.wav"
            )

            at_mic_signal = apply_hrtf(music_signal, hrtf_left, hrtf_rigth)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Possible clipped samples in output"
                )
                at_mic_signal = normalise_lufs_level(
                    at_mic_signal, music_signal, cfg.sample_rate
                )

            # Save the signal
            save_path = scene_path / f"{stem_name}.wav"
            write_signal(
                save_path, at_mic_signal, cfg.sample_rate, floating_point=False
            )

        precreated_samples.append(sample_name)

    # Save the metadata
    with open(cfg.path.output_music_file, "w", encoding="utf-8") as f:
        json.dump(out_music, f, indent=4)

    logger.info(f"Saved metadata to: {cfg.path.output_music_file}")
    logger.info("Done.")


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    run()
