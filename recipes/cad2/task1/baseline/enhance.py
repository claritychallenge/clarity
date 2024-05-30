"""Baseline enhancement for CAD2 task1."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch

from omegaconf import DictConfig

from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal
from recipes.cad2.task1.ConvTasNet.local.tasnet import ConvTasNetStereo


class MultibandCompressor:
    def __init__(
        self,
        crossover_frequencies: float = 2000.0,
        order: int = 4,
        sample_rate: float = 44100,
        compressors_params: dict | None = None,
    ):
        self.xover_freqs = np.array([crossover_frequencies])
        self.sample_rate = sample_rate
        self.order = order
        self.compressors_params = compressors_params

    def __call__(self, signal):
        return signal


def get_device(device: str) -> tuple:
    """Get the Torch device.

    Args:
        device (str): device type, e.g. "cpu", "gpu0", "gpu1", etc.

    Returns:
        torch.device: torch.device() appropiate to the hardware available.
        str: device type selected, e.g. "cpu", "cuda".
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"

    if device.startswith("gpu"):
        device_index = int(device.replace("gpu", ""))
        if device_index > torch.cuda.device_count():
            raise ValueError(f"GPU device index {device_index} is not available.")
        return torch.device(f"cuda:{device_index}"), "cuda"

    if device == "cpu":
        return torch.device("cpu"), "cpu"

    raise ValueError(f"Unsupported device type: {device}")


def load_separation_model(causality: str, device: torch.device) -> ConvTasNetStereo:
    """
    Load the separation model.
    Args:
        causality (str): Causality of the model (causal or noncausal).
        device (torch.device): Device to load the model.

    Returns:
        model: Separation model.
    """
    if causality == "causal":
        model = ConvTasNetStereo.from_pretrained(
            "cadenzachallenge/ConvTasNet_LyricsSeparation_Causal",
            force_download=True,
        ).to(device)
    else:
        model = ConvTasNetStereo.from_pretrained(
            "cadenzachallenge/ConvTasNet_LyricsSeparation_NonCausal"
        ).to(device)
    return model


def make_scene_listener_list(scenes_listeners: dict, small_test: bool = False) -> list:
    """Make the list of scene-listener pairing to process

    Args:
        scenes_listeners (dict): Dictionary of scenes and listeners.
        small_test (bool): Whether to use a small test set.

    Returns:
        list: List of scene-listener pairings.

    """
    scene_listener_pairs = [
        (scene, listener)
        for scene in scenes_listeners
        for listener in scenes_listeners[scene]
    ]

    # Can define a standard 'small_test' with just 1/15 of the data
    if small_test:
        scene_listener_pairs = scene_listener_pairs[::15]

    return scene_listener_pairs


@hydra.main(config_path="", config_name="config", version_base="1.1")
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The system decomposes the music into vocals and accompaniment.
    Then, vocals are enhanced according to alpha values.
    Finally, the music is amplified according hearing loss and downmix to stereo.
    Args:
        config (dict): Dictionary of configuration options for enhancing music.

    Returns 8 stems for each song:
        - left channel vocal, drums, bass, and other stems
        - right channel vocal, drums, bass, and other stems
    """
    if config.separator.causality not in ["causal", "noncausal"]:
        raise ValueError(
            f"Causality must be causal or noncausal, {config.separator.causality} was"
            " provided."
        )

    device, _ = get_device(config.separator.device)

    # Set folder to save the enhanced music
    enhanced_folder = Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    # Load listener dictionary
    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    # Load alphas
    with Path(config.path.alphas_file).open("r", encoding="utf-8") as file:
        alphas = json.load(file)

    # Load scenes
    with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    # Load scene-listeners
    with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    # Load songs
    with Path(config.path.musics_file).open("r", encoding="utf-8") as file:
        songs = json.load(file)

    # Load separation model
    separation_model = load_separation_model(config.separator.causality, device)

    # Load multiband compressor for amplification
    mbc = MultibandCompressor(
        crossover_frequencies=config.compressor.crossover_filter.frequencies,
        order=config.compressor.crossover_filter.order,
        sample_rate=config.compressor.crossover_filter.sample_rate,
        compressors_params=config.compressor.parameters,
    )

    # Make the list of scene-listener pairings to process
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )

    # Process each scene-listener pair
    for scene_id, listener_id in scene_listener_pairs:
        scene = scenes[scene_id]
        listener = listener_dict[listener_id]

        # Load the music
        music = read_signal(
            Path(config.path.music_dir) / songs[scene["track_name"]],
            offset=int(scene["start_time"] * config.input_sample_rate),
            n_samples=int(
                (scene["end_time"] - scene["start_time"]) * config.input_sample_rate
            ),
        )

        # Separate the music
        separated = separation_model(music, device=device)

        # Enhance the vocals
        enhanced_vocals = separated["vocals"]
        enhanced_vocals = mbc(enhanced_vocals)

        # Amplify the music
        amplified = listener.amplify(separated, alphas[scene["alpha"]])

        # Downmix to stereo
        stereo = listener.downmix(amplified)

        # Save the enhanced music
        enhanced_path = enhanced_folder / f"{scene_id}_{listener_id}.wav"
        stereo.save(enhanced_path)

        print(f"Enhanced music saved to {enhanced_path}")


if __name__ == "__main__":
    enhance()
