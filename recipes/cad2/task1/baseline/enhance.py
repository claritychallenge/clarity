"""Baseline enhancement for CAD2 task1."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from numpy import ndarray

import torch
from torchaudio.transforms import Fade


from omegaconf import DictConfig

from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal, write_signal
from clarity.utils.signal_processing import resample
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

    def __call__(self, signal, listener: Listener):
        return signal


def separate_sources(
    model: torch.nn.Module,
    mix: torch.Tensor | ndarray,
    sample_rate: int,
    segment: float = 10.0,
    overlap: float = 0.1,
    number_sources: int = 4,
    device: torch.device | str | None = None,
):
    """
    Apply model to a given mixture.
    Use fade, and add segments together in order to add model segment by segment.

    Args:
        model (torch.nn.Module): model to use for separation
        mix (torch.Tensor): mixture to separate, shape (batch, channels, time)
        sample_rate (int): sampling rate of the mixture
        segment (float): segment length in seconds
        overlap (float): overlap between segments, between 0 and 1
        number_sources (int): number of sources to separate
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.

    Returns:
        torch.Tensor: estimated sources

    Based on https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html
    """
    device = mix.device if device is None else torch.device(device)
    mix = torch.as_tensor(mix, dtype=torch.float, device=device)

    if mix.ndim == 1:
        # one track and mono audio
        mix = mix.unsqueeze(0)
    elif mix.ndim == 2:
        # one track and stereo audio
        mix = mix.unsqueeze(0)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, number_sources, channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0

    return final


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
        alpha = alphas[scene["alpha"]]

        # Load the music
        music = read_signal(
            Path(config.path.music_dir)
            / songs[scene["segment_id"]]["path"]
            / "mixture.wav",
            offset=int(
                songs[scene["segment_id"]]["start_time"] * config.input_sample_rate
            ),
            n_samples=int(
                (
                    songs[scene["segment_id"]]["end_time"]
                    - songs[scene["segment_id"]]["start_time"]
                )
                * config.input_sample_rate
            ),
        )

        # Separate the music
        sources = separate_sources(
            separation_model,
            music.T,
            device=device,
            **config.separator.separation,
        )
        sources = sources.squeeze(0).cpu().detach().numpy()

        vocals, accompaniment = sources

        # Enhance the vocals
        enhanced_vocals = mbc(vocals, listener)

        # Downmix to stereo
        enhanced_signal = enhanced_vocals * alpha + accompaniment * (1 - alpha)

        # Save the enhanced music
        enhanced_path = enhanced_folder / f"{scene_id}_{listener_id}_A{alpha}.wav"
        write_signal(
            enhanced_path,
            resample(
                enhanced_signal.T, config.input_sample_rate, config.output_sample_rate
            ),
            config.output_sample_rate,
            floating_point=False,
        )

        print(f"Enhanced music saved to {enhanced_path}")


if __name__ == "__main__":
    enhance()
