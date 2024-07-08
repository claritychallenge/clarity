"""Baseline enhancement for CAD2 task1."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from numpy import ndarray
from omegaconf import DictConfig
from torchaudio.transforms import Fade

from clarity.enhancer.multiband_compressor import MultibandCompressor
from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal
from clarity.utils.flac_encoder import save_flac_signal
from recipes.cad2.task1.ConvTasNet.local.tasnet import ConvTasNetStereo

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


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
        mix = mix.unsqueeze(0).unsqueeze(0)
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

    # Can define a standard 'small_test' with just 1/50 of the data
    if small_test:
        scene_listener_pairs = scene_listener_pairs[::400]

    return scene_listener_pairs


def downmix_signal(
    vocals: ndarray,
    accompaniment: ndarray,
    beta: float,
) -> ndarray:
    """
    Downmix the vocals and accompaniment to stereo.
    Args:
        vocals (np.ndarray): Vocal signal.
        accompaniment (np.ndarray): Accompaniment signal.
        beta (float): Downmix parameter.

    Returns:
        np.ndarray: Downmixed signal.

    Notes:
        When beta is 0, the downmix is the accompaniment.
        When beta is 1, the downmix is the vocals.
    """
    # Vocals +1Db, Accompaniment -1Db
    return (
        vocals * 10 ** (1 / 20) * (beta / 2 + 0.5)
        + accompaniment * 10 ** (-1 / 20) * (1 - beta) / 2
    )


@hydra.main(config_path="", config_name="config", version_base=None)
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The system decomposes the music into vocals and accompaniment.
    Then, vocals are enhanced according to alpha values.
    Finally, the music is amplified according hearing loss and downmix to stereo.

    Args:
        config (dict): Dictionary of configuration options for enhancing music.

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

    # Load compressor params
    with Path(config.path.enhancer_params_file).open("r", encoding="utf-8") as file:
        enhancer_params = json.load(file)

    # Load separation model
    separation_model = load_separation_model(config.separator.causality, device)

    # create hearing aid
    enhancer = MultibandCompressor(
        crossover_frequencies=config.enhancer.crossover_frequencies,
        sample_rate=config.input_sample_rate,
    )

    # Make the list of scene-listener pairings to process
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )

    # Process each scene-listener pair
    for idx, scene_listener_ids in enumerate(scene_listener_pairs, 1):
        logger.info(
            f"[{idx:04d}/{len(scene_listener_pairs):04d}] Processing scene-listener"
            f" pair: {scene_listener_ids}"
        )

        scene_id, listener_id = scene_listener_ids
        scene = scenes[scene_id]

        # This recipe is not using the listener metadata
        # listener = listener_dict[listener_id]

        alpha = alphas[scene["alpha"]]

        # Load the music
        music = read_signal(
            Path(config.path.music_dir)
            / songs[scene["segment_id"]]["path"]
            / "mixture.wav",
            offset=int(
                songs[scene["segment_id"]]["start_time"] * config.input_sample_rate
            ),
            offset_is_samples=True,
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
        vocals, accompaniment = sources.squeeze(0).cpu().detach().numpy()

        # Get the listener's compressor params
        mbc_params_listener = {"left": {}, "right": {}}

        for ear in ["left", "right"]:
            mbc_params_listener[ear]["release"] = config.enhancer.release
            mbc_params_listener[ear]["attack"] = config.enhancer.attack
            mbc_params_listener[ear]["threshold"] = config.enhancer.threshold
        mbc_params_listener["left"]["ratio"] = enhancer_params[listener_id]["cr_l"]
        mbc_params_listener["right"]["ratio"] = enhancer_params[listener_id]["cr_r"]
        mbc_params_listener["left"]["makeup_gain"] = enhancer_params[listener_id][
            "gain_l"
        ]
        mbc_params_listener["right"]["makeup_gain"] = enhancer_params[listener_id][
            "gain_r"
        ]

        # Downmix to stereo
        enhanced_signal = downmix_signal(vocals, accompaniment, beta=alpha)

        # Apply Amplification
        enhancer.set_compressors(**mbc_params_listener["left"])
        left_enhanced = enhancer(signal=enhanced_signal[0, :])

        enhancer.set_compressors(**mbc_params_listener["right"])
        right_enhanced = enhancer(signal=enhanced_signal[1, :])

        enhanced_signal = np.stack((left_enhanced[0], right_enhanced[0]), axis=1)

        # Save the enhanced music
        filename = enhanced_folder / f"{scene_id}_{listener_id}_A{alpha}_remix.flac"
        filename.parent.mkdir(parents=True, exist_ok=True)

        save_flac_signal(
            signal=enhanced_signal,
            filename=filename,
            signal_sample_rate=config.input_sample_rate,
            output_sample_rate=config.remix_sample_rate,
            do_clip_signal=True,
            do_soft_clip=config.soft_clip,
        )

    logger.info("Done!")


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
