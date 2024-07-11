""" Run the dummy enhancement. """

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

# pylint: disable=import-error
import hydra
import numpy as np
import torch
from numpy import ndarray
from omegaconf import DictConfig
from torchaudio.transforms import Fade

from clarity.enhancer.multiband_compressor import MultibandCompressor
from clarity.utils.audiogram import Listener
from clarity.utils.flac_encoder import read_flac_signal, save_flac_signal
from clarity.utils.source_separation_support import get_device
from recipes.cad2.task2.baseline.evaluate import (
    adjust_level,
    apply_gains,
    make_scene_listener_list,
    remix_stems,
)
from recipes.cad2.task2.ConvTasNet.local.tasnet import ConvTasNetStereo

logger = logging.getLogger(__name__)


def check_repeated_source(gains: dict, source_list: dict) -> dict:
    """Check if mixture has 2 voices of the same instrument.
    Apply average gain to both voices.

    Args:
        gains (dict): Dictionary of original gains.
        source_list (dict): Dictionary of sources in mixture.

    Returns:
        dict: Dictionary of modified gains.
    """
    count_dict = Counter(source_list.values())
    two_voices = [key for key, value in source_list.items() if count_dict[value] > 1]
    two_voices_gain = [gain for source, gain in gains.items() if source in two_voices]
    two_voices_gain = np.mean(two_voices_gain)

    new_gains = {}
    for key, value in gains.items():
        if key in two_voices:
            new_gains[key] = two_voices_gain
        else:
            new_gains[key] = value
    return new_gains


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


# pylint: disable=unused-argument
def decompose_signal(
    model: dict[str, ConvTasNetStereo],
    signal: ndarray,
    signal_sample_rate: int,
    device: torch.device,
    sources_list: dict,
    listener: Listener,
) -> dict[str, ndarray]:
    """
    Decompose the signal into the estimated sources.

    The listener is ignored by the baseline system as it
     is not performing personalised decomposition.

    Args:
        model (dict): Dictionary of separation models.
        model_sample_rate (int): Sample rate of the separation model.
        signal (ndarray): Signal to decompose.
        signal_sample_rate (int): Sample rate of the signal.
        device (torch.device): Device to use for separation.
        sources_list (dict): List of sources to separate.
        listener (Listener): Listener audiogram.


    Returns:
        dict: Dictionary of estimated sources.
    """
    est_sources = {}
    for idx, source in enumerate(sources_list, 1):
        sources = separate_sources(
            model=model[sources_list[source]],
            mix=signal,
            sample_rate=signal_sample_rate,
            number_sources=2,
            device=device,
        )
        target, accompaniment = sources.squeeze(0).cpu().detach().numpy()
        target += accompaniment * 0.15
        est_sources[f"source_{idx}"] = target.T
    return est_sources


def load_separation_model(
    causality: str, device: torch.device
) -> dict[str, ConvTasNetStereo]:
    """
    Load the separation model.
    Args:
        causality (str): Causality of the model (causal or noncausal).
        device (torch.device): Device to load the model.

    Returns:
        model: Separation model.
    """
    models = {}
    causal = {"causal": "Causal", "noncausal": "NonCausal"}

    for instrument in [
        "Bassoon",
        "Cello",
        "Clarinet",
        "Flute",
        "Oboe",
        "Sax",
        "Viola",
        "Violin",
    ]:
        logger.info(
            "Loading model "
            f"cadenzachallenge/ConvTasNet_{instrument}_{causal[causality]}"
        )
        models[instrument] = ConvTasNetStereo.from_pretrained(
            f"cadenzachallenge/ConvTasNet_{instrument}_{causal[causality]}",
            force_download=True,
        ).to(device)
    return models


def process_remix_for_listener(
    signal: ndarray, enhancer: MultibandCompressor, enhancer_params: dict, listener
) -> ndarray:
    """Process the stems from sources.

    Args:

    Returns:
        ndarray: Processed signal.
    """

    output = []
    for side, ear in enumerate(["left", "right"]):
        enhancer.set_compressors(**enhancer_params[ear])
        output.append(enhancer(signal[:, side]))

    return np.vstack(output).T


@hydra.main(config_path="", config_name="config", version_base=None)
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The system decomposes the music into the estimated sources.
    Next, applies the target gain per source.
    Then, remixes the sources to create the enhanced signal.
    Finally, the enhanced signal is amplified for the listener.

    Args:
        config (dict): Dictionary of configuration options for enhancing music.

    """
    if config.separator.causality not in ["causal", "noncausal"]:
        raise ValueError(
            f"Causality must be causal or noncausal, {config.separator.causality} was"
            " provided."
        )

    # Set the output directory where processed signals will be saved
    enhanced_folder = Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    device, _ = get_device(config.separator.device)

    # Load listener audiograms and songs
    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    # Load gains
    with Path(config.path.gains_file).open("r", encoding="utf-8") as file:
        gains = json.load(file)

    # Load Scenes
    with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    # Load scene listeners
    with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    # load songs
    with Path(config.path.music_file).open("r", encoding="utf-8") as file:
        songs = json.load(file)

    # Load compressor params
    with Path(config.path.enhancer_params_file).open("r", encoding="utf-8") as file:
        enhancer_params = json.load(file)

    # Load separation model
    separation_models = load_separation_model(config.separator.causality, device)

    # create hearing aid
    enhancer = MultibandCompressor(
        crossover_frequencies=config.enhancer.crossover_frequencies,
        sample_rate=config.input_sample_rate,
    )

    # Select a batch to process
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )

    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    num_scenes = len(scene_listener_pairs)
    for idx, scene_listener_pair in enumerate(scene_listener_pairs, 1):
        scene_id, listener_id = scene_listener_pair

        scene = scenes[scene_id]
        song_name = scene["music"]

        logger.info(
            f"[{idx:03d}/{num_scenes:03d}] "
            f"Processing {scene_id}: song {song_name} for listener {listener_id}"
        )
        # Get the listener's audiogram
        listener = listener_dict[listener_id]

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

        # Read the mixture signal
        # Convert to 32-bit floating point and transpose
        # from [samples, channels] to [channels, samples]
        source_list = {
            f"source_{idx}": s["instrument"].split("_")[0]
            for idx, s in enumerate(songs[song_name].values(), 1)
            if "Mixture" not in s["instrument"]
        }

        mixture_signal, mix_sample_rate = read_flac_signal(
            filename=Path(config.path.music_dir) / songs[song_name]["mixture"]["track"]
        )
        assert mix_sample_rate == config.input_sample_rate

        start = songs[song_name]["mixture"]["start"]
        end = start + songs[song_name]["mixture"]["duration"]
        mixture_signal = mixture_signal[
            int(start * mix_sample_rate) : int(end * mix_sample_rate),
            :,
        ]
        stems: dict[str, ndarray] = decompose_signal(
            model=separation_models,
            signal=mixture_signal,
            signal_sample_rate=config.input_sample_rate,
            device=device,
            sources_list=source_list,
            listener=listener,
        )

        # Apply gains to sources
        gain_scene = check_repeated_source(gains[scene["gain"]], source_list)
        stems = apply_gains(stems, config.input_sample_rate, gain_scene)

        # Downmix to stereo
        enhanced_signal = remix_stems(stems)

        # adjust levels to get roughly -40 dB before compressor
        enhanced_signal = adjust_level(enhanced_signal, gains[scene["gain"]])

        # Apply compressor
        enhanced_signal = process_remix_for_listener(
            signal=enhanced_signal,
            enhancer=enhancer,
            enhancer_params=mbc_params_listener,
            listener=listener,
        )

        # Save the enhanced signal in the corresponding directory
        if 0 < int(scene_id[1:]) < 49999:
            out_dir = "train"
        elif 50000 < int(scene_id[1:]) < 59999:
            out_dir = "valid"
        else:
            out_dir = "test"

        filename = (
            Path(enhanced_folder) / out_dir / f"{scene_id}_{listener.id}_remix.flac"
        )

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
