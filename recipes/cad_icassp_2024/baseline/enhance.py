""" Run the dummy enhancement. """
from __future__ import annotations

import json
import logging
from pathlib import Path

# pylint: disable=import-error
import hydra
import numpy as np
import torch
from evaluate import apply_gains, make_scene_listener_list, remix_stems
from numpy import ndarray
from omegaconf import DictConfig
from source_separation_utils import get_device, separate_sources
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram, Listener
from clarity.utils.file_io import read_signal, write_signal
from clarity.utils.signal_processing import (
    denormalize_signals,
    normalize_signal,
    resample,
)

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument
def decompose_signal(
    model: torch.nn.Module,
    model_sample_rate: int,
    signal: ndarray,
    signal_sample_rate: int,
    device: torch.device,
    sources_list: list[str],
    listener: Listener,
    normalise: bool = True,
) -> dict[str, ndarray]:
    """
    Decompose signal into 8 stems.

    The listener is ignored by the baseline system as it
     is not performing personalised decomposition.
    Instead, it performs a standard music decomposition using a pre-trained
     model trained on the MUSDB18 dataset.

    Args:
        model (torch.nn.Module): Torch model.
        model_sample_rate (int): Sample rate of the model.
        signal (ndarray): Signal to be decomposed.
        signal_sample_rate (int): Sample frequency.
        device (torch.device): Torch device to use for processing.
        sources_list (list): List of strings used to index dictionary.
        listener (Listener).
        normalise (bool): Whether to normalise the signal.

     Returns:
         Dictionary: Indexed by sources with the associated model as values.
    """
    if signal.shape[0] > signal.shape[1]:
        signal = signal.T

    if signal_sample_rate != model_sample_rate:
        signal = resample(signal, signal_sample_rate, model_sample_rate)

    if normalise:
        signal, ref = normalize_signal(signal)

    sources = separate_sources(
        model,
        torch.from_numpy(signal.astype(np.float32)),
        model_sample_rate,
        device=device,
    )

    # only one element in the batch
    sources = sources[0]
    if normalise:
        sources = denormalize_signals(sources, ref)

    sources = np.transpose(sources, (0, 2, 1))
    return dict(zip(sources_list, sources))


def apply_baseline_ha(
    enhancer: NALR,
    compressor: Compressor | None,
    signal: ndarray,
    audiogram: Audiogram,
    apply_compressor: bool = False,
) -> np.ndarray:
    """
    Apply NAL-R prescription hearing aid to a signal.

    Args:
        enhancer (NALR): A NALR object that enhances the signal.
        compressor (Compressor | None): A Compressor object that compresses the signal.
        signal (ndarray): An ndarray representing the audio signal.
        audiogram (Audiogram): An Audiogram object representing the listener's
            audiogram.
        apply_compressor (bool): Whether to apply the compressor.

    Returns:
        An ndarray representing the processed signal.
    """
    nalr_fir, _ = enhancer.build(audiogram)
    proc_signal = enhancer.apply(nalr_fir, signal)
    if apply_compressor:
        if compressor is None:
            raise ValueError("Compressor must be provided to apply compressor.")

        proc_signal, _, _ = compressor.process(proc_signal)
    return proc_signal


def process_remix_for_listener(
    signal: ndarray,
    enhancer: NALR,
    compressor: Compressor,
    listener: Listener,
    apply_compressor: bool = False,
) -> ndarray:
    """Process the stems from sources.

    Args:
        stems (dict) : Dictionary of stems
        sample_rate (float) : Sample rate of the signal
        enhancer (NALR) : NAL-R prescription hearing aid
        compressor (Compressor) : Compressor
        listener: Listener object
        apply_compressor (bool) : Whether to apply the compressor
    Returns:
        ndarray: Processed signal.
    """
    left_output = apply_baseline_ha(
        enhancer, compressor, signal[:, 0], listener.audiogram_left, apply_compressor
    )
    right_output = apply_baseline_ha(
        enhancer, compressor, signal[:, 1], listener.audiogram_right, apply_compressor
    )

    return np.stack([left_output, right_output], axis=1)


@hydra.main(config_path="", config_name="config")
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The system decomposes the music into vocal, drums, bass, and other stems.
    Then, the NAL-R prescription procedure is applied to each stem.
    Args:
        config (dict): Dictionary of configuration options for enhancing music.

    Returns 8 stems for each song:
        - left channel vocal, drums, bass, and other stems
        - right channel vocal, drums, bass, and other stems
    """

    # Set the output directory where processed signals will be saved
    enhanced_folder = Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    # Loading pretrained source separation model
    if config.separator.model == "demucs":
        separation_model = HDEMUCS_HIGH_MUSDB.get_model()
        model_sample_rate = HDEMUCS_HIGH_MUSDB.sample_rate
        sources_order = separation_model.sources
        normalise = True
    elif config.separator.model == "openunmix":
        separation_model = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq", niter=0)
        model_sample_rate = separation_model.sample_rate
        sources_order = ["vocals", "drums", "bass", "other"]
        normalise = False
    else:
        raise ValueError(f"Separator model {config.separator.model} not supported.")

    device, _ = get_device(config.separator.device)
    separation_model.to(device)

    # Load listener audiograms and songs
    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    with Path(config.path.gains_file).open("r", encoding="utf-8") as file:
        gains = json.load(file)

    with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    with Path(config.path.music_file).open("r", encoding="utf-8") as file:
        songs = json.load(file)

    enhancer = NALR(**config.nalr)
    compressor = Compressor(**config.compressor)

    # Select a batch to process
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )
    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    previous_song = ""
    num_scenes = len(scene_listener_pairs)
    for idx, scene_listener_pair in enumerate(scene_listener_pairs):
        scene_id, listener_id = scene_listener_pair

        scene = scenes[scene_id]
        song_name = f"{scene['music']}-{scene['head_position']}"

        logger.info(
            f"[{idx:03d}/{num_scenes:03d}] "
            f"Processing {song_name} for listener {listener_id}"
        )
        # Get the listener's audiogram
        listener = listener_dict[listener_id]

        # Read the mixture signal
        # Convert to 32-bit floating point and transpose
        # from [samples, channels] to [channels, samples]
        if song_name != previous_song:
            mixture_signal = read_signal(
                filename=Path(config.path.music_dir)
                / songs[song_name]["Path"]
                / "mixture.wav",
                sample_rate=config.sample_rate,
                allow_resample=True,
            )

            stems: dict[str, ndarray] = decompose_signal(
                model=separation_model,
                model_sample_rate=model_sample_rate,
                signal=mixture_signal,
                signal_sample_rate=config.sample_rate,
                device=device,
                sources_list=sources_order,
                listener=listener,
                normalise=normalise,
            )

            stems = apply_gains(stems, config.sample_rate, gains[scene["gain"]])
            enhanced_signal = remix_stems(stems, mixture_signal, model_sample_rate)

        enhanced_signal = process_remix_for_listener(
            signal=enhanced_signal,
            enhancer=enhancer,
            compressor=compressor,
            listener=listener,
            apply_compressor=config.apply_compressor,
        )

        filename = Path(
            enhanced_folder
            / f"{listener.id}"
            / f"{song_name}"
            / f"{scene_id}_{listener.id}_remix.wav"
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        write_signal(
            filename=filename,
            signal=enhanced_signal,
            sample_rate=config.sample_rate,
            floating_point=False,
            strict=False,
        )

    logger.info("Done!")


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
