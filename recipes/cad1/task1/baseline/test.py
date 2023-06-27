""" Run the baseline enhancement. """
from __future__ import annotations

# pylint: disable=import-error
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.io import wavfile
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from recipes.cad1.task1.baseline.enhance import (
    decompose_signal,
    get_device,
    process_stems_for_listener,
    remix_signal,
    save_flac_signal,
)
from recipes.cad1.task1.baseline.evaluate import make_song_listener_list

# pylint: disable=too-many-locals

logger = logging.getLogger(__name__)


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

    if config.separator.model not in ["demucs", "openunmix"]:
        raise ValueError(f"Separator model {config.separator.model} not supported.")

    enhanced_folder = Path("enhanced_signals") / "inferring"
    enhanced_folder.mkdir(parents=True, exist_ok=True)

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

    # Processing Validation Set
    # Load listener audiograms and songs
    with open(config.path.listeners_eval_file, encoding="utf-8") as file:
        listener_audiograms = json.load(file)

    with open(config.path.music_eval_file, encoding="utf-8") as file:
        song_data = json.load(file)
    songs_details = pd.DataFrame.from_dict(song_data)

    song_listener_pairs = make_song_listener_list(
        songs_details["Track Name"], listener_audiograms
    )
    # Select a batch to process
    song_listener_pairs = song_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    # Create hearing aid objects
    enhancer = NALR(**config.nalr)
    compressor = Compressor(**config.compressor)

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    prev_song_name = None
    num_song_list_pair = len(song_listener_pairs)
    for idx, song_listener in enumerate(song_listener_pairs, 1):
        song_name, listener_name = song_listener
        logger.info(
            f"[{idx:03d}/{num_song_list_pair:03d}] "
            f"Processing {song_name} for {listener_name}..."
        )
        # Get the listener's audiogram
        listener_info = listener_audiograms[listener_name]

        # Find the music split directory
        split_directory = (
            "test"
            if songs_details.loc[
                songs_details["Track Name"] == song_name, "Split"
            ].iloc[0]
            == "test"
            else "train"
        )

        critical_frequencies = np.array(listener_info["audiogram_cfs"])
        audiogram_left = np.array(listener_info["audiogram_levels_l"])
        audiogram_right = np.array(listener_info["audiogram_levels_r"])

        # Baseline Steps
        # 1. Decompose the mixture signal into vocal, drums, bass, and other stems
        #    We validate if 2 consecutive signals are the same to avoid
        #    decomposing the same song multiple times
        if prev_song_name != song_name:
            # Decompose song only once
            prev_song_name = song_name

            sample_rate, mixture_signal = wavfile.read(
                Path(config.path.music_dir)
                / split_directory
                / song_name
                / "mixture.wav"
            )
            mixture_signal = (mixture_signal / 32768.0).astype(np.float32).T
            assert sample_rate == config.sample_rate

            # Decompose mixture signal into stems
            stems = decompose_signal(
                separation_model,
                model_sample_rate,
                mixture_signal,
                sample_rate,
                device,
                sources_order,
                audiogram_left,
                audiogram_right,
                normalise,
            )

        # 2. Apply NAL-R prescription to each stem
        #     Baseline applies NALR prescription to each stem instead of using the
        #     listener's audiograms in the decomposition. This step can be skipped
        #     if the listener's audiograms are used in the decomposition
        processed_stems = process_stems_for_listener(
            stems,
            enhancer,
            compressor,
            audiogram_left,
            audiogram_right,
            critical_frequencies,
            config.apply_compressor,
        )

        # 3. Save processed stems
        for stem_str, stem_signal in processed_stems.items():
            filename = (
                enhanced_folder
                / f"{listener_name}"
                / f"{song_name}"
                / f"{listener_name}_{song_name}_{stem_str}.flac"
            )
            filename.parent.mkdir(parents=True, exist_ok=True)
            save_flac_signal(
                signal=stem_signal,
                filename=filename,
                signal_sample_rate=config.sample_rate,
                output_sample_rate=config.stem_sample_rate,
                do_scale_signal=True,
            )

        # 3. Remix Signal
        enhanced = remix_signal(processed_stems)

        # 5. Save enhanced (remixed) signal
        filename = (
            enhanced_folder
            / f"{listener_info['name']}"
            / f"{song_name}"
            / f"{listener_info['name']}_{song_name}_remix.flac"
        )
        save_flac_signal(
            signal=enhanced,
            filename=filename,
            signal_sample_rate=config.sample_rate,
            output_sample_rate=config.remix_sample_rate,
            do_clip_signal=True,
            do_soft_clip=config.soft_clip,
        )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
