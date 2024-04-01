"""Evaluate the enhanced signals using the HAAQI metric."""

# pylint: disable=too-many-locals
# pylint: disable=import-error
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.audiogram import Listener
from clarity.utils.flac_encoder import read_flac_signal
from clarity.utils.results_support import ResultsFile
from recipes.cad1.task2.baseline.audio_manager import AudioManager
from recipes.cad1.task2.baseline.baseline_utils import (
    load_hrtf,
    load_listeners_and_scenes,
    make_scene_listener_list,
    read_mp3,
)
from recipes.cad1.task2.baseline.car_scene_acoustics import CarSceneAcoustics

logger = logging.getLogger(__name__)


def set_scene_seed(scene: str):
    """Set a seed that is unique for the given song
    based on the last 8 characters of the 'md5'
    `.hexdigest` of the scene itself.
    """
    scene_encoded = hashlib.md5(scene.encode("utf-8")).hexdigest()
    scene_md5 = int(scene_encoded, 16) % (10**8)
    np.random.seed(scene_md5)


# pylint: disable=too-many-arguments
def evaluate_scene(
    ref_signal: np.ndarray,
    enh_signal: np.ndarray,
    sample_rate: int,
    scene_id: str,
    current_scene: dict,
    listener: Listener,
    car_scene_acoustic: CarSceneAcoustics,
    hrtf: dict,
    config: DictConfig,
) -> tuple[float, float]:
    """Evaluate a single scene and return HAAQI scores for left and right ears

    Args:
        ref_signal (np.ndarray): A numpy array of shape (2, n_samples)
            containing the reference signal.
        enh_signal (np.ndarray): A numpy array of shape (2, n_samples)
            containing the enhanced signal.
        sample_rate (int): The sampling frequency of the reference and enhanced signals.
        scene_id (str): A string identifier for the scene being evaluated.
        current_scene (dict): A dictionary containing information about the scene being
            evaluated, including the song ID, the listener ID, the car noise type, and
            the split.
        listener (Listener): the listener to use
        car_scene_acoustic (CarSceneAcoustics): An instance of the CarSceneAcoustics
            class, which is used to generate car noise and add binaural room impulse
            responses (BRIRs) to the enhanced signal.
        hrtf (dict): A dictionary containing the head-related transfer functions (HRTFs)
            for the listener being evaluated. This includes the left and right HRTFs for
            the car and the anechoic room.
        config (DictConfig): A dictionary-like object containing various configuration
            parameters for the evaluation. This includes the path to the enhanced signal
            folder,the path to the music directory, and a flag indicating whether to set
            a random seed.

    Returns:
        Tuple[float, float]: A tuple containing HAAQI scores for left and right ears.

    """
    audio_manager = AudioManager(
        output_audio_path=Path("evaluation_signals")
        / f"{listener.id}"
        / f"{current_scene['song']}",
        sample_rate=sample_rate,
        soft_clip=config.soft_clip,
    )

    if config.evaluate.set_random_seed:
        set_scene_seed(scene_id)

    # Applies the Car Acoustics to the enhanced signal, i.e., the speakers output

    processed_signal = car_scene_acoustic.apply_car_acoustics_to_signal(
        enh_signal,
        current_scene,
        listener,
        hrtf,
        audio_manager,
        config,
    )

    # Normalise reference signal level to ha_processed_signal level
    # ref_signal = ref_signal * scale_factor
    # Following Spotify standard, Max level is -11 LUFS to avoid clipping
    # https://artists.spotify.com/en/help/article/loudness-normalization
    ref_signal = car_scene_acoustic.add_hrtf_to_stereo_signal(
        ref_signal, hrtf["anechoic"], "Anechoic"
    )
    if config.evaluate.save_intermediate_wavs:
        audio_manager.add_audios_to_save("ref_signal_anechoic", ref_signal)

    ref_signal = car_scene_acoustic.equalise_level(
        signal=ref_signal, reference_signal=processed_signal, max_level=-14
    )
    audio_manager.add_audios_to_save("ref_signal_for_eval", ref_signal)

    audio_manager.save_audios()

    # Compute HAAQI scores
    aq_score_l = compute_haaqi(
        processed_signal[0, :],
        ref_signal[0, :],
        sample_rate,
        sample_rate,
        listener.audiogram_left,
    )
    aq_score_r = compute_haaqi(
        processed_signal[1, :],
        ref_signal[1, :],
        sample_rate,
        sample_rate,
        listener.audiogram_right,
    )
    return aq_score_l, aq_score_r


@hydra.main(config_path="", config_name="config")
def run_calculate_audio_quality(config: DictConfig) -> None:
    """Evaluate the enhanced signals using the HAAQI metric."""

    # Load scenes and listeners depending on config.evaluate.split
    scenes, listener_dict, scenes_listeners = load_listeners_and_scenes(config)
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )
    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    hrtfs = load_hrtf(config)

    enhanced_folder = Path("enhanced_signals")
    logger.info(f"Evaluating from {enhanced_folder} directory")

    scores_headers = [
        "scene",
        "song",
        "genre",
        "listener",
        "score",
        "haaqi_left",
        "haaqi_right",
    ]

    results_file_name = "scores.csv"
    if config.evaluate.batch_size > 1:
        results_file_name = (
            f"scores_{config.evaluate.batch + 1}-{config.evaluate.batch_size}.csv"
        )

    results_file = ResultsFile(
        file_name=results_file_name,
        header_columns=scores_headers,
    )

    # Initialize acoustic scene model
    car_scene_acoustic = CarSceneAcoustics(
        track_duration=30,
        sample_rate=config.sample_rate,
        hrtf_dir=config.path.hrtf_dir,
        config_nalr=config.nalr,
        config_compressor=config.compressor,
        extend_noise=0.2,
    )

    # Iterate over scenes
    for scene_id, listener_id in tqdm(scene_listener_pairs):
        current_scene = scenes[scene_id]

        # Retrieve audiograms
        listener = listener_dict[listener_id]

        # Retrieve HRTF according to the listener's head orientation
        hrtf_scene = hrtfs[str(current_scene["hr"])]

        # Load reference signal
        reference_song_path = (
            Path(config.path.music_dir) / f"{current_scene['song_path']}"
        )
        # Read MP3 reference signal using librosa
        reference_signal, _ = read_mp3(
            reference_song_path.as_posix(), sample_rate=config.sample_rate
        )

        # Load enhanced signal
        enhanced_folder = Path("enhanced_signals") / config.evaluate.split
        # Read WAV enhanced signal using scipy.io.wavfile
        enhanced_signal, enhanced_sample_rate = read_flac_signal(
            enhanced_folder
            / f"{listener.id}"
            / f"{scene_id}_{listener.id}_{current_scene['song']}.flac"
        )

        assert enhanced_sample_rate == config.enhanced_sample_rate

        # Evaluate scene
        aq_score_l, aq_score_r = evaluate_scene(
            reference_signal,
            enhanced_signal.T,
            config.sample_rate,
            scene_id,
            current_scene,
            listener,
            car_scene_acoustic,
            hrtf_scene,
            config,
        )

        # Compute combined score and save
        score = np.mean([aq_score_r, aq_score_l])
        results_file.add_result(
            {
                "scene": scene_id,
                "song": current_scene["song"],
                "genre": current_scene["song_path"].split("/")[-2],
                "listener": listener.id,
                "score": float(score),
                "haaqi_left": aq_score_l,
                "haaqi_right": aq_score_r,
            }
        )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run_calculate_audio_quality()
