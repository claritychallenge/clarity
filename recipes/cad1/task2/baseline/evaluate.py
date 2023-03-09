"""Evaluate the enhanced signals using the HAAQI metric."""
# pylint: disable=too-many-locals
# pylint: disable=import-error

import csv
import hashlib
import logging
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from clarity.evaluator.haaqi import compute_haaqi
from recipes.cad1.task2.baseline.audio_manager import AudioManager
from recipes.cad1.task2.baseline.baseline_utils import (
    load_listeners_and_scenes,
    read_mp3,
)
from recipes.cad1.task2.baseline.car_scene_acoustics import CarSceneAcoustics

logger = logging.getLogger(__name__)


class ResultsFile:
    """A utility class for writing results to a CSV file.

    Attributes:
        file_name (str): The name of the file to write results to.
    """

    def __init__(self, file_name):
        """Initialize the ResultsFile instance.

        Args:
            file_name (str): The name of the file to write results to.
        """
        self.file_name = file_name

    def write_header(self):
        """Write the header row to the CSV file."""
        with open(self.file_name, "w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [
                    "song",
                    "listener",
                    "score",
                    "haaqi_left",
                    "haaqi_right",
                ]
            )

    def add_result(
        self,
        scene: str,
        listener: str,
        score: float,
        haaqi_left: float,
        haaqi_right: float,
    ):
        """Add a result to the CSV file.

        Args:
            scene (str): The name of the scene that the result is for.
            listener (str): The name of the listener who submitted the result.
            score (float): The combined score for the result.
            haaqi_left (float): The HAAQI score for the left channel.
            haaqi_right (float): The HAAQI score for the right channel.
        """

        logger.info(f"The combined score is {score}")

        with open(self.file_name, "a", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [
                    scene,
                    listener,
                    str(score),
                    str(haaqi_left),
                    str(haaqi_right),
                ]
            )


def set_scene_seed(scene: str):
    """Set a seed that is unique for the given song"""
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
    listener_audiogram: dict,
    car_scene_acoustic: CarSceneAcoustics,
    config: DictConfig,
) -> Tuple[float, float]:
    """Evaluate a single scene and return HAAQI scores for left and right ears

    Args:
        ref_signal (np.ndarray): A numpy array of shape (2, n_samples)
            containing the reference signal.
        enh_signal (np.ndarray): A numpy array of shape (2, n_samples)
            containing the enhanced signal.
        sample_rate (int): The sampling frequency of the reference and enhanced signals.
        scene_id (str): A string identifier for the scene being evaluated.
        current_scene (dict): A dictionary containing information about the scene being evaluated,
            including the song ID, the listener ID, the car noise type, and the split.
        listener_audiogram (dict): A dictionary containing the listener's audiogram data,
            including the center frequencies (cfs) and the hearing levels for both ears
            (audiogram_levels_l and audiogram_levels_r).
        car_scene_acoustic (CarSceneAcoustics): An instance of the CarSceneAcoustics class,
            which is used to generate car noise and add binaural room impulse responses (BRIRs)
            to the enhanced signal.
        config (DictConfig): A dictionary-like object containing various configuration
            parameters for the evaluation. This includes the path to the enhanced signal folder,
            the path to the music directory, and a flag indicating whether to set a random seed.

    Returns:
        Tuple[float, float]: A tuple containing the HAAQI scores for the left and right ears.

    """
    audio_manager = AudioManager(
        output_audio_path=(Path("evaluation_signals") / scene_id).as_posix(),
        sample_rate=sample_rate,
        soft_clip=config.evaluate.soft_clip,
    )

    if config.evaluate.set_random_seed:
        set_scene_seed(scene_id)

    center_frequencies = np.array(listener_audiogram["audiogram_cfs"])
    audiogram_left = np.array(listener_audiogram["audiogram_levels_l"])
    audiogram_right = np.array(listener_audiogram["audiogram_levels_r"])

    # 1. Generates car noise and adds anechoic HRTFs to the car noise
    # car_noise_anechoic = car_noise + anechoic HRTF

    car_noise = car_scene_acoustic.get_car_noise(current_scene["car_noise_parameters"])
    car_noise_anechoic = car_scene_acoustic.add_anechoic_hrtf(car_noise)

    if config.evaluate.save_intermediate_wavs:
        audio_manager.add_audios_to_save("car_noise_anechoic", car_noise_anechoic)

    # 2. Add HRTFs to enhanced signal
    # processed_signal = enh_signal + car HRTF
    processed_signal = car_scene_acoustic.add_car_hrtf(enh_signal, current_scene["hr"])

    if config.evaluate.save_intermediate_wavs:
        audio_manager.add_audios_to_save("enh_signal", enh_signal)
        audio_manager.add_audios_to_save("enh_signal_hrtf", processed_signal)

    # 3. Scale noise to target SNR
    # car_noise_anechoic = car_noise_anechoic * scale_factor

    car_noise_anechoic = car_scene_acoustic.scale_signal_to_snr(
        signal=car_noise_anechoic,
        reference_signal=processed_signal,
        snr=float(current_scene["snr"]),
    )
    if config.evaluate.save_intermediate_wavs:
        audio_manager.add_audios_to_save(
            "car_noise_anechoic_scaled", car_noise_anechoic
        )

    # 4. Add the scaled anechoic car noise to the enhanced signal
    # processed_signal = (enh_signal * car HRTF) + (car_noise * Anechoic HRTF) * scale_factor
    processed_signal = (
        processed_signal + car_noise_anechoic[:, : processed_signal.shape[1]]
    )
    if config.evaluate.save_intermediate_wavs:
        audio_manager.add_audios_to_save(
            "enh_signal_hrtf_plus_car_noise_anechoic", processed_signal
        )

    # 5. Apply Hearing Aid to Left and Right channels and join them
    processed_signal_left = car_scene_acoustic.apply_hearing_aid(
        processed_signal[0, :], audiogram_left, center_frequencies
    )

    processed_signal_right = car_scene_acoustic.apply_hearing_aid(
        processed_signal[1, :], audiogram_right, center_frequencies
    )

    processed_signal = np.stack([processed_signal_left, processed_signal_right], axis=0)
    if config.evaluate.save_intermediate_wavs:
        audio_manager.add_audios_to_save(
            "ha_processed_signal_left", processed_signal_left
        )
        audio_manager.add_audios_to_save(
            "ha_processed_signal_right", processed_signal_right
        )

    # 6. Clip enhanced signal to [-1, 1]
    # processed_signal = np.clip(processed_signal, -1.0, 1.0)
    n_clipped, processed_signal = audio_manager.clip_audio(-1.0, 1.0, processed_signal)

    if n_clipped > 0:
        logger.warning(f"Evaluation Signal: {n_clipped} samples clipped")

    audio_manager.add_audios_to_save("ha_processed_signal", processed_signal)

    # 7. Normalise reference signal level to ha_processed_signal level
    # ref_signal = ref_signal * scale_factor

    ref_signal = car_scene_acoustic.scale_signal_to_snr(
        signal=ref_signal, reference_signal=processed_signal, snr=None
    )

    audio_manager.add_audios_to_save("ref_signal_normalised", ref_signal)

    audio_manager.save_audios()

    # 8. Compute HAAQI scores
    aq_score_l = compute_haaqi(
        processed_signal[0, :],
        ref_signal[0, :],
        audiogram_left,
        center_frequencies,
        sample_rate,
    )
    aq_score_r = compute_haaqi(
        processed_signal[1, :],
        ref_signal[1, :],
        audiogram_right,
        center_frequencies,
        sample_rate,
    )
    return aq_score_l, aq_score_r


@hydra.main(config_path="", config_name="config")
def run_calculate_aq(config: DictConfig) -> None:
    """Evaluate the enhanced signals using the HAAQI metric."""

    # Load scenes and listeners depending on config.evaluate.split
    scenes, listener_audiograms = load_listeners_and_scenes(config)

    enhanced_folder = Path("enhanced_signals")
    logger.info(f"Evaluating from {enhanced_folder} directory")

    results_file = ResultsFile("scores.csv")
    results_file.write_header()

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
    for scene_id, current_scene in tqdm(scenes.items()):
        # Retrieve audiograms
        listener = current_scene["listener"]
        listener_audiogram = listener_audiograms[listener]

        # Load reference signal
        reference_song_path = (
            Path(config.path.music_dir)
            / f"{current_scene['split']}"
            / f"{current_scene['song']:06d}.mp3"
        )
        # Read MP3 reference signal using librosa
        reference_signal, _ = read_mp3(
            reference_song_path.as_posix(), sample_rate=config.sample_rate
        )

        if reference_signal.ndim == 1:
            # If mono, duplicate to stereo
            reference_signal = np.stack([reference_signal, reference_signal], axis=0)

        # Load enhanced signal
        enhanced_folder = Path("enhanced_signals") / config.evaluate.split
        enhanced_song_id = f"{current_scene['listener']}_{current_scene['song']}"
        enhanced_song_path = enhanced_folder / f"{enhanced_song_id}.wav"

        # Read WAV enhanced signal using scipy.io.wavfile
        enhanced_sample_rate, enhanced_signal = wavfile.read(enhanced_song_path)
        enhanced_signal = enhanced_signal / 32768.0
        assert enhanced_sample_rate == config.sample_rate

        # Evaluate scene
        aq_score_l, aq_score_r = evaluate_scene(
            reference_signal,
            enhanced_signal.T,
            config.sample_rate,
            scene_id,
            current_scene,
            listener_audiogram,
            car_scene_acoustic,
            config,
        )

        # Compute combined score and save
        score = np.mean([aq_score_r, aq_score_l])
        results_file.add_result(
            scene_id,
            listener,
            score=float(score),
            haaqi_left=aq_score_l,
            haaqi_right=aq_score_r,
        )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run_calculate_aq()
