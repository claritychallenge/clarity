"""Module to Generate the metadata for the scenes in the CAD-1 Task-2 challenge."""
# pylint: disable=import-error

import json
import logging
from typing import Any, Dict

import hydra
import numpy as np
from omegaconf import DictConfig

from clarity.utils.car_noise_simulator.carnoise_parameters_generator import (
    CarNoiseParametersGenerator,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set the seed for the random number generator."""
    np.random.seed(seed)


def get_random_dict_item(input_dict: dict) -> Any:
    """
    Selects a random item from a dictionary.

    Args:
        input_dict (dict): A dictionary where keys are bird IDs and values are bird data.

    Returns:
        A random item from the input dictionary.
    """
    random_key = np.random.choice(list(input_dict.keys()), size=1, replace=False)[0]
    return input_dict[random_key]


def get_random_car_params(min_speed: int = 50, max_speed: int = 120) -> Dict:
    """
    Returns a dictionary with the parameters for a car noise.
    These parameters are generated randomly, based on the car speed.
    The parameters are to be used by the CarNoiseSignalGeneration class

    Parameters:
        - min_speed (int): The minimum speed that can be returned (default 50).
        - max_speed (int): The maximum speed that can be returned (default 120).

    Returns:
        A dictionary containing the parameters needed by the CarNoiseSignalGeneration class.
    """
    speed = np.random.randint(min_speed, max_speed)
    car_params = CarNoiseParametersGenerator().gen_parameters(speed)
    return car_params


def read_json(path_file) -> Dict:
    """Function the read a json file and return the data as a dictionary
    or only the keys if ```return_keys``` is True.

    Args:
        path_file (str): Path to the json file.
    """

    with open(path_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def get_random_snr(min_snr, max_snr, round_to=4) -> float:
    """Function to get a random SNR value.

    Args:
        min_snr (float): The minimum SNR value.
        max_snr (float): The maximum SNR value.
        round_to (int): The number of decimals to round the SNR value to.

    Returns:
        float: A random SNR value.
    """

    return float(np.round(np.random.uniform(min_snr, max_snr, 1), round_to))


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    """Main function to generate the metadata for the scenes in the CAD-1 Task-2 challenge.

    This function relies on a random seed to generate the metadata.
    The seed is set to 2023 by default and it should always be present.
    This ensures all participant have access to the same initial scenes.
    However, participants can augment the number of scenes by adding more seeds
    in the config file. Note the first seed must always be 2023 and cannot be change.

    Args:
        cfg (DictConfig): The configuration object.
    """

    logger.info("Generating metadata")

    assert (
        cfg.seed[0] == 2023
    ), f"First training seed must be 2023, {cfg.valid_seed} was provided"

    assert (
        cfg.valid_seed == 2023
    ), f"Validation seed should be 2023, {cfg.valid_seed} was provided"

    # read metadata in json format
    train_songs = read_json(cfg.path.train_music_file)
    valid_songs = read_json(cfg.path.valid_music_file)
    train_listeners = read_json(cfg.path.listeners_train_file)
    valid_listeners = read_json(cfg.path.listeners_valid_file)
    brir = read_json(cfg.path.brir_file)

    # Start generating scenes for training
    all_scenes = {}
    logger.info("... training metadata")
    for seed in cfg.seed:
        logger.info(f"...... seed {seed}")
        set_seed(seed)
        train_songs_shuffled = np.random.permutation(list(train_songs.keys()))
        for song in train_songs_shuffled:
            listener = np.random.choice(list(train_listeners.keys()), 1, replace=False)[
                0
            ]
            all_scenes[f"T-{song}_{listener}_S{seed}"] = {
                "song": song,
                "song_path": train_songs[song]["path"],
                "listener": listener,
                "hr": get_random_dict_item(brir["training"]),
                "car_noise_parameters": get_random_car_params(
                    min_speed=50, max_speed=120
                ),
                "snr": get_random_snr(0, 15.0),
                "split": "train",
            }

    # Start generating scenes for validation
    logger.info("... validation metadata with seed 2023")

    seed = cfg.valid_seed
    set_seed(seed)
    valid_songs_shuffled = np.random.permutation(list(valid_songs.keys()))
    for song in valid_songs_shuffled:
        listener = np.random.choice(list(valid_listeners.keys()), 1, replace=False)[0]
        all_scenes[f"T-{song}_{listener}_S{seed}"] = {
            "song": song,
            "song_path": valid_songs[song]["path"],
            "listener": listener,
            "hr": get_random_dict_item(brir["development"]),
            "car_noise_parameters": get_random_car_params(min_speed=50, max_speed=120),
            "snr": get_random_snr(0, 15.0),
            "split": "valid",
        }

    logger.info(f"Saving scenes metadata in {cfg.path.out_scene_file}")
    with open(cfg.path.out_scene_file, "w", encoding="utf-8") as file:
        json.dump(all_scenes, file, indent=4)

    logger.info("Done")


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()
