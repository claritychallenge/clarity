"""Test the generate scenes script for the ICASSP 2024 CAD recipe."""
# pylint:: disable=import-error
import json

from omegaconf import DictConfig

from recipes.cad_icassp_2024.generate_dataset.generate_train_scenes import (
    choose_samples,
    generate_scene_listener,
    generate_scenes,
)


def test_choose_samples():
    # Test with a source list containing integers
    source = [1, 2, 3, 4, 5]
    number = 3
    samples = choose_samples(source, number)

    # Check if the number of samples returned is equal to the specified number
    assert len(samples) == number

    # Check if all samples are in the source list
    assert all(sample in source for sample in samples)

    # Test with an empty source list
    source = []
    number = 3
    samples = choose_samples(source, number)

    # Check if an empty list is returned when the source list is empty
    assert len(samples) == 0

    # Test with a source list containing a single element
    source = [42]
    number = 3
    samples = choose_samples(source, number)

    # Check if the returned list contains the single element
    assert len(samples) == 1
    assert samples[0] == 42


def test_generate_scenes(tmp_path):
    # Define a mock configuration dictionary
    tmp = tmp_path.as_posix()
    cfg = DictConfig(
        {
            "path": {
                "gains_file": f"{tmp}/fake_gains_file.json",
                "head_loudspeaker_positions_file": f"{tmp}/fake_positions_file.json",
                "metadata_dir": tmp,
            },
            "scene": {
                "number_scenes_per_song": 2,  # Adjust as needed
            },
        }
    )

    # Create fake gains and head loudspeaker positions files
    gains_data = {"gain1": 1, "gain2": 2, "gain3": 3}
    positions_data = {"position1": [1, 2, 3], "position2": [4, 5, 6]}

    with open(tmp_path / cfg.path.gains_file, "w", encoding="utf-8") as gains_file:
        json.dump(gains_data, gains_file)

    with open(
        tmp_path / cfg.path.head_loudspeaker_positions_file, "w", encoding="utf-8"
    ) as positions_file:
        json.dump(positions_data, positions_file)

    # Create a fake music metadata file
    music_data = [{"Track Name": "track1"}, {"Track Name": "track2"}]

    music_path = tmp_path / "musdb18.train.json"
    music_path.parent.mkdir(exist_ok=True, parents=True)
    with open(music_path, "w", encoding="utf-8") as music_file:
        json.dump(music_data, music_file)

    # Run the function
    generate_scenes(cfg)

    # Verify the results
    scenes_file = tmp_path / "scenes.train.json"
    assert scenes_file.exists()

    with open(scenes_file, encoding="utf-8") as scenes_file:
        scenes_data = json.load(scenes_file)

    # Check if the number of scenes is correct
    assert len(scenes_data) == 4

    # Check if the scenes are correctly generated
    assert scenes_data["scene_10001"]["music"] == "track1"
    assert scenes_data["scene_10001"]["gain"] == "gain2"
    assert scenes_data["scene_10001"]["head_loudspeaker_positions"] == "position2"
    assert scenes_data["scene_10002"]["music"] == "track1"
    assert scenes_data["scene_10002"]["gain"] == "gain1"
    assert scenes_data["scene_10002"]["head_loudspeaker_positions"] == "position1"
    assert scenes_data["scene_10003"]["music"] == "track2"
    assert scenes_data["scene_10003"]["gain"] == "gain3"
    assert scenes_data["scene_10003"]["head_loudspeaker_positions"] == "position2"
    assert scenes_data["scene_10004"]["music"] == "track2"
    assert scenes_data["scene_10004"]["gain"] == "gain1"
    assert scenes_data["scene_10004"]["head_loudspeaker_positions"] == "position1"


def test_generate_scene_listener(tmp_path):
    # Define a mock configuration dictionary
    cfg = DictConfig(
        {
            "path": {
                "metadata_dir": tmp_path.as_posix(),
            },
            "scene_listener": {
                "number_listeners_per_scene": 2,  # Adjust as needed
            },
        }
    )

    # Create a fake scenes metadata file
    scenes_data = {
        "scene1": {},
        "scene2": {},
        "scene3": {},
    }
    scenes_file = tmp_path / "scenes.train.json"
    with open(scenes_file, "w", encoding="utf-8") as f:
        json.dump(scenes_data, f)

    # Create a fake listeners metadata file
    listeners_data = {
        "listener1": {},
        "listener2": {},
        "listener3": {},
    }
    listeners_file = tmp_path / "listeners.train.json"
    with open(listeners_file, "w", encoding="utf-8") as f:
        json.dump(listeners_data, f)

    # Run the function
    generate_scene_listener(cfg)

    # Verify the results
    scene_listeners_file = tmp_path / "scene_listeners.train.json"
    assert scene_listeners_file.exists()

    with open(scene_listeners_file, encoding="utf-8") as f:
        scene_listeners_data = json.load(f)

    # Check if the number of scene-listeners is correct
    assert len(scene_listeners_data) == 3
    assert len(scene_listeners_data["scene1"]) == 2
    assert len(scene_listeners_data["scene2"]) == 2
    assert len(scene_listeners_data["scene3"]) == 2
