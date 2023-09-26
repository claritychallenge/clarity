"""Tests for the evaluation module"""
from pathlib import Path

# pylint: disable=import-error
import numpy as np
import pytest
from omegaconf import DictConfig
from scipy.io import wavfile

from clarity.utils.audiogram import Audiogram, Listener
from recipes.cad1.task1.baseline.evaluate import (
    ResultsFile,
    _evaluate_song_listener,
    make_song_listener_list,
    set_song_seed,
)


def test_results_file(tmp_path):
    """Test the class ResultsFile"""
    results_file = tmp_path / "results.csv"
    result_file = ResultsFile(results_file.as_posix())
    result_file.write_header()
    result_file.add_result(
        listener_id="My listener",
        song="My favorite song",
        score=0.9,
        instruments_scores={
            "left_bass": 0.8,
            "right_bass": 0.8,
            "left_drums": 0.9,
            "right_drums": 0.9,
            "left_other": 0.8,
            "right_other": 0.8,
            "left_vocals": 0.95,
            "right_vocals": 0.95,
        },
    )
    with open(results_file, encoding="utf-8") as file:
        contents = file.read()
        assert (
            "My favorite song,My listener,0.9,0.8,0.8,0.9,0.9,0.8,0.8,0.95,0.95"
            in contents
        )


@pytest.mark.parametrize(
    "song,expected_result",
    [("my favorite song", 83), ("another song", 3)],
)
def test_set_song_seed(song, expected_result):
    """Thest the function set_song_seed using 2 different inputs"""
    # Set seed for the same song
    set_song_seed(song)
    assert np.random.randint(100) == expected_result


def test_make_song_listener_list():
    """Test the function generates the correct list of songs and listeners pairs"""
    songs = ["song1", "song2", "song3"]
    listeners = {"listener1": 1, "listener2": 2, "listener3": 3}
    expected_output = [
        ("song1", "listener1"),
        ("song1", "listener2"),
        ("song1", "listener3"),
        ("song2", "listener1"),
        ("song2", "listener2"),
        ("song2", "listener3"),
        ("song3", "listener1"),
        ("song3", "listener2"),
        ("song3", "listener3"),
    ]

    assert make_song_listener_list(songs, listeners) == expected_output

    # Test with small_test = True
    expected_output_small_test = [("song1", "listener1")]
    assert (
        make_song_listener_list(songs, listeners, small_test=True)
        == expected_output_small_test
    )


@pytest.mark.parametrize(
    "song,listener_id,config,split_dir,listener_audiograms,expected_results",
    [
        (
            "punk_is_not_dead",
            "my_music_listener",
            {
                "evaluate": {"set_random_seed": True},
                "path": {"music_dir": None},
                "sample_rate": 44100,
                "nalr": {"sample_rate": 44100},
            },
            "test",
            {
                "my_music_listener": {
                    "audiogram_levels_l": [20, 30, 35, 45, 50, 60, 65, 60],
                    "audiogram_levels_r": [20, 30, 35, 45, 50, 60, 65, 60],
                    "audiogram_cfs": [250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
                }
            },
            {
                "left_drums": 0.14229422779265366,
                "right_drums": 0.15044965630960655,
                "left_bass": 0.1333774836344767,
                "right_bass": 0.14541827476097585,
                "left_other": 0.16310480582621734,
                "right_other": 0.15427835764875864,
                "left_vocals": 0.12291980372806624,
                "right_vocals": 0.1368378217706031,
            },
        )
    ],
)
def test_evaluate_song_listener(
    song,
    listener_id,
    config,
    split_dir,
    listener_audiograms,
    expected_results,
    tmp_path,
):
    """Test the function _evaluate_song_listener returns correct results given input"""
    np.random.seed(2023)
    listener_data = listener_audiograms[listener_id]
    audiogram_left = Audiogram(
        listener_data["audiogram_levels_l"], listener_data["audiogram_cfs"]
    )
    audiogram_right = Audiogram(
        listener_data["audiogram_levels_r"], listener_data["audiogram_cfs"]
    )
    listener = Listener(audiogram_left, audiogram_right, id=listener_id)
    # Generate reference and enhanced wav files
    enhanced_folder = tmp_path / "enhanced"
    enhanced_folder.mkdir()

    config = DictConfig(config)
    config.path.music_dir = (tmp_path / "reference").as_posix()

    instruments = ["drums", "bass", "other", "vocals"]

    # Create reference and enhanced wav samples
    for lr_instrument in list(expected_results.keys()):
        # enhanced signals are mono
        enh_file = (
            enhanced_folder
            / f"{listener.id}"
            / f"{song}"
            / f"{listener.id}_{song}_{lr_instrument}.wav"
        )
        enh_file.parent.mkdir(exist_ok=True, parents=True)

        # Using very short 100 ms signals to speed up the test
        wavfile.write(
            enh_file,
            44100,
            np.random.uniform(-1, 1, 4410).astype(np.float32) * 32768,
        )

    for instrument in instruments:
        # reference signals are stereo
        ref_file = Path(config.path.music_dir) / split_dir / song / f"{instrument}.wav"
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(
            ref_file,
            44100,
            np.random.uniform(-1, 1, (4410, 2)).astype(np.float32) * 32768,
        )

    # Call the function
    combined_score, per_instrument_score = _evaluate_song_listener(
        song,
        listener,
        config,
        split_dir,
        enhanced_folder,
    )

    # Check the outputs
    # Combined score
    assert isinstance(combined_score, float)
    assert combined_score == pytest.approx(
        0.14358505393391977, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    # Per instrument score
    assert isinstance(per_instrument_score, dict)
    for instrument in expected_results:
        assert instrument in per_instrument_score
        assert isinstance(per_instrument_score[instrument], float)
        assert per_instrument_score[instrument] == pytest.approx(
            expected_results[instrument],
            rel=pytest.rel_tolerance,
            abs=pytest.abs_tolerance,
        )


@pytest.mark.skip(reason="Not implemented yet")
def test_run_calculate_aq():
    """test run_calculate_aq function"""
    # run_calculate_aq()
