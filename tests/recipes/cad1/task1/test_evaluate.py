"""Tests for the evaluation module"""
# pylint: disable=import-error

from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig
from scipy.io import wavfile

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
        listener="My listener",
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
    with open(results_file, "r", encoding="utf-8") as file:
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
    "song,listener,config,split_dir,listener_audiograms,expected_results",
    [
        (
            "punk_is_not_dead",
            "my_music_listener",
            {
                "evaluate": {"set_random_seed": True},
                "path": {"music_dir": None},
                "nalr": {"fs": 44100},
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
                "left_drums": 0.107854176,
                "right_drums": 0.104024261,
                "left_bass": 0.111090873,
                "right_bass": 0.108046217,
                "left_other": 0.111885722,
                "right_other": 0.110098967,
                "left_vocals": 0.103490312,
                "right_vocals": 0.108655100,
            },
        )
    ],
)
def test_evaluate_song_listener(
    song, listener, config, split_dir, listener_audiograms, expected_results, tmp_path
):
    """Test the function _evaluate_song_listener returns the correct results given the input"""
    np.random.seed(2023)

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
            / f"{listener}"
            / f"{song}"
            / f"{listener}_{song}_{lr_instrument}.wav"
        )
        enh_file.parent.mkdir(exist_ok=True, parents=True)

        wavfile.write(
            enh_file,
            44100,
            np.random.uniform(-1, 1, 44100 * 5).astype(np.float32) * 32768,
        )

    for instrument in instruments:
        # reference signals are stereo
        ref_file = Path(config.path.music_dir) / split_dir / song / f"{instrument}.wav"
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(
            ref_file,
            44100,
            np.random.uniform(-1, 1, (44100 * 5, 2)).astype(np.float32) * 32768,
        )

    # Call the function
    combined_score, per_instrument_score = _evaluate_song_listener(
        song,
        listener,
        config,
        split_dir,
        listener_audiograms[listener],
        enhanced_folder,
    )

    # Check the outputs
    # Combined score
    assert isinstance(combined_score, float)
    assert combined_score == pytest.approx(0.1081432040, rel=1e-7)

    # Per instrument score
    assert isinstance(per_instrument_score, dict)
    for instrument in list(expected_results.keys()):
        assert instrument in per_instrument_score.keys()
        assert isinstance(per_instrument_score[instrument], float)
        assert per_instrument_score[instrument] == pytest.approx(
            expected_results[instrument], rel=1e-7
        )
