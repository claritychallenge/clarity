"""Tests for the evaluation module"""
import numpy as np
import pytest
from omegaconf import DictConfig
from scipy.io import wavfile

from recipes.cad1.task1.baseline.evaluate import (
    ResultsFile,
    _evaluate_song_listener,
    compute_haaqi,
    make_song_listener_list,
    set_song_seed,
)


def test_ResultsFile(tmp_path):
    results_file = tmp_path / "results.csv"
    rf = ResultsFile(results_file.as_posix())
    rf.write_header()
    rf.add_result(
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
    with open(results_file, "r") as f:
        contents = f.read()
        assert (
            "My favorite song,My listener,0.9,0.8,0.8,0.9,0.9,0.8,0.8,0.95,0.95"
            in contents
        )


def test_compute_haaqi():
    np.random.seed(42)

    fs = 16000
    enh_signal = np.random.uniform(-1, 1, fs * 10)
    ref_signal = np.random.uniform(-1, 1, fs * 10)

    audiogram = np.array([10, 20, 30, 40, 50, 60])
    audiogram_frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])

    # Compute HAAQI score
    score = compute_haaqi(
        processed_signal=enh_signal,
        reference_signal=ref_signal,
        audiogram=audiogram,
        audiogram_frequencies=audiogram_frequencies,
        sample_rate=fs,
    )

    # Check that the score is a float between 0 and 1
    assert score == pytest.approx(0.117063418, rel=1e-7)


@pytest.mark.parametrize(
    "song,expected_result",
    (["my favorite song", 83], ["another song", 3]),
)
def test_set_song_seed(song, expected_result):
    # Set seed for the same song
    set_song_seed(song)
    assert np.random.randint(100) == expected_result


def test_make_song_listener_list():
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


def test_evaluate_song_listener(tmp_path):
    np.random.seed(2023)

    expected_results = {
        "left_drums": 0.107854176,
        "right_drums": 0.104024261,
        "left_bass": 0.111090873,
        "right_bass": 0.108046217,
        "left_other": 0.111885722,
        "right_other": 0.110098967,
        "left_vocals": 0.103490312,
        "right_vocals": 0.108655100,
    }

    # Generate reference and enhanced wav files
    enhanced_folder = tmp_path / "enhanced"
    reference_folder = tmp_path / "reference"
    enhanced_folder.mkdir()
    reference_folder.mkdir()

    left_right_instruments = list(expected_results.keys())
    instruments = ["drums", "bass", "other", "vocals"]

    # Define test inputs
    song = "punk_is_not_dead"
    listener = "my_music_listener"
    config = DictConfig(
        {
            "evaluate": {"set_random_seed": True},
            "path": {"music_dir": reference_folder.as_posix()},
            "nalr": {"fs": 44100},
        }
    )
    split_dir = "test"
    listener_audiograms = {
        "my_music_listener": {
            "audiogram_levels_l": [20, 30, 35, 45, 50, 60, 65, 60],
            "audiogram_levels_r": [20, 30, 35, 45, 50, 60, 65, 60],
            "audiogram_cfs": [250, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        }
    }

    # Create reference and enhanced wav samples
    for lr_instrument in left_right_instruments:
        # enhanced signals are mono
        enh_file = enhanced_folder / f"{listener}_{song}_{lr_instrument}.wav"
        temp_signal = np.random.uniform(-1, 1, 44100 * 5).astype(np.float32) * 32768
        wavfile.write(enh_file, 44100, temp_signal)

    for instrument in instruments:
        # reference signals are stereo
        ref_file = reference_folder / split_dir / song / f"{instrument}.wav"
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        temp_signal = (
            np.random.uniform(-1, 1, (44100 * 5, 2)).astype(np.float32) * 32768
        )
        wavfile.write(ref_file, 44100, temp_signal)

    # Call the function
    combined_score, per_instrument_score = _evaluate_song_listener(
        song,
        listener,
        config,
        split_dir,
        listener_audiograms[listener],
        enhanced_folder,
    )

    print(combined_score, per_instrument_score)
    # Check the outputs
    # Combined score
    assert isinstance(combined_score, float)
    assert combined_score == pytest.approx(0.1081432040, rel=1e-7)

    # Per instrument score
    assert isinstance(per_instrument_score, dict)
    for instrument in left_right_instruments:
        assert instrument in per_instrument_score.keys()
        assert isinstance(per_instrument_score[instrument], float)
        assert per_instrument_score[instrument] == pytest.approx(
            expected_results[instrument], rel=1e-7
        )
