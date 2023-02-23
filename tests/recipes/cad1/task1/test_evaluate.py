"""Tests for the evaluation module"""
import numpy as np
import pytest

from recipes.cad1.task1.baseline.evaluate import (
    ResultsFile,
    compute_haaqi,
    make_song_listener_list,
    set_song_seed,
)

fs = 16000
audiogram_freq = np.array([250, 500, 1000, 2000, 4000, 6000])
audiogram = np.array([10, 20, 30, 40, 50, 60])
audiogram_frequencies = np.array([125, 250, 500, 1000, 2000, 4000, 8000])


def test_ResultsFile():
    results_file = pytest.tmp_path / "results.csv"
    rf = ResultsFile(results_file)
    rf.write_header()
    rf.add_result(
        listener="My listener",
        song="My favorite song",
        score=0.9,
        instruments_scores={
            "l_bass": 0.8,
            "r_bass": 0.8,
            "l_drums": 0.9,
            "r_drums": 0.9,
            "l_other": 0.8,
            "r_other": 0.8,
            "l_vocals": 0.95,
            "r_vocals": 0.95,
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
        enh_signal=enh_signal,
        ref_signal=ref_signal,
        audiogram=audiogram,
        audiogram_frequencies=audiogram_frequencies,
        fs_signal=fs,
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
