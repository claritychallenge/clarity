"""Tests for the enhance module"""
# pylint:: disable=import-error
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
from scipy.io import wavfile

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram
from recipes.cad_icassp_2024.baseline.evaluate import (
    apply_gains,
    apply_ha,
    level_normalisation,
    load_reference_stems,
    make_scene_listener_list,
    remix_stems,
    set_song_seed,
)


@pytest.mark.parametrize(
    "apply_compressor,expected_result",
    [(True, 5130.70234905519), (False, 11039.797138690315)],
)
def test_apply_ha(apply_compressor, expected_result):
    """Test apply_ha"""
    np.random.seed(2024)

    sample_rate = 44100
    duration = 0.5

    signal = np.random.rand(int(sample_rate * duration))

    audiogram = Audiogram(
        levels=np.ones(9),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000]),
    )
    enhancer = NALR(nfir=220, sample_rate=16000)
    compressor = Compressor(
        threshold=0.35, attenuation=0.1, attack=50, release=1000, rms_buffer_size=0.064
    )

    output = apply_ha(enhancer, compressor, signal, audiogram, apply_compressor)

    assert np.sum(output) == pytest.approx(expected_result)


@pytest.mark.parametrize(
    "gains,expected_result",
    [
        ({"vocals": 0, "bass": 0, "drums": 0, "other": 0}, 88207.89862265925),
        ({"vocals": 10, "bass": 0, "drums": 0, "other": 0}, 135801.06478181464),
        ({"vocals": 0, "bass": 10, "drums": 0, "other": 0}, 135716.2197962051),
        ({"vocals": 0, "bass": 0, "drums": 10, "other": 0}, 136040.5632722622),
        ({"vocals": 0, "bass": 0, "drums": 0, "other": 10}, 136003.7152825283),
        ({"vocals": 6, "bass": 0, "drums": 0, "other": 0}, 110114.28261802963),
        ({"vocals": 3, "bass": 0, "drums": 0, "other": 0}, 97288.123734263),
        ({"vocals": -3, "bass": 0, "drums": 0, "other": 0}, 81779.59153364039),
        ({"vocals": -6, "bass": 0, "drums": 0, "other": 0}, 77228.6986292428),
        ({"vocals": -10, "bass": 0, "drums": 0, "other": 0}, 73157.61801048138),
    ],
)
def test_apply_gains(tmp_path, gains, expected_result):
    """Test apply_gains
    It test the 4 different gains for the 4 different stems
    """
    np.random.seed(2024)

    sample_rate = 44100
    duration = 0.5

    for stem in ["vocals", "drums", "bass", "other", "mixture"]:
        filename = Path(tmp_path) / f"{stem}.wav"
        signal = np.random.uniform(size=(int(sample_rate * duration), 2))
        wavfile.write(filename, sample_rate, signal)

    stems, _ = load_reference_stems(tmp_path)
    output = apply_gains(stems, sample_rate, gains)

    result = 0
    for _, stem_signal in output.items():
        result += np.sum(stem_signal)

    assert result == pytest.approx(expected_result)


def test_level_normalisation():
    """Test level_normalisation"""
    np.random.seed(2024)

    sample_rate = 44100
    duration = 0.5

    reference_signal = np.random.rand(int(sample_rate * duration))
    signal = np.random.rand(int(sample_rate * duration)) * 2

    meter = pyln.Meter(int(sample_rate))

    normed_signal = level_normalisation(signal, reference_signal, sample_rate)

    assert np.sum(normed_signal) == pytest.approx(10899.6694813693)

    normed_signal_lufs = meter.integrated_loudness(normed_signal)
    reference_signal_lufs = meter.integrated_loudness(reference_signal)
    assert normed_signal_lufs == pytest.approx(reference_signal_lufs)


def test_remix_stems(tmp_path):
    """Test remix_stems"""
    np.random.seed(2024)

    sample_rate = 44100
    duration = 0.5

    for stem in ["vocals", "drums", "bass", "other", "mixture"]:
        filename = Path(tmp_path) / f"{stem}.wav"
        signal = np.random.rand(int(sample_rate * duration))
        wavfile.write(filename, sample_rate, signal)

    stems, mixture = load_reference_stems(tmp_path)
    remix = remix_stems(stems, mixture, sample_rate)
    assert np.sum(remix) == pytest.approx(21950.335598725396)


def test_make_scene_listener_list():
    """Test make_scene_listener_list"""
    song_list = {
        "my favorite song": ["list1", "list2"],
        "another song": ["list3", "list4"],
    }
    expected_output = [
        ("my favorite song", "list1"),
        ("my favorite song", "list2"),
        ("another song", "list3"),
        ("another song", "list4"),
    ]
    assert make_scene_listener_list(song_list) == pytest.approx(expected_output)


@pytest.mark.parametrize(
    "song,expected_result",
    [("my favorite song", 83), ("another song", 3)],
)
def test_set_song_seed(song, expected_result):
    """Thest the function set_song_seed using 2 different inputs"""
    # Set seed for the same song
    set_song_seed(song)
    assert np.random.randint(100) == pytest.approx(expected_result)


def test_load_reference_stems(tmp_path):
    """Test load_reference_stems"""
    np.random.seed(2024)

    sample_rate = 44100
    duration = 0.5

    for stem in ["vocals", "drums", "bass", "other", "mixture"]:
        filename = Path(tmp_path) / f"{stem}.wav"
        signal = np.random.rand(int(sample_rate * duration))
        wavfile.write(filename, sample_rate, signal)

    stems, mixture = load_reference_stems(tmp_path)
    assert len(stems) == 4
    assert np.sum(stems["vocals"]) == pytest.approx(11040.05074128)
    assert np.sum(stems["drums"]) == pytest.approx(10970.61284548)
    assert np.sum(stems["bass"]) == pytest.approx(11062.77053343)
    assert np.sum(stems["other"]) == pytest.approx(11058.65518849)
    assert np.sum(mixture) == pytest.approx(11046.02394025)
