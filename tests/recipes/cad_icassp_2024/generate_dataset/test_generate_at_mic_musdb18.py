"""Test the generate dataset script for the ICASSP 2024 CAD recipe."""
# pylint:: disable=import-error
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import lfilter

from recipes.cad_icassp_2024.generate_dataset.generate_at_mic_musdb18 import (
    apply_hrtf,
    find_precreated_samples,
    load_hrtf_signals,
    normalise_lufs_level,
)

# Create a mock object for the pyln.Meter class


@pytest.fixture(name="temp_dir_with_samples")
def fixture_temp_dir_with_samples(tmp_path):
    source_dir = Path(tmp_path)
    sample_dirs = ["sample_dir1", "sample_dir2"]
    for sample_dir in sample_dirs:
        sample_dir_path = Path(source_dir) / "train" / sample_dir
        sample_dir_path.mkdir(exist_ok=True, parents=True)
        sample_files = ["sample1.wav", "sample2.wav"]
        for sample_file in sample_files:
            with open(sample_dir_path / sample_file, "w", encoding="utf-8") as f:
                f.write("Sample content")
    return source_dir


def test_apply_hrtf_valid_signal():
    # Test the function with a valid signal and HRTF
    signal = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    hrtf_left = np.array([[0.5, 0.3], [0.2, 0.1]], dtype=np.float64)
    hrtf_right = np.array([[0.2, 0.1], [0.5, 0.3]], dtype=np.float64)

    result = apply_hrtf(signal, hrtf_left, hrtf_right)

    # Verify the result for each channel
    expected_left_ear = lfilter(hrtf_left[:, 0], 1, signal[:, 0]) + lfilter(
        hrtf_right[:, 0], 1, signal[:, 1]
    )
    expected_right_ear = lfilter(hrtf_left[:, 1], 1, signal[:, 0]) + lfilter(
        hrtf_right[:, 1], 1, signal[:, 1]
    )

    expected_result = np.stack([expected_left_ear, expected_right_ear], axis=1)

    np.testing.assert_allclose(result, expected_result, rtol=1e-3, atol=1e-6)


def test_apply_hrtf_zero_signal():
    # Test the function with a zero signal and HRTF
    signal = np.zeros((3, 2), dtype=np.float64)
    hrtf_left = np.array([[0.5, 0.3], [0.2, 0.1]], dtype=np.float64)
    hrtf_right = np.array([[0.2, 0.1], [0.5, 0.3]], dtype=np.float64)

    result = apply_hrtf(signal, hrtf_left, hrtf_right)

    # The result should also be all zeros
    expected_result = np.zeros((3, 2), dtype=np.float64)

    np.testing.assert_allclose(result, expected_result, rtol=1e-3, atol=1e-6)


def test_load_hrtf_signals(tmp_path):
    # Create temporary files for testing
    np.random.seed(2024)
    hrtf_path = Path(tmp_path) / "hrtf_directory"
    hrtf_path.mkdir(exist_ok=True, parents=True)

    hp = {
        "mic": "mic_name",
        "subject": "subject_name",
        "left_angle": -30,
        "right_angle": 45,
    }

    left_file = hrtf_path / f"{hp['mic']}-{hp['subject']}-n{abs(hp['left_angle'])}.wav"
    right_file = (
        hrtf_path / f"{hp['mic']}-{hp['subject']}-p{abs(hp['right_angle'])}.wav"
    )

    left_signal = np.random.rand(44100)
    right_signal = np.random.rand(44100)

    wavfile.write(left_file, 44100, left_signal)
    wavfile.write(right_file, 44100, right_signal)

    # Call the function
    left_result, right_result = load_hrtf_signals(str(hrtf_path), hp)

    # Verify that the function returns the expected signals
    assert np.sum(left_result) == pytest.approx(np.sum(left_signal))
    assert np.sum(right_result) == pytest.approx(np.sum(right_signal))


def test_normalise_lufs_level_same_loudness():
    # Test when the signal and reference signal have the same loudness
    np.random.seed(2024)
    signal = np.random.rand(44100)
    sample_rate = 44100.0

    result = normalise_lufs_level(signal, signal, sample_rate)

    # Since the loudness is the same, the result should be the same as the input signal
    assert np.sum(result) == pytest.approx(np.sum(signal))


def test_normalise_lufs_level_reference_quieter():
    # Test when the reference signal is quieter than the signal
    np.random.seed(2024)
    signal = np.random.rand(44100)
    reference_signal = np.random.rand(44100) / 0.2
    sample_rate = 44100.0

    result = normalise_lufs_level(signal, reference_signal, sample_rate)

    # The result should be a normalized version of the signal so
    # that it matches the reference loudness
    expected_result = 109580.6307104611
    assert np.sum(result) == pytest.approx(expected_result)


def test_normalise_lufs_level_reference_louder():
    # Test when the reference signal is louder than the signal
    np.random.seed(2024)
    signal = np.random.rand(44100)
    reference_signal = np.random.rand(44100) * 0.2
    sample_rate = 44100.0

    result = normalise_lufs_level(signal, reference_signal, sample_rate)

    # The result should be a normalized version of the signal so
    # that it matches the reference loudness
    expected_result = 4383.225228418446
    assert np.sum(result) == pytest.approx(expected_result)


def test_find_precreated_samples(temp_dir_with_samples):
    # Get the list of precreated samples from the temporary directory
    source_dir = temp_dir_with_samples
    precreated_samples = find_precreated_samples(source_dir)

    # Check if the expected sample files are in the result
    assert "sample_dir1" in precreated_samples
    assert "sample_dir1" in precreated_samples


def test_find_precreated_samples_empty_directory(tmp_path):
    # Test when the source directory is empty
    source_dir = Path(tmp_path)
    precreated_samples = find_precreated_samples(source_dir)

    # The result should be an empty list
    assert precreated_samples == []
