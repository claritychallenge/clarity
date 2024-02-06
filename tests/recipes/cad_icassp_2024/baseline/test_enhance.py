"""Tests for the enhance module"""
# pylint:: disable=import-error
from pathlib import Path

import numpy as np
import pytest
import torch
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram, Listener
from clarity.utils.flac_encoder import read_flac_signal
from recipes.cad_icassp_2024.baseline.enhance import (
    decompose_signal,
    process_remix_for_listener,
    save_flac_signal,
)

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources" / "recipes" / "cad_icassp_2024"


def test_save_flac_signal(tmp_path):
    """Test save flac signal"""
    np.random.seed(2024)
    sample_rate = 44100
    duration = 0.5
    signal = np.random.rand(int(sample_rate * duration))

    filename = Path(tmp_path) / "signal.flac"
    save_flac_signal(signal, filename, sample_rate, sample_rate)

    signal_out, sample_rate_out = read_flac_signal(filename)
    assert np.sum(signal) == pytest.approx(11040.050741283)
    assert np.sum(signal_out) == pytest.approx(11039.716)
    assert sample_rate_out == sample_rate


@pytest.mark.parametrize(
    "separation_model",
    [
        pytest.param("demucs"),
        pytest.param("openunmix", marks=pytest.mark.slow),
    ],
)
def test_decompose_signal(separation_model):
    """Takes a signal and decomposes it into
    VDBO sources using the HDEMUCS model"""
    np.random.seed(2024)
    # Load Separation Model
    if separation_model == "demucs":
        model = HDEMUCS_HIGH_MUSDB.get_model()
        model_sample_rate = HDEMUCS_HIGH_MUSDB.sample_rate
        sources_order = model.sources

    elif separation_model == "openunmix":
        model = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq")
        model_sample_rate = model.sample_rate
        sources_order = ["vocals", "drums", "bass", "other"]

    device = torch.device("cpu")
    model.to(device)

    # Create a mock signal to decompose
    sample_rate = 44100
    duration = 0.5
    signal = np.random.uniform(size=(1, 2, int(sample_rate * duration))).astype(
        np.float32
    )

    # Call the decompose_signal function and check that the output has the expected keys
    cfs = np.array([250, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000])
    audiogram = Audiogram(levels=np.ones(9), frequencies=cfs)
    listener = Listener(audiogram, audiogram)
    output = decompose_signal(
        model,
        model_sample_rate,
        signal,
        sample_rate,
        device,
        sources_order,
        listener,
    )
    expected_results = np.load(
        RESOURCES / f"test_enhance.test_decompose_signal_{separation_model}.npy",
        allow_pickle=True,
    )[()]

    for key, item in output.items():
        np.testing.assert_array_almost_equal(item, expected_results[key])


@pytest.mark.parametrize(
    "apply_compressor",
    [
        True,
        False,
    ],
)
def test_process_remix_for_listener(apply_compressor):
    """Test the process remix for listener"""
    np.random.seed(2024)
    sample_rate = 44100
    duration = 0.5
    signal = np.random.uniform(size=(2, int(duration * sample_rate)))

    audiogram = Audiogram(
        levels=np.ones(9),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000]),
    )
    listener = Listener(audiogram_left=audiogram, audiogram_right=audiogram)
    enhancer = NALR(nfir=220, sample_rate=16000)
    compressor = Compressor(
        threshold=0.35, attenuation=0.1, attack=50, release=1000, rms_buffer_size=0.064
    )

    output = process_remix_for_listener(
        signal, enhancer, compressor, listener, apply_compressor=apply_compressor
    )

    if apply_compressor:
        expected_results = np.load(
            RESOURCES / "test_enhance.test_process_remix_for_listener_w_compressor.npy",
            allow_pickle=True,
        )[()]
    else:
        expected_results = np.load(
            RESOURCES
            / "test_enhance.test_process_remix_for_listener_wo_compressor.npy",
            allow_pickle=True,
        )[()]

    assert np.sum(output) == pytest.approx(np.sum(expected_results))
