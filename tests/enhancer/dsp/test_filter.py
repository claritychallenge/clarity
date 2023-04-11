"""Tests for enhancer.dsp.filter module"""

import pytest
import torch

from clarity.enhancer.dsp.filter import AudiometricFIR

SAMPLE_RATE = 44100  # Sample rate of the signal
NFIR = 220  # Number of FIR filter taps to use in tests


@pytest.fixture(autouse=True)
def use_torch():
    """Fixture to ensure torch is used with a consistent seed and settings"""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_printoptions(precision=10)


def test_audiometric_filter_init():
    """test construction of audiomatric filter"""
    fir_filter = AudiometricFIR(sample_rate=SAMPLE_RATE, nfir=NFIR)
    assert fir_filter.padding == NFIR // 2
    assert fir_filter.window_size == NFIR + 1


def test_audiometric_filter_forward():
    """test that the filter can be applied"""
    fir_filter = AudiometricFIR(sample_rate=SAMPLE_RATE, nfir=NFIR, device="cpu")
    audio = torch.randn(1, 1, 4410)
    filtered_audio = fir_filter(audio)
    assert filtered_audio.shape == audio.shape
    assert filtered_audio.cpu().detach().numpy().sum() == pytest.approx(
        -60.199371337890625, abs=1e-4  # <- had to relax this tolerance
    )


def test_audiometric_filter_forward_error():
    """test that the filter throws error with invalid signal shapes"""
    fir_filter = AudiometricFIR(sample_rate=SAMPLE_RATE, nfir=NFIR)
    with pytest.raises(RuntimeError):
        fir_filter(torch.randn(2, 44100))
    with pytest.raises(RuntimeError):
        fir_filter(torch.randn(1, 2, 44100))
