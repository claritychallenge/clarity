""" Test module for the source separation support"""
# pylint: disable=import-error
from pathlib import Path

import numpy as np
import torch

from clarity.utils.source_separation_support import get_device, separate_sources

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources" / "utils"


def test_separate_sources():
    """Test that the separate_sources function returns the expected output"""
    np.random.seed(123456789)

    # Create a dummy model
    class DummyModel(torch.nn.Module):  # pylint: disable=too-few-public-methods
        """Dummy source separation model"""

        def __init__(self, sources):
            """dummy init"""
            super().__init__()
            self.sources = sources

        def forward(self, x):
            """dummy forward"""
            return torch.Tensor(
                np.random.uniform(size=(x.shape[0], len(self.sources), *x.shape[1:]))
            )

    # Set up some dummy input data
    batch_size = 1
    num_channels = 1
    length = 1
    sample_rate = 16000
    sources = ["vocals", "drums", "bass", "other"]
    mix = np.random.randn(batch_size, num_channels, length * sample_rate)
    device = torch.device("cpu")

    # Create a dummy model
    model = DummyModel(sources)

    # Call separate_sources
    output = separate_sources(model, mix, sample_rate, device=device)

    expected_results = np.load(
        RESOURCES / "test_esource_separation_support.test_separate_sources.npy",
        allow_pickle=True,
    )
    # Check that the output has the correct shape
    assert output.shape == expected_results.shape
    np.testing.assert_array_almost_equal(output, expected_results)


def test_get_device():
    """Test the correct device selection given the inputs"""
    # Test default case (no argument passed)
    device, device_type = get_device(None)
    assert (
        device == torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    assert device_type == "cuda" if torch.cuda.is_available() else "cpu"
