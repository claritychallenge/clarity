import numpy as np
import pytest

from clarity.enhancer.multiband_compressor.compressor_qmul import Compressor


@pytest.fixture
def default_compressor():
    """Return a Compressor object with default parameters."""
    return Compressor()


@pytest.fixture
def custom_compressor():
    """Return a Compressor object with custom parameters."""
    return Compressor(
        threshold=-30.0,
        ratio=4.0,
        attack=10.0,
        release=100.0,
        makeup_gain=1.25,
        sample_rate=44100.0,
        knee_width=10.0,
    )


@pytest.fixture
def random_signal():
    """Generate a random signal for testing."""
    return np.random.randn(1, 1000)


def test_initialization_default(default_compressor):
    """Test the initialization of the Compressor class."""
    assert default_compressor.threshold == 0.0
    assert default_compressor.ratio == 1.0
    assert default_compressor.attack == 15.0
    assert default_compressor.release == 100.0
    assert default_compressor.makeup_gain == 0.0
    assert default_compressor.sample_rate == 44100.0
    assert default_compressor.knee_width == 0.0


def test_initialization_custom(custom_compressor):
    """Test the initialization of the Compressor class with custom parameters."""
    assert custom_compressor.threshold == -30.0
    assert custom_compressor.ratio == 4.0
    assert custom_compressor.attack == 10.0
    assert custom_compressor.release == 100.0
    assert custom_compressor.makeup_gain == 1.25
    assert custom_compressor.sample_rate == 44100.0
    assert custom_compressor.knee_width == 10.0


def test_call_default(default_compressor, random_signal):
    """Test the call method of the Compressor class with default parameters."""
    processed_signal = default_compressor(random_signal)
    assert processed_signal.shape == random_signal.shape


def test_call_custom(custom_compressor, random_signal):
    """Test the call method of the Compressor class with custom parameters."""
    processed_signal = custom_compressor(random_signal)
    assert processed_signal.shape == random_signal.shape
