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
        gain=1.25,
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
    assert default_compressor.gain == 0.0
    assert default_compressor.sample_rate == 44100.0
    assert default_compressor.knee_width == 0.0


def test_initialization_custom(custom_compressor):
    """Test the initialization of the Compressor class with custom parameters."""
    assert custom_compressor.threshold == -30.0
    assert custom_compressor.ratio == 4.0
    assert custom_compressor.attack == 10.0
    assert custom_compressor.release == 100.0
    assert custom_compressor.gain == 1.25
    assert custom_compressor.sample_rate == 44100.0
    assert custom_compressor.knee_width == 10.0


def test_warning_threshold():
    """Test the warning for a threshold outside the recommended range."""
    with pytest.warns(UserWarning, match="Threshold outside the recommended range"):
        Compressor(threshold=10.0)


def test_warning_ratio():
    """Test the warning for a ratio outside the recommended range."""
    with pytest.warns(UserWarning, match="Ratio outside the recommended range"):
        Compressor(ratio=0.5)


def test_warning_attack():
    """Test the warning for an attack outside the recommended range."""
    with pytest.warns(UserWarning, match="Attack outside the recommended range"):
        Compressor(attack=0.05)


def test_warning_release():
    """Test the warning for a release outside the recommended range."""
    with pytest.warns(UserWarning, match="Release outside the recommended range"):
        Compressor(release=1500.0)


def test_warning_gain():
    """Test the warning for a gain outside the recommended range."""
    with pytest.warns(UserWarning, match="Make-up gain outside the recommended range"):
        Compressor(gain=25.0)


def test_call_default(default_compressor, random_signal):
    """Test the call method of the Compressor class with default parameters."""
    processed_signal = default_compressor(random_signal)
    assert processed_signal.shape == random_signal.shape


def test_call_custom(custom_compressor, random_signal):
    """Test the call method of the Compressor class with custom parameters."""
    processed_signal = custom_compressor(random_signal)
    assert processed_signal.shape == random_signal.shape
