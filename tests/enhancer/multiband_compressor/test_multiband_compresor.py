import numpy as np
import pytest

from clarity.enhancer.multiband_compressor.compressor_qmul import Compressor
from clarity.enhancer.multiband_compressor.multiband_compressor import (
    MultibandCompressor,
)


@pytest.fixture
def default_compressor():
    return Compressor()


def test_compressor_initialization(default_compressor):
    """Test the initialization of the Compressor class."""
    assert default_compressor.threshold == 0.0
    assert default_compressor.ratio == 1.0
    assert default_compressor.attack == 15.0
    assert default_compressor.release == 100.0
    assert default_compressor.makeup_gain == 0.0
    assert default_compressor.sample_rate == 44100.0


def test_compressor_signal_processing(default_compressor, rel_tolerance, abs_tolerance):
    """Test the signal processing of the Compressor class."""
    input_signal = np.array([[1, 2, 3, 4, 5]])
    output_signal = default_compressor(input_signal)
    # Add assertions to verify the output signal
    assert isinstance(output_signal, np.ndarray)
    assert len(output_signal) == len(input_signal)
    assert np.sum(output_signal) == pytest.approx(
        15, rel=rel_tolerance, abs=abs_tolerance
    )


@pytest.fixture
def default_multiband_compressor():
    """Fixture for the MultibandCompressor class with default parameters."""
    return MultibandCompressor(
        crossover_frequencies=np.array([250, 500, 1000, 2000, 4000]) * np.sqrt(2)
    )


def test_multiband_compressor_onefreq():
    """Test the initialization of the MultibandCompressor class with one frequency."""
    multiband_compressor = MultibandCompressor(crossover_frequencies=250)
    assert multiband_compressor.num_compressors == 2
    assert multiband_compressor.xover_freqs.shape[0] == 1


def test_multiband_compressor_initialization(
    default_multiband_compressor, rel_tolerance, abs_tolerance
):
    """Test the initialization of the MultibandCompressor class."""
    assert np.sum(default_multiband_compressor.xover_freqs) == pytest.approx(
        np.sum(
            np.array(
                [
                    250,
                    500,
                    1000,
                    2000,
                    4000,
                ]
            )
            * np.sqrt(2)
        ),
        rel=rel_tolerance,
        abs=abs_tolerance,
    )

    assert np.sum(default_multiband_compressor.xover_freqs) == pytest.approx(
        np.sum(np.array(default_multiband_compressor.xover_freqs)),
        rel=rel_tolerance,
        abs=abs_tolerance,
    )
    assert default_multiband_compressor.sample_rate == 44100.0
    assert default_multiband_compressor.num_compressors == 6
    assert default_multiband_compressor.attack == 15.0
    assert default_multiband_compressor.release == 100.0
    assert default_multiband_compressor.threshold == 0.0
    assert default_multiband_compressor.ratio == 1.0
    assert default_multiband_compressor.makeup_gain == 0.0
    assert default_multiband_compressor.knee_width == 0.0


def test_multiband_compressor_set_compressors(default_multiband_compressor):
    """Test the set_compressors method of the MultibandCompressor class."""
    default_multiband_compressor.set_compressors(
        attack=10.0, release=50.0, threshold=-20.0, ratio=2.0, makeup_gain=6.0
    )
    assert len(default_multiband_compressor.compressor) == 6


def test_multiband_compressor_call(default_multiband_compressor, rel_tolerance):
    """Test the __call__ method of the MultibandCompressor class."""
    np.random.seed(0)
    signal = np.random.rand(1000)

    default_multiband_compressor.set_compressors()
    compressed_signal = default_multiband_compressor(signal)
    assert isinstance(compressed_signal, np.ndarray)

    compressed_signal, bands = default_multiband_compressor(signal, return_bands=True)
    assert isinstance(compressed_signal, np.ndarray)
    assert isinstance(bands, np.ndarray)
    assert bands.shape[0] == 6

    assert np.sum(compressed_signal) == pytest.approx(
        441.09490986, rel=rel_tolerance, abs=0.0005
    )


def test_multiband_compressor_str(default_multiband_compressor):
    """Test the __str__ method of the MultibandCompressor class."""
    summary_text = str(default_multiband_compressor)
    assert "Multiband Compressor Summary:" in summary_text


def test_multiband_wrong_number_attack():
    """Test the initialization of the MultibandCompressor class with
    invalid parameters.
    """
    with pytest.raises(ValueError):
        MultibandCompressor(
            crossover_frequencies=np.array([250]),
            compressors_params={
                "attack": [1],
            },
        )


def test_multiband_wrong_number_release():
    """Test the initialization of the MultibandCompressor class with
    invalid parameters.
    """
    with pytest.raises(ValueError):
        MultibandCompressor(
            crossover_frequencies=np.array([250]),
            compressors_params={
                "release": [1],
            },
        )


def test_multiband_wrong_number_threshold():
    """Test the initialization of the MultibandCompressor class with
    invalid parameters.
    """
    with pytest.raises(ValueError):
        MultibandCompressor(
            crossover_frequencies=np.array([250]),
            compressors_params={
                "threshold": [1],
            },
        )


def test_multiband_wrong_number_ratio():
    """Test the initialization of the MultibandCompressor class with
    invalid parameters.
    """
    with pytest.raises(ValueError):
        MultibandCompressor(
            crossover_frequencies=np.array([250]),
            compressors_params={
                "ratio": [1],
            },
        )


def test_multiband_wrong_number_gain():
    """Test the initialization of the MultibandCompressor class with
    invalid parameters.
    """
    with pytest.raises(ValueError):
        MultibandCompressor(
            crossover_frequencies=np.array([250]),
            compressors_params={
                "makeup_gain": [1],
            },
        )


def test_multiband_wrong_number_knee_width():
    """Test the initialization of the MultibandCompressor class with
    invalid parameters.
    """
    with pytest.raises(ValueError):
        MultibandCompressor(
            crossover_frequencies=np.array([250]),
            compressors_params={
                "knee_width": [1],
            },
        )


def test_multiband_stereo_signal(default_multiband_compressor):
    """Test the __call__ method of the MultibandCompressor class with
    stereo signal."""
    signal = np.random.rand(2, 1000)

    default_multiband_compressor.set_compressors()
    compressed_signal = default_multiband_compressor(signal)
    assert isinstance(compressed_signal, np.ndarray)
    assert compressed_signal.shape == signal.shape
