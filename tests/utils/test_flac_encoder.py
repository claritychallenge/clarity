"""Tests for the FlacEncoder class."""
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from clarity.utils.flac_encoder import FlacEncoder


def test_encode_decode():
    """Test that the FlacEncoder can encode and decode a signal."""
    # create a random signal
    np.random.seed(0)

    sample_rate = 8000
    signal = np.random.randint(-32768, 32768, int(0.5 * sample_rate)).astype(np.int16)

    # encode the signal to FLAC
    encoder = FlacEncoder()
    encoded_bytes = encoder.encode(signal, sample_rate=sample_rate, output_file=None)

    # write the encoded bytes to a temporary file
    with NamedTemporaryFile(suffix=".flac", delete=False) as tmpfile:
        tmpfile.write(encoded_bytes)
        tmpfile.flush()

        # decode the FLAC file
        decoded_signal, decoded_sr = FlacEncoder.decode(Path(tmpfile.name))

    # check that the decoded signal matches the original signal
    assert np.sum(signal) == pytest.approx(
        np.sum(decoded_signal), rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    # check that the sample rate of the decoded signal is correct
    assert decoded_sr == sample_rate
