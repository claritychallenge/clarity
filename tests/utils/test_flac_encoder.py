"""Tests for the FlacEncoder class."""
# pylint: disable=import-error
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
    signal = np.random.uniform(-1, 1, int(0.5 * sample_rate))
    signal_int16 = signal * 32768.0
    signal_int16 = np.clip(signal_int16, -32768, 32767).astype(np.int16)

    # write the encoded bytes to a temporary file
    with NamedTemporaryFile(suffix=".flac", delete=False) as tmpfile:
        encoder = FlacEncoder()
        # encode
        _ = encoder.encode(
            signal_int16, sample_rate=sample_rate, output_file=tmpfile.name
        )
        # decode
        decoded_signal, decoded_sr = encoder.decode(Path(tmpfile.name))

    # check that the decoded signal matches the original signal
    assert np.sum(signal_int16) == pytest.approx(
        np.sum(decoded_signal),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )

    # check that the sample rate of the decoded signal is correct
    assert decoded_sr == sample_rate
