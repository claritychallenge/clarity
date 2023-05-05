"""Tests for the FlacEncoder class."""
# pylint: disable=import-error
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from clarity.utils.flac_encoder import FlacEncoder, read_flac_signal


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


def test_read_flac_signal(tmp_path):
    # Create a test FLAC file
    np.random.seed(2023)

    filename = tmp_path / "test.flac"

    sample_rate = 16000
    signal = np.random.rand(1600)
    scale_factor = np.max(np.abs(signal))

    signal_scaled = signal / scale_factor
    signal_scaled = signal_scaled * 32768.0
    signal_scaled = np.clip(signal_scaled, -32768.0, 32767.0)
    signal_scaled = signal_scaled.astype(np.dtype("int16"))

    flac_encoder = FlacEncoder()
    flac_encoder.encode(signal_scaled, sample_rate, filename)

    # Create a test scale factor file
    scale_filename = tmp_path / "test.txt"
    with open(scale_filename, "w", encoding="utf-8") as fp:
        fp.write(str(scale_factor))

    # Call the function and check the output
    signal_out, sample_rate_out = read_flac_signal(filename)

    # As a result of the quantization, the signal is not exactly the same
    # after encoding and decoding, so I'm changing the tolerance
    # for this test
    # np.sum(signal_out) = 2190.4271092907347
    # np.sum(signal) = 2190.494140495932

    assert np.sum(signal_out) == pytest.approx(np.sum(signal), rel=1e-4, abs=1e-4)
    assert sample_rate_out == sample_rate
