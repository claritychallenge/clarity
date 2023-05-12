"""Test the file_io module."""
import numpy as np
import pytest

from clarity.utils.file_io import read_jsonl, read_signal, write_jsonl, write_signal


def test_read_jsonl():
    """Test the read_jsonl function."""
    expected = [
        {"id": 1, "name": "xxx"},
        {"id": 2, "name": "yyy"},
        {"id": 3, "name": "zzz"},
    ]
    data = read_jsonl("tests/test_data/filetypes/valid.jsonl")
    assert data == expected


@pytest.mark.parametrize(
    "records",
    [
        [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
    ],
)
def test_jsonl_write_read_loop(records, tmp_path):
    """Test the write_jsonl and read_jsonl functions."""
    write_jsonl(tmp_path / "test1.jsonl", records)
    assert read_jsonl(tmp_path / "test1.jsonl") == records


@pytest.mark.parametrize(
    "records",
    [
        [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
    ],
)
def test_jsonl_append_read_loop(records, tmp_path):
    """Test the write_jsonl correctly appends to existing files."""
    write_jsonl(tmp_path / "test2.jsonl", records)
    write_jsonl(tmp_path / "test2.jsonl", records)
    assert read_jsonl(tmp_path / "test2.jsonl") == records + records


@pytest.mark.parametrize(
    "signal, floating_point, strict",
    [
        (np.array([1.1, -2.0, 0.0, 44.0, -54.0]), True, True),  # float32
        (np.array([0.1, -0.2, 0.1, -0.5, 0.99]), False, True),  # int16
        (np.array([0.1, -0.2, 0.1, -1.00, 0.99]), False, True),  # int16
        (np.array([0.1, -0.2, 0.1, -1.00, 1.99]), False, False),  # int16
        (np.array([0.1, -0.2, 0.1, -1.00, 0.99]), False, True),  # int16
    ],
)
def test_write_read_loop(tmp_path, signal, floating_point, strict):
    """Test write_signal and read_signal"""
    tmp_filename = tmp_path / "test.wav"

    sample_rate = 16000.0

    write_signal(
        tmp_filename,
        signal,
        sample_rate=int(sample_rate),  # <-- Sample rate needs to be cast to int
        floating_point=floating_point,
        strict=strict,
    )
    result = read_signal(tmp_filename, sample_rate=sample_rate)

    # Some precision is lost as convert to int16 and back again
    assert result.shape == signal.shape
    # The test where strict is False has overflow which is not caught and hence
    # reading back the signal has changed
    if strict:
        assert result == pytest.approx(signal, abs=1.0 / 16384)
    else:
        # Deliberate fail: shows why strict is True is needed
        assert result != pytest.approx(signal, abs=1.0 / 16384)


def test_write_mono_as_2D_signal(tmp_path):
    """Test special case of writing signals for shape [N, 1]"""
    tmp_filename_1d = tmp_path / "test_1d.wav"
    tmp_filename_2d = tmp_path / "test_2d.wav"
    signal_1d = np.ones((10,)) * 0.5
    signal_2d = np.ones((10, 1)) * 0.5

    write_signal(tmp_filename_1d, signal_1d, sample_rate=16000)
    write_signal(tmp_filename_2d, signal_2d, sample_rate=16000)

    result_1d = read_signal(tmp_filename_1d)
    result_2d = read_signal(tmp_filename_2d)

    # Both 1 and 2D signals should be read back as 1D
    assert result_1d.shape == (10,)
    assert result_2d.shape == (10,)


def test_read_write_multichannel(tmp_path):
    """Test write_signal and read_signal"""
    tmp_filename = tmp_path / "test.wav"

    signal = np.ones((2, 10)) * 0.5
    sample_rate = 16000.0

    write_signal(
        tmp_filename,
        signal,
        sample_rate=int(sample_rate),  # <-- Sample rate needs to be cast to int
        floating_point=False,
        strict=True,
    )
    result = read_signal(tmp_filename, sample_rate=sample_rate)

    # Some precision is lost as convert to int16 and back again
    assert result.shape == signal.shape


def test_read_write_sample_mismatch_error(tmp_path):
    """Should raise an error if the sample rate is not the same"""
    tmp_filename = tmp_path / "test.wav"

    signal = np.ones((2, 10)) * 0.5
    sample_rate = 16000.0

    write_signal(
        tmp_filename,
        signal,
        sample_rate=int(sample_rate),  # <-- Sample rate needs to be cast to int
        floating_point=False,
        strict=True,
    )

    with pytest.raises(ValueError):
        read_signal(tmp_filename, sample_rate=8000, allow_resample=False)


def test_read_write_channel_mismatch(tmp_path):
    """Should raise an error if the sample rate is not the same"""
    tmp_filename = tmp_path / "test.wav"

    signal = np.ones((10, 2)) * 0.5
    sample_rate = 16000.0

    write_signal(
        tmp_filename,
        signal,
        sample_rate=int(sample_rate),  # <-- Sample rate needs to be cast to int
        floating_point=False,
        strict=True,
    )

    # Correct number of channels - will read OK
    read_signal(tmp_filename, sample_rate=16000, n_channels=2)

    # Incorrect number of channels - will raise an error
    with pytest.raises(ValueError):
        read_signal(tmp_filename, sample_rate=16000, n_channels=1)


def test_read_write_sample_with_resample(tmp_path):
    """Should if the sample rate is not the same and allow_resample is True"""
    tmp_filename = tmp_path / "test.wav"
    signal = np.ones((10, 2)) * 0.5
    sample_rate = 16000.0

    write_signal(
        tmp_filename,
        signal,
        sample_rate=int(sample_rate),  # <-- Sample rate needs to be cast to int
        floating_point=False,
        strict=True,
    )

    x = read_signal(tmp_filename, sample_rate=8000, allow_resample=True)

    assert x.shape == (5, 2)


def test_write_clipping(tmp_path):
    """Should raise an error if the sample rate is not the same"""
    tmp_filename = tmp_path / "test.wav"

    signal = np.array([-1.0, 0.0, 1.0])  # <-- Clipping because 1.0 not OK

    # strict=True, i.e. Clipping not allowed - will throw an error

    with pytest.raises(ValueError):
        write_signal(
            tmp_filename,
            signal,
            sample_rate=16000,
            floating_point=False,
            strict=True,
        )

    # strict=False, i.e. Clipping allowed - will write OK but log warning
    write_signal(
        tmp_filename,
        signal,
        sample_rate=16000,
        floating_point=False,
        strict=False,
    )

    signal_read = read_signal(tmp_filename, sample_rate=16000)

    # Note that the +1.0 is clipped to 0.99996948
    assert signal_read == pytest.approx(np.array([-1.0, 0, 0.99996948]))
    # This is the standard behaviour of soundfile and arises due to the
    # asymmetric nature of the int16 format, i.e. -32768 to 32767


def test_read_write_with_offset(tmp_path):
    """Should if the sample rate is not the same and allow_resample is True"""
    tmp_filename = tmp_path / "test.wav"
    signal = np.ones((10, 2)) * 0.5
    sample_rate = 10000

    write_signal(
        tmp_filename,
        signal,
        sample_rate=sample_rate,  # <-- Sample rate needs to be cast to int
        floating_point=False,
        strict=True,
    )

    x1 = read_signal(tmp_filename, offset=3, offset_is_samples=True)
    assert x1.shape == (7, 2)

    x2 = read_signal(tmp_filename, offset=0.0003, offset_is_samples=False)
    assert x2.shape == (7, 2)
