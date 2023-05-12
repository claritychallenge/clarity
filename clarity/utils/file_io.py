"""File I/O functions."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import soundfile
from numpy import ndarray
from soundfile import SoundFile

from clarity.utils.signal_processing import resample

# Function for reading and writing jsonl files


def read_jsonl(filename: str) -> list:
    """Read a jsonl file into a list of dictionaries."""
    with open(filename, encoding="utf-8") as fp:
        records = [json.loads(line) for line in fp]
    return records


def write_jsonl(filename: str, records: list) -> None:
    """Write a list of dictionaries to a jsonl file."""
    with open(filename, "a", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")


# Functions for reading and writing audio signals.
#
# - Signals are passed and returns as numpy arrays of floats of shape
#   (n_samples, n_channels). Mono signals can be passed as (n_samples, 1)
#   but are returned as (n_samples,) which is more convenient for most purposes.
#   Floating point and 16 bit integer file formats are supported.
# - Signals are assumed to be in the range [-1.0 to 1.0) and will be scaled to
#   and from -32768 to 32767 when writing to or reading from 16 bit files.
#   On writing to 16 bit files, overflow can be set to silently wrap or to raise
#   an error.
# - Any sample rate can be used for writing and floating points values are
#   rounded to integers. For reading, the 'expected sample rate' is supplied and
#   the file is resample if necessary or can be set to throw an error. Again,
#   this is more convenient for most purposes.


def write_signal(
    filename: str | Path,
    signal: ndarray,
    sample_rate: float,
    floating_point: bool = True,
    strict: bool = False,
) -> None:
    """Write a signal as fixed or floating point wav file.

    Signals are passed as numpy arrays of floats of shape (n_samples, n_channels)
    for n_channels >= 1 or (n_samples,) for n_channels = 1.

    Signals are floating point in the range [-1.0 to 1.0) but can be written
    as wav file with either 16 bit integers or floating point. In the former,
    the signals are scaled to map to the range -32768 to 32767 and clipped
    if necessary.

    NB: setting 'strict' to True will raise error on overflow. This would be
    a more natural default but it would break existing code that did not
    check for overflow.

    Args:
        filename (str|Path): name of file in to write to.
        signal (ndarray): signal to write.
        sample_rate (float): sampling frequency.
        floating_point (bool): write as floating point else an ints (default: True).
        strict (bool): raise error if signal out of range for int16 (default: False).
    """

    if floating_point is False:
        subtype = "PCM_16"
        # Signal is float and we want to convert to int16
        # *NB* Not  *= in next line as we need to make a copy
        signal = signal * 32768.0
        if np.max(signal) > 32767.0 or np.min(signal) < -32768.0:
            if strict:
                raise ValueError("Signal out of range [-1.0, 1.0)")
            logging.warning(
                f"Writing {filename}. Signal out of range [-1.0, 1.0) - clipping."
            )
            signal = np.clip(signal, -32768.0, 32767.0)
        signal = signal.astype(np.dtype("int16"))
    else:
        subtype = "FLOAT"

    soundfile.write(filename, signal, sample_rate, subtype=subtype)


def read_signal(
    filename: str | Path,
    sample_rate: float = 0.0,
    offset: int | float = 0,  # offset can be in samples or seconds
    n_samples: int = -1,
    n_channels: int = 0,
    offset_is_samples: bool = False,
    allow_resample: bool = True,
) -> ndarray:
    """Read a wav format audio file and return as numpy array of floats.

    The returned value will be a numpy array of floats of shape (n_samples,
    n_channels) for n_channels >= 2, and shape (n_samples,) for n_channels = 1.

    If n_samples is set to a value other than -1, the specified number of samples
    will be read, or until the end of the file. An 'offset' can be set to start
    reading from a specified sample or time (in seconds).

    The expected number of channels can be specified. If the file has a different
    number of channels, an error will be raised.

    The expected sample rate can be specified. If the file has a different sample
    the file will be resampled to the expected sample rate, unless
    'allow_resample' is set to False, in which case an error will be raised.

    Args:
        filename (str|Path): Name of file to read
        sample_rate (float): The expected sample rate (default: 0.0 = any rate OK)
        offset (int, optional): Offset in samples or seconds (from start).
            Default is 0.
        n_samples (int): Number of samples.
        n_channels (int): expected number of channel (default: 0 = any number OK)
        offset_is_samples (bool): is offset measured in samples, (True) or
            seconds (False) (default: False)
        allow_resample (bool): allow resampling if sample rate is different
            from expected rate. Else raise error (default: True)

    Returns:
        np.ndarray: audio signal
    """

    wave_file = SoundFile(filename)

    if n_channels not in (0, wave_file.channels):
        raise ValueError(
            f"Wav file ({filename}) was expected to have {n_channels} channels."
        )

    if not offset_is_samples:  # Default behaviour
        offset = np.rint(offset * wave_file.samplerate).astype(int)

    if offset != 0:
        wave_file.seek(offset)

    signal = wave_file.read(frames=n_samples)

    if sample_rate not in (0, wave_file.samplerate):
        if allow_resample:
            signal = resample(signal, wave_file.samplerate, sample_rate)
        else:
            raise ValueError(
                f"Sample rate of {wave_file.samplerate} "
                "does not match expected {sample_rate}"
            )

    return signal
