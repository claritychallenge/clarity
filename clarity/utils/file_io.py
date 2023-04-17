"""File I/O functions for jsonl files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile
from numpy import ndarray
from soundfile import SoundFile


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


def write_signal(
    filename: str | Path,
    signal: ndarray,
    sample_rate: float,
    floating_point: bool = True,
    strict: bool = False,
) -> None:
    """Write a signal as fixed or floating point wav file.

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
        signal = signal * 32768
        if strict and (np.max(signal) > 32767 or np.min(signal) < -32768):
            raise ValueError("Signal out of range -1.0 to 1.0")
        signal = signal.astype(np.dtype("int16"))
    else:
        subtype = "FLOAT"

    soundfile.write(filename, signal, sample_rate, subtype=subtype)


def read_signal(
    filename: str | Path,
    sample_rate: float = 0,
    offset: int | float = 0,  # offset can be in samples or seconds
    n_samples: int = -1,
    n_channels: int = 0,
    offset_is_samples: bool = False,
    allow_resample: bool = True,
) -> ndarray:
    """Read a wavefile and return as numpy array of floats.

    Args:
        filename (str|Path): Name of file to read
        offset (int, optional): Offset in samples or seconds (from start). Default is 0.
        nsamples (int): Number of samples.
        nchannels (int): expected number of channel (default: 0 = any number OK)
        offset_is_samples (bool): measurement units for offset (default: False)

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
            signal = scipy.signal.resample(
                signal, int(sample_rate * signal.shape[0] / wave_file.samplerate)
            )
        else:
            raise ValueError(
                f"Sample rate of {wave_file.samplerate} "
                "does not match expected {sample_rate}"
            )

    return signal
