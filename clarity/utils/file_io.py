"""File I/O functions for jsonl files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile
from numpy import ndarray


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
