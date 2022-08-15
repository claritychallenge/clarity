"""Temporal smearing component of MSBG model."""
import math

import numpy as np

FFT_SIZE = 512
FRAME_SIZE = 256
SHIFT = 64


def audfilt(rl, ru, sampfreq, asize=256):
    """Calculate an auditory filter array.

    Args:
        rl (float): broadening factor on the lower side
        ru (float): broadening factor on the upper side
        sampfreq (float): signal sampling frequency
        asize (int, optional): number of taps in filter (default: {256})

    Returns:
        ndarray: A bank of auditory filters stored as 2-d numpy array

    """
    asize = int(asize)
    aud_filter = np.zeros((asize, asize))

    aud_filter[0, 0] = 1.0
    # Dividing by the erb to remove spectral tilt from the excitation pattern
    aud_filter[0, 0] = aud_filter[0, 0] / ((rl + ru) / 2)

    g = np.zeros(asize)
    for i in np.linspace(1, asize - 1, asize - 1, dtype=int):
        f_hz = i * np.divide(sampfreq, (2 * asize))
        f_erb = 24.7 * ((f_hz * 0.00437) + 1.0)
        # For lower side of the filter
        jj = np.arange(0, i)
        g = np.abs((i - jj) / i) * (4.0 * f_hz / (f_erb * rl))
        aud_filter[i, jj] = (1 + g) * np.exp(-g)
        # For upper side of the filter and centre
        jj = np.arange(i, asize)
        g = abs((i - jj) / i) * (4.0 * f_hz / (f_erb * ru))
        aud_filter[i, jj] = (1 + g) * np.exp(-g)
        aud_filter[i, :] = np.divide(aud_filter[i, :], (f_erb * (rl + ru) / (2 * 24.7)))

    return aud_filter


def make_smear_mat3(rl, ru, fs):
    """Make the smearing filter matrix.

    Args:
        rl (float): filter broadening factor on the lower side
        ru (float): filter broadening factor on the upper side
        fs (float): sampling frequency


    Returns:
        ndarray: The FFT_SIZE/2 X FFT_SIZE/2 smearing filter matrix

    """
    # FFTSIZE is assumed to contain a factor of 4
    assert FFT_SIZE % 4 == 0
    nyquist = int(FFT_SIZE / 2)
    half_nyquist = int(FFT_SIZE / 4)

    f_normal = audfilt(1, 1, fs, nyquist)
    f_wide = audfilt(rl, ru, fs, nyquist)
    # Extend the normal matrix so that the left-divide works better
    f_next = np.concatenate((f_normal, np.zeros((nyquist, half_nyquist))), axis=1)

    for i in np.arange(half_nyquist, nyquist):
        f_next[i, nyquist : min(2 * i - 1, 3 * half_nyquist)] = f_normal[
            i, 2 * i - nyquist : max(1, 2 * i - 3 * half_nyquist) : -1
        ]

    # This is equivalent to multiplying (convolving) the inverse of the
    # normal filters with the wide filters.
    f_smear = np.real(np.linalg.lstsq(f_next, f_wide, rcond=-1)[0])

    # Pruning to remove the extra bit
    f_smear = f_smear[0:nyquist, :]

    return f_smear


def smear3(f_smear, inbuffer):
    """Direct translation of smear3.m from MSBG hearing loss model.

    Args:
        f_smear (ndarray): The FFT_SIZE/2 X FFT_SIZE/2 smearing filter matrix
        inbuffer (ndarray): signal with prepended tone and noise

    Returns:
        ndarray: outbuffer

    """
    inlength = len(inbuffer)
    inpointer = 0
    outbuffer = np.zeros(int(np.ceil(inlength / SHIFT) + 3) * SHIFT)
    outpointer = 0

    overlaps = int(FRAME_SIZE / SHIFT)
    outwave = np.zeros([FRAME_SIZE, overlaps])
    buffer = np.arange(0, overlaps)
    nyquist = int(FFT_SIZE / 2)

    window = 0.5 - 0.5 * np.cos(
        2 * np.pi * (np.arange(1, FRAME_SIZE + 1) - 0.5) / FRAME_SIZE
    )
    window = window / math.sqrt(1.5)

    samplecount = min(FRAME_SIZE, inlength - inpointer)
    inwave = inbuffer[inpointer + np.arange(0, samplecount)]
    while samplecount > 0:
        winwave = np.zeros(FFT_SIZE)
        winwave[0:FRAME_SIZE] = window * inwave.flatten()
        spectrum = np.fft.fft(winwave, FFT_SIZE)
        power = spectrum[0:nyquist] * np.conj(spectrum[0:nyquist])
        mag = np.sqrt(power)
        phasor = spectrum[0:nyquist] / (mag + (mag == 0))
        smeared = np.dot(f_smear, power)
        spectrum[0:nyquist] = np.sqrt(smeared) * phasor
        spectrum[nyquist] = 0
        spectrum[(nyquist + 1) : FFT_SIZE] = np.conj(spectrum[nyquist - 1 : 0 : -1])
        winwave = np.real(np.fft.ifft(spectrum, FFT_SIZE))
        outwave[:, buffer[0]] = winwave[0:FRAME_SIZE] * window
        outframe = np.zeros(SHIFT)
        j = 0
        for i in np.arange(0, overlaps):
            outframe += outwave[j + np.arange(0, SHIFT), buffer[i]]
            j += SHIFT

        outbuffer[outpointer + np.arange(0, SHIFT)] = outframe
        outpointer += SHIFT

        buffer = np.roll(buffer, 1)
        inwave = np.append(inwave[SHIFT:FRAME_SIZE], np.zeros(SHIFT))
        inpointer += SHIFT
        samplecount = min(SHIFT, inlength - inpointer)
        inframe = inbuffer[inpointer + np.arange(0, samplecount)]
        inwave[FRAME_SIZE - SHIFT : FRAME_SIZE - SHIFT + samplecount] = inframe

    for k in np.arange(overlaps - 1, 0, -1):
        outframe = np.zeros(SHIFT)
        j = (overlaps - 1 - k) * SHIFT
        for i in np.arange(0, k):
            outframe += outwave[j + np.arange(0, SHIFT), buffer[i]]
            j += SHIFT
        outbuffer[outpointer + np.arange(0, SHIFT)] = outframe
        outpointer += SHIFT

    return outbuffer


class Smearer:
    """Class to hold the re-usable smearing filter."""

    rl: np.ndarray
    ru: np.ndarray
    fs: np.ndarray
    f_smear: np.ndarray

    def __init__(self, rl, ru, fs):
        self.rl = rl
        self.ru = ru
        self.fs = fs
        self.f_smear = make_smear_mat3(rl, ru, fs)

    def smear(self, input_signal: np.ndarray) -> np.ndarray:
        """Smear a given input signal."""
        return smear3(self.f_smear, input_signal)
