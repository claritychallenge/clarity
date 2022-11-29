"""Support for the MSBG hearing loss model."""
import json
import logging
import math
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy
import scipy.signal
import soundfile
from soundfile import SoundFile

# measure rms parameters
WIN_SECS = 0.01

# read & write signal parameters
MSBG_FS = 44100
TEST_NBITS = 16

# fmt: off
HZ = np.array(
    [
        0.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,
        250.0, 315.0, 400.0, 500.0, 630.0, 750.0, 800.0, 1000.0, 1250.0, 1500.0,
        1600.0, 2000.0, 2500.0, 3000.0, 3150.0, 4000.0, 5000.0, 6000.0, 6300.0,
        8000.0, 9000.0, 10000.0, 11200.0, 12500.0, 14000.0, 15000.0, 16000.0,
        20000.0, 48000,
    ]
)

MIDEAR = np.array(
    [
        50.0, 39.6, 32.0, 25.85, 21.4, 18.5, 15.9, 14.1, 12.4, 11.0, 9.6, 8.3, 7.4,
        6.2, 4.8, 3.8, 3.3, 2.9, 2.6, 2.6, 4.5, 5.4, 6.1, 8.5, 10.4, 7.3, 7.0, 6.6,
        7.0, 9.2, 10.2, 12.2, 10.8, 10.1, 12.7, 15.0, 18.2, 23.8, 32.3, 50.0, 50.0,
    ]
)

# Free field (frontal)FF_ED correction for threshold (was ISO std Table 1 - 4.2 dB)
# i.e. relative to 0.0 dB at 1000 Hz, Shaw 1974

FF_ED = np.array(
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.9, 1.4,
        1.6, 1.7, 2.5, 2.7, 2.6, 2.6, 3.2, 5.2, 6.6, 12.0, 16.8, 15.3, 15.2, 14.2,
        10.7, 7.1, 6.4, 1.8, -0.9, -1.6, 1.9, 4.9, 2.0, -2.0, 2.5, 2.5, 2.5,
    ]
)

# DIFFUSE field ( relative to 0.0 dB at 1000Hz)
# from 2008 file [corrections08.m] used in Samsung collaboration

DF_ED = np.array(
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.4, 0.5, 1.0,
        1.6, 1.7, 2.2, 2.7, 2.9, 3.8, 5.3, 6.8, 7.2, 10.2, 14.9, 14.5, 14.4,
        12.7, 10.8, 8.9, 8.7, 8.5, 6.2, 5.0, 4.5, 4.0, 3.3, 2.6, 2.0, 2.0, 2.0,
    ]
)

# ITU Rec P 58 08/96 Head and Torso Simulator transfer fns. from Peter Hugher BTRL,
# 4-June-2001. Negative of values in Table 14a of ITU P58 (05/2013), accessible at
# http://www.itu.int/rec/T-REC-P.58-201305-I/en
# Freely available. Converts from ear reference point (ERP) to eardrum reference
# point (DRP). EXCEPT extra 2 points added for 20k & 48k by MAS, MAr 2012

ITU_HZ = np.array(
    [
        0, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 20000, 48000,
    ]
)
# Ear Reference Point to Drum Reference Point (ERP-DRP) transfer function,
# Table 14A/P.58, sect 6.2. NB negative of table since defined other way round.

ITU_ERP_DRP = np.array(
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.5, 0.6, 0.7, 1.1, 1.7, 2.6, 4.2,
        6.5, 9.4, 10.3, 6.6, 3.2, 3.3, 16, 14.4, 14.4, 14.4
    ]
)
# fmt: on

# Ideal pre-emphasis (starts off as SII 1997), then rescaled for Moore et al. 2008
# NB last two are -15dB/oct before rescaling below
# (Moore et al 2008 E&H paper suggests that shape would be better as -7.5
# dB/oct, at least up to 8, and -13 dB/oct above there.)

GEN_NOISE_HZ = np.array(
    [0, 100, 200, 450, 550, 707, 1000, 1414, 2000, 2828, 4000, 5656, 8000, 16e3, 32e3]
)
EMPHASIS = np.array(
    [0, 0.0, 0.0, 0, -0.5, -4.5, -9.0, -13.5, -18, -22.5, -27, -31.5, -36.0, -51, -66]
) * (7.5 / 9)


def read_gtf_file(gtf_file: str) -> Dict:
    """Read a gammatone filterbank file.

    List data is converted into numpy arrays.

    """

    # Fix filename if necessary
    dirname = os.path.dirname(os.path.abspath(__file__))
    gtf_file = os.path.join(dirname, gtf_file)
    with open(gtf_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    for key in data:
        if isinstance(data[key], list):
            data[key] = np.array(data[key])
    return data


def firwin2(
    n_taps: int,
    frequencies: np.ndarray,
    filter_gains: np.ndarray,
    window: Union[None, str, tuple] = None,
    antisymmetric: Optional[bool] = None,
):  # pylint: disable=W0613
    """FIR filter design using the window method.

    Partial implementation of scipy firwin2 but using our own MATLAB-derived fir2.

    Args:
        n_taps (int): The number of taps in the FIR filter.
        frequencies (ndarray): The frequency sampling points. 0.0 to 1.0 with 1.0 being Nyquist.
        filter_gains (ndarray): The filter gains at the frequency sampling points.
        window (string or (string, float), optional): See scipy.firwin2 (default: (None))
        antisymmetric (bool, optional): Unused but present to main compatability
            with scipy firwin2.

    Returns:
        ndarray:  The filter coefficients of the FIR filter, as a 1-D array of length n.

    """
    window_shape = None
    window_type = None
    if isinstance(window, tuple):
        window_type, window_param = window if window is not None else (None, 0)
    else:
        window_type = window

    order = n_taps - 1

    if window_type == "kaiser":
        window_shape = scipy.signal.kaiser(n_taps, window_param)

    if window_shape is None:
        filter_coef, _ = fir2(order, frequencies, filter_gains)
    else:
        filter_coef, _ = fir2(order, frequencies, filter_gains, window_shape)

    return filter_coef


def fir2(
    filter_length: int,
    frequencies: np.ndarray,
    filter_gains: np.ndarray,
    n_interpolate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """FIR arbitrary shape filter design using the frequency sampling method.

    Translation of MATLAB fir2.

    Args:
        filter_length (int): Order
        frequencies (ndarray): The frequency sampling points (0 < frequencies < 1) where 1 is Nyquist rate.
                        First and last elements must be 0 and 1 respectively.
        filter_gains (ndarray): The filter gains at the frequency sampling points.
        n_interpolate (int, optional): Number of points for freq response interpolation
            (default: max(smallest power of 2 greater than nn, 512))

    Returns:
        np.ndarray: nn + 1 filter coefficients, 1

    """
    # Work with filter length instead of filter order
    filter_length += 1

    if n_interpolate is None:
        wind = scipy.signal.hamming(filter_length)
    else:
        wind = n_interpolate
    n_interpolate = (
        2 ** np.ceil(math.log(filter_length) / math.log(2))
        if filter_length >= 1024
        else 512
    )

    lap = np.fix(n_interpolate / 25).astype(int)

    nbrk = max(len(frequencies), len(filter_gains))

    frequencies[0] = 0
    frequencies[nbrk - 1] = 1

    H = np.zeros(n_interpolate + 1)
    nint = nbrk - 1
    df = np.diff(frequencies, n=1)

    n_interpolate += 1
    nb = 0
    H[0] = filter_gains[0]

    for i in np.arange(nint):
        if df[i] == 0:
            nb = int(np.ceil(nb - lap / 2))
            ne: int = nb + lap - 1
        else:
            ne = int(np.fix(frequencies[i + 1] * n_interpolate)) - 1

        j = np.arange(nb, ne + 1)
        inc: Union[float, np.ndarray] = 0.0 if nb == ne else (j - nb) / (ne - nb)
        H[nb : (ne + 1)] = inc * filter_gains[i + 1] + (1 - inc) * filter_gains[i]
        nb = ne + 1

    dt = 0.5 * (filter_length - 1)
    rad = -dt * 1j * math.pi * np.arange(0, n_interpolate) / (n_interpolate - 1)
    H = H * np.exp(rad)

    H = np.concatenate((H, H[n_interpolate - 2 : 0 : -1].conj()))
    ht = np.real(np.fft.ifft(H))

    b = ht[0:filter_length] * wind

    return b, 1


def gen_tone(
    freq: float, duration: float, sample_frequency: float = 44100.0, level: float = 0.0
) -> np.ndarray:
    """Generate a pure tone.

    Args:
        freq (float): Frequency of tone in Hz.
        duration (float): Duration of tone in seconds.
        sample_frequency (float, optional): Sample rate of generated tone in Hz. (default: 44100)
        level (float, optional): Level of tone in dB SPL. (default: 0)

    Returns:
        np.ndarray
    """
    return (
        1.4142
        * np.power(10, (0.05 * level))
        * np.sin(
            2
            * np.pi
            * freq
            * np.arange(1, duration * sample_frequency + 1)
            / sample_frequency
        )
    )


def gen_eh2008_speech_noise(
    duration: float,
    sample_frequency: float = 44100.0,
    level: Optional[float] = None,
    supplied_b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate speech shaped noise.

    Start with white noise and re-shape to ideal SII, ie flat to 500 Hz, and sloping -9db/oct beyond that.

    Slightly different shape from SII stylised same as EarHEar 2008 paper, Moore et al.

    Args:
        duration (float): Duration of signal in seconds
        sample_frequency (float): Sampling rate
        level (float, optional): Normalise to level dB if present
        supplied_b (ndarray, optional): High-pass filter (default: uses built-in pre-emphasis filter)

    Returns:
        ndarray: Noise signal

    """
    sample_frequency = int(sample_frequency)
    n_samples = int(duration * sample_frequency)

    # this rescales so that we get -7.5 dB/oct up to 8kHz, and -13 dB/oct above that
    norm_freq = GEN_NOISE_HZ / (sample_frequency / 2)
    last_f_idx = np.max(np.where(norm_freq < 1))
    norm_freq = np.append(norm_freq[0 : last_f_idx + 1], 1)

    # -9 dB/oct constant slope
    emph_nyq = EMPHASIS[last_f_idx] + 9 * np.log10(norm_freq[last_f_idx]) / np.log10(2)
    norm_emph = np.append(EMPHASIS[0 : last_f_idx + 1], emph_nyq)
    m = np.exp(np.log(10) * norm_emph / 20)

    # Create type II filter with 10 msec window and even number of taps
    n_taps = int(2 * np.ceil(10 * (sample_frequency / 2000))) + 1
    b = (
        supplied_b
        if supplied_b is not None
        else firwin2(n_taps, norm_freq, m, window="hamming", antisymmetric=False)
    )

    # white noise, 0 DC
    n_burst = np.random.random((1, n_samples + len(b))) - 0.5

    # remove low-freq noise that may bias RMS estimate, -33dB at 50 Hz
    eh2008_nse = scipy.signal.lfilter(b, 1, n_burst)

    # high-pass filter to remove low freqs (will be 2-pass with filtfilt)
    high_pass_filter = scipy.signal.ellip(
        3, 0.1, 50, 100 / (sample_frequency / 2), "high"
    )
    padlen = 3 * (max(len(high_pass_filter[1]), len(high_pass_filter[0])) - 1)
    eh2008_nse = scipy.signal.filtfilt(
        *high_pass_filter, eh2008_nse, padlen=padlen
    ).flatten()

    # this introduces a delay so remove it, ie time-ADVANCE audio
    # compensating shift to time-align all filter outputs
    delay_shift = int(np.floor(len(b) / 2))
    valid_len = int(np.size(eh2008_nse) - delay_shift)  # _advance_ filter outputs
    # time advance
    eh2008_nse[0:valid_len] = eh2008_nse[delay_shift:]
    eh2008_nse = eh2008_nse[0:n_samples]

    if level is not None:
        eh2008_nse = (
            eh2008_nse
            * np.power(10, 0.05 * level)
            / np.sqrt(np.sum(np.power(eh2008_nse, 2)) / len(eh2008_nse))
        )

    return eh2008_nse


def generate_key_percent(
    signal: np.ndarray,
    threshold_db: float,
    window_length: int,
    percent_to_track: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Generate key percent.
    Locates frames above some energy threshold or tracks a certain percentage
    of frames. To track a certain percentage of frames in order to get measure
    of rms, adaptively sets threshold after looking at histogram of whole recording

    Args:
        signal (ndarray): The signal to analyse.
        threshold_db (float): fixed energy threshold (dB).
        window_length (int): length of window in samples.
        percent_to_track (float, optional): Track a percentage of frames (default: {None}).

    Raises:
        ValueError: percent_to_track is set too high.

    Returns:
        (tuple): containig
        - key (ndarray): The key array of indices of samples used in rms calculation.
        - used_threshold_db (float): Root Mean Squared threshold.

            and the threshold used to get a more accurate rms calculation
    """
    window_length = int(window_length)
    signal = signal.flatten()
    if window_length != math.floor(
        window_length
    ):  # whoops on fractional indexing: 7-March 2002
        window_length = math.floor(window_length)
        logging.warning(f"Window length must be integer: now {window_length}")

    signal_length = len(signal)

    expected = threshold_db
    # new Dec 2003. Possibly track percentage of frames rather than fixed threshold
    if percent_to_track is not None:
        logging.info("tracking %s percentage of frames", percent_to_track)
    else:
        logging.info("tracking fixed threshold")

    # put floor into histogram distribution
    non_zero = np.power(10, (expected - 30) / 10)

    n_frames = -1
    total_frames = math.floor(signal_length / window_length)
    every_db = np.zeros(total_frames)

    for ix in np.arange(
        0, window_length * total_frames - 1, window_length
    ):  # pylint: disable=invalid-name
        n_frames += 1
        this_sum = np.sum(
            np.power(signal[ix : (ix + window_length)].astype("float"), 2)
        )
        every_db[n_frames] = 10 * np.log10(non_zero + this_sum / window_length)
    n_frames += 1

    # from now on save only those analysed
    every_db = every_db[:n_frames]

    # Bec 2003, was 100 to give about a 0.5 dB quantising of levels
    n_bins, levels = np.histogram(every_db, 140)
    if percent_to_track is not None:
        # min number of bins to use
        inactive_bins = (100 - percent_to_track) * n_frames / 100
        n_levels = len(levels)
        inactive_ix = 0
        ix_count = 0
        for ix in np.arange(0, n_levels, 1):  # pylint: disable=invalid-name
            inactive_ix = inactive_ix + n_bins[ix]
            if inactive_ix > inactive_bins:
                break
            ix_count += 1
        if ix == 1:
            logging.warning("Counted every bin.........")
        elif ix == n_levels:
            raise ValueError("Generate_key_percent: no levels to count")
        expected = levels[max(1, ix_count)]

    # set new threshold conservatively to include more bins than desired
    used_threshold_db = expected

    # histogram should produce a two-peaked curve: thresh should be set in valley
    # between the two peaks, and set threshold a bit above that,
    # as it heads for main peak
    # FixMe : Could Otsu's method (from image processing) be used here? https://en.wikipedia.org/wiki/Otsu's_method
    frame_index = np.nonzero(every_db >= expected)[0]
    valid_frames = len(frame_index)
    key = np.zeros((1, valid_frames * window_length))[0]

    # convert frame numbers into indices for sig
    for ix in np.arange(valid_frames):  # pylint: disable=invalid-name
        meas_span = np.arange(
            (frame_index[ix] * window_length), (frame_index[ix] + 1) * window_length
        )
        key_span = np.arange(((ix) * window_length), (ix + 1) * window_length, 1)
        key[key_span] = meas_span
        key = key.flatten()

    return key, used_threshold_db


def measure_rms(
    signal: np.ndarray,
    sample_frequency: float,
    db_rel_rms: float,
    percent_to_track: Optional[float] = None,
) -> tuple:
    """Measure Root Mean Square.

    A sophisticated method of measuring RMS in a file. It splits the signal up into
    short windows, performs  a histogram of levels, calculates an approximate RMS,
    and then uses that RMS to calculate a threshold level in the histogram and then
    re-measures the RMS only using those durations whose individual RMS exceed that
    threshold.

    Args:
        signal (ndarray): the signal of which to measure the Root Mean Square.
        sample_frequency (float): sampling frequency.
        db_rel_rms (float): threshold for frames to track.
        percent_to_track (float, optional): track percentage of frames,
            rather than threshold (default: {None})
    Returns:
        (tuple): tuple containing
        - rms (float): overall calculated rms (linear)
        - key (ndarray): "key" array of indices of samples used in rms calculation
        - rel_db_thresh (float): fixed threshold value of -12 dB
        - active (float): proportion of values used in rms calculation
    """
    sample_frequency = int(sample_frequency)
    # first RMS is of all signal.
    first_stage_rms = np.sqrt(np.sum(np.power(signal, 2) / len(signal)))
    # use this RMS to generate key threshold to get more accurate RMS
    key_thr_db = max(20 * np.log10(first_stage_rms) + db_rel_rms, -80)

    # move key_thr_db to account for noise less peakier than signal
    key, used_thr_db = generate_key_percent(
        signal,
        key_thr_db,
        round(WIN_SECS * sample_frequency),
        percent_to_track=percent_to_track,
    )

    idx = key.astype(int)  # move into generate_key_percent
    # statistic to be reported later, BUT save for later
    # (for independent==1 loop where it sets a target for rms measure)
    active = 100 * len(key) / len(signal)
    rms = np.sqrt(np.sum(np.power(signal[idx], 2)) / len(key))
    rel_db_thresh = used_thr_db - 20 * np.log10(rms)

    return rms, idx, rel_db_thresh, active


def pad(signal, length):
    """Zero pad signal to required length.

    Assumes required length is not less than input length.
    """
    assert length >= signal.shape[0]
    return np.pad(
        signal, [(0, length - signal.shape[0])] + [(0, 0)] * (len(signal.shape) - 1)
    )


def read_signal(
    filename: str,
    offset: int = 0,
    nsamples: int = -1,
    nchannels: int = 0,
    offset_is_samples: bool = False,
) -> np.ndarray:
    """Read a wavefile and return as numpy array of floats.

    Args:
        filename (string): Name of file to read
        offset (int, optional): Offset in samples or seconds (from start). Defaults to 0.
        nsamples (int): Number of samples.
        nchannels (int): expected number of channel (default: 0 = any number OK)
        offset_is_samples (bool): measurement units for offset (default: False)

    Returns:
        np.ndarray: audio signal
    """
    try:
        wave_file = SoundFile(filename)
    except Exception as e:
        # Ensure incorrect error (24 bit) is not generated
        raise Exception(f"Unable to read {filename}.") from e

    if nchannels not in (0, wave_file.channels):
        raise Exception(
            f"Wav file ({filename}) was expected to have {nchannels} channels."
        )

    if not offset_is_samples:  # Default behaviour
        offset = int(offset * wave_file.samplerate)

    if offset != 0:
        wave_file.seek(offset)

    signal = wave_file.read(frames=nsamples)

    if wave_file.samplerate != MSBG_FS:
        signal = scipy.signal.resample(
            signal, int(MSBG_FS * signal.shape[0] / wave_file.samplerate)
        )

    return signal


def write_signal(
    filename: str,
    signal: np.ndarray,
    sample_frequency: float,
    floating_point: bool = True,
) -> None:
    """Write a signal as fixed or floating point wav file.

    Args:
        filename (str): name of file in to write to.
        signal (ndarray): signal to write.
        sample_frequency (float): sampling frequency.
        floating_point (bool): write as floating point else an ints (default: True).
    """

    if sample_frequency != MSBG_FS:
        logging.warning(
            f"Sampling rate mismatch: {filename} with sample frequency = {sample_frequency}."
        )
        # raise ValueError("Sampling rate mismatch")

    if floating_point is False:
        if TEST_NBITS == 16:
            subtype = "PCM_16"
            # If signal is float and we want int16
            signal *= 32768
            signal = signal.astype(np.dtype("int16"))
            assert np.max(signal) <= 32767 and np.min(signal) >= -32768
        elif TEST_NBITS == 24:
            subtype = "PCM_24"
    else:
        subtype = "FLOAT"

    soundfile.write(filename, signal, sample_frequency, subtype=subtype)
