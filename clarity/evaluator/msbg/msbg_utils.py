import json
import logging
import math
import os

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

HZ = np.array(
    [
        0.0,
        20.0,
        25.0,
        31.5,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
        750.0,
        800.0,
        1000.0,
        1250.0,
        1500.0,
        1600.0,
        2000.0,
        2500.0,
        3000.0,
        3150.0,
        4000.0,
        5000.0,
        6000.0,
        6300.0,
        8000.0,
        9000.0,
        10000.0,
        11200.0,
        12500.0,
        14000.0,
        15000.0,
        16000.0,
        20000.0,
        48000,
    ]
)

MIDEAR = np.array(
    [
        50.0,
        39.6,
        32.0,
        25.85,
        21.4,
        18.5,
        15.9,
        14.1,
        12.4,
        11.0,
        9.6,
        8.3,
        7.4,
        6.2,
        4.8,
        3.8,
        3.3,
        2.9,
        2.6,
        2.6,
        4.5,
        5.4,
        6.1,
        8.5,
        10.4,
        7.3,
        7.0,
        6.6,
        7.0,
        9.2,
        10.2,
        12.2,
        10.8,
        10.1,
        12.7,
        15.0,
        18.2,
        23.8,
        32.3,
        50.0,
        50.0,
    ]
)

# ------------------------------------------------------------------------------

# Free field (frontal)FF_ED correction for threshold (was ISO std Table 1 - 4.2 dB)
# i.e. relative to 0.0 dB at 1000 Hz, Shaw 1974

FF_ED = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.3,
        0.5,
        0.9,
        1.4,
        1.6,
        1.7,
        2.5,
        2.7,
        2.6,
        2.6,
        3.2,
        5.2,
        6.6,
        12.0,
        16.8,
        15.3,
        15.2,
        14.2,
        10.7,
        7.1,
        6.4,
        1.8,
        -0.9,
        -1.6,
        1.9,
        4.9,
        2.0,
        -2.0,
        2.5,
        2.5,
        2.5,
    ]
)

# ------------------------------------------------------------------------------

# DIFFUSE field ( relative to 0.0 dB at 1000Hz)
# from 2008 file [corrections08.m] used in Samsung collaboration

DF_ED = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.3,
        0.4,
        0.5,
        1.0,
        1.6,
        1.7,
        2.2,
        2.7,
        2.9,
        3.8,
        5.3,
        6.8,
        7.2,
        10.2,
        14.9,
        14.5,
        14.4,
        12.7,
        10.8,
        8.9,
        8.7,
        8.5,
        6.2,
        5.0,
        4.5,
        4.0,
        3.3,
        2.6,
        2.0,
        2.0,
        2.0,
    ]
)


# ------------------------------------------------------------------------------

# ITU Rec P 58 08/96 Head and Torso Simulator transfer fns. from Peter Hugher BTRL,
# 4-June-2001. Negative of values in Table 14a of ITU P58 (05/2013), accessible at
# http://www.itu.int/rec/T-REC-P.58-201305-I/en
# Freely available. Converts from ear reference point (ERP) to eardrum reference
# point (DRP). EXCEPT extra 2 points added for 20k & 48k by MAS, MAr 2012

ITU_HZ = np.array(
    [
        0,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        20000,
        48000,
    ]
)

# Ear Reference Point to Drum Reference Point (ERP-DRP) transfer function,
# Table 14A/P.58, sect 6.2. NB negative of table since defined other way round.

ITU_ERP_DRP = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3,
        0.2,
        0.5,
        0.6,
        0.7,
        1.1,
        1.7,
        2.6,
        4.2,
        6.5,
        9.4,
        10.3,
        6.6,
        3.2,
        3.3,
        16,
        14.4,
        14.4,
        14.4,
    ]
)


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


def read_gtf_file(gtf_file):
    """Read a gammatone filterbank file.

    List data is converted into numpy arrays.

    """

    # Fix filename if necessary
    dirname = os.path.dirname(os.path.abspath(__file__))
    gtf_file = os.path.join(dirname, gtf_file)
    with open(gtf_file, "r") as fp:
        data = json.load(fp)
    for key in data:
        if type(data[key]) == list:
            data[key] = np.array(data[key])
    return data


def firwin2(n, f, a, window=None, antisymmetric=None):
    """FIR filter design using the window method.

    Partial implementation of scipy firwin2 but using our own MATLAB-derived fir2.

    Args:
        n (int): The number of taps in the FIR filter.
        f (ndarray): The frequency sampling points. 0.0 to 1.0 with 1.0 being Nyquist.
        a (ndarray): The filter gains at the frequency sampling points.
        window (string or (string, float), optional): See scipy.firwin2 (default: (None))
        antisymmetric (bool, optional): Unused but present to main compatability with scipy firwin2.

    Returns:
        ndarray:  The filter coefficients of the FIR filter, as a 1-D array of length n.

    """
    window_shape = None
    if type(window) == tuple:
        window_type, window_param = window if window is not None else (None, 0)
    else:
        window_type = window

    order = n - 1

    if window_type == "kaiser":
        window_shape = scipy.signal.kaiser(n, window_param)

    if window_shape is None:
        b, _ = fir2(order, f, a)
    else:
        b, _ = fir2(order, f, a, window_shape)

    return b


def fir2(nn, ff, aa, npt=None):
    """FIR arbitrary shape filter design using the frequency sampling method.

    Translation of MATLAB fir2.

    Args:
        nn (int): Order
        ff (ndarray): Frequency breakpoints (0 < F < 1) where 1 is Nyquist rate.
                        First and last elements must be 0 and 1 respectively
        aa (ndarray): Magnitude breakpoints
        npt (int, optional): Number of points for freq response interpolation
            (default: max (smallest power of 2 greater than nn, 512))

    Returns:
        ndarray: nn + 1 filter coefficients

    """
    # Work with filter length instead of filter order
    nn += 1

    if npt is None:
        npt = 2.0 ** np.ceil(math.log(nn) / math.log(2)) if nn >= 1024 else 512
        wind = scipy.signal.hamming(nn)
    else:
        wind = npt
        npt = 2.0 ** np.ceil(math.log(nn) / math.log(2)) if nn >= 1024 else 512
    lap = np.fix(npt / 25)

    nbrk = max(len(ff), len(aa))

    ff[0] = 0
    ff[nbrk - 1] = 1

    H = np.zeros(npt + 1)
    nint = nbrk - 1
    df = np.diff(ff, n=1)

    npt += 1
    nb = 0
    H[0] = aa[0]

    for i in np.arange(nint):
        if df[i] == 0:
            nb = int(np.ceil(nb - lap / 2))
            ne = nb + lap - 1
        else:
            ne = int(np.fix(ff[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)
        inc = 0 if nb == ne else (j - nb) / (ne - nb)
        H[nb : (ne + 1)] = inc * aa[i + 1] + (1 - inc) * aa[i]
        nb = ne + 1

    dt = 0.5 * (nn - 1)
    rad = -dt * 1j * math.pi * np.arange(0, npt) / (npt - 1)
    H = H * np.exp(rad)

    H = np.concatenate((H, H[npt - 2 : 0 : -1].conj()))
    ht = np.real(np.fft.ifft(H))

    b = ht[0:nn] * wind

    return b, 1


def gen_tone(freq, duration, fs=44100, level=0):
    return (
        1.4142
        * np.power(10, (0.05 * level))
        * np.sin(2 * np.pi * freq * np.arange(1, duration * fs + 1) / fs)
    )


def gen_eh2008_speech_noise(duration, fs=44100, level=None, supplied_b=None):
    """Generate speech shaped noise.

    Start with white noise and re-shape to ideal SII, ie flat to 500 Hz, and
    sloping -9db/oct beyond that.

    Slightly different shape from SII stylised same as
    EarHEar 2008 paper, Moore et al.

    Args:
        duration (int): Duration of signal in seconds
        fs (int): Sampling rate
        level (float, optional): Normalise to level dB if present
        supplied_b (ndarray, optional): High-pass filter
            (default: uses built-in pre-emphasis filter)

    Returns:
        ndarray: Noise signal

    """
    duration = int(duration)
    fs = int(fs)
    n_samples = duration * fs

    # this rescales so that we get -7.5 dB/oct up to 8kHz, and -13 dB/oct above that
    norm_freq = GEN_NOISE_HZ / (fs / 2)
    last_f_idx = np.max(np.where(norm_freq < 1))
    norm_freq = np.append(norm_freq[0 : last_f_idx + 1], 1)

    # -9 dB/oct constant slope
    emph_nyq = EMPHASIS[last_f_idx] + 9 * np.log10(norm_freq[last_f_idx]) / np.log10(2)
    norm_emph = np.append(EMPHASIS[0 : last_f_idx + 1], emph_nyq)
    m = np.exp(np.log(10) * norm_emph / 20)

    # Create type II filter with 10 msec window and even number of taps
    n_taps = int(2 * np.ceil(10 * (fs / 2000))) + 1
    b = (
        supplied_b
        if supplied_b is not None
        else firwin2(n_taps, norm_freq, m, window="hamming", antisymmetric=False)
    )

    # white noise, 0 DC
    nburst = np.random.random((1, n_samples + len(b))) - 0.5

    # remove low-freq noise that may bias RMS estimate, -33dB at 50 Hz
    eh2008_nse1 = scipy.signal.lfilter(b, 1, nburst)

    # high-pass filter to remove low freqs (will be 2-pass with filtfilt)
    hpfB, hpfA = scipy.signal.ellip(3, 0.1, 50, 100 / (fs / 2), "high")
    padlen = 3 * (max(len(hpfA), len(hpfB)) - 1)
    eh2008_nse = scipy.signal.filtfilt(hpfB, hpfA, eh2008_nse1, padlen=padlen).flatten()

    # this introduces a delay so remove it, ie time-ADVANCE audio
    # compensating shift to time-align all filter outputs
    dly_shift = int(np.floor(len(b) / 2))
    valid_len = int(np.size(eh2008_nse) - dly_shift)  # _advance_ filter outputs
    # time advance
    eh2008_nse[0:valid_len] = eh2008_nse[dly_shift:]
    eh2008_nse = eh2008_nse[0:n_samples]

    if level is not None:
        eh2008_nse = (
            eh2008_nse
            * np.power(10, 0.05 * level)
            / np.sqrt(np.sum(np.power(eh2008_nse, 2)) / len(eh2008_nse))
        )

    return eh2008_nse


def generate_key_percent(sig, thr_dB, winlen, percent_to_track=None):
    """Generate key percent.
    Locates frames above some energy threshold or tracks a certain percentage
    of frames. To track a certain percentage of frames in order to get measure
    of rms, adaptively sets threshold after looking at histogram of whole recording
    Args:
        sig (ndarray): The signal to analyse
        thr_dB (float): fixed energy threshold (dB)
        winlen (int): length of window in samples
        percent_to_track (float, optional): Track a percentage of frames (default: {None})
    Raises:
        ValueError: percent_to_track is set too high
    Returns:
        (ndarray, float) -- "key" and rms threshold
            The key array of indices of samples used in rms calculation,
            and the threshold used to get a more accurate rms calculation
    """
    winlen = int(winlen)
    sig = sig.flatten()
    if winlen != math.floor(winlen):  # whoops on fractional indexing: 7-March 2002
        winlen = math.floor(winlen)
        logging.warning(f"Window length must be integer: now {winlen}")

    siglen = len(sig)

    expected = thr_dB
    # new Dec 2003. Possibly track percentage of frames rather than fixed threshold
    if percent_to_track is not None:
        logging.info(f"tracking {percent_to_track} percentage of frames")
    else:
        logging.info("tracking fixed threshold")

    # put floor into histogram distribution
    non_zero = np.power(10, (expected - 30) / 10)

    nframes = -1
    totframes = math.floor(siglen / winlen)
    every_dB = np.zeros(totframes)

    for ix in np.arange(0, winlen * totframes - 1, winlen):
        nframes += 1
        this_sum = np.sum(np.power(sig[ix : (ix + winlen)].astype("float"), 2))
        every_dB[nframes] = 10 * np.log10(non_zero + this_sum / winlen)
    nframes += 1

    # from now on save only those analysed
    every_dB = every_dB[:nframes]

    # Bec 2003, was 100 to give about a 0.5 dB quantising of levels
    n_bins, levels = np.histogram(every_dB, 140)
    if percent_to_track is not None:
        # min number of bins to use
        inactive_bins = (100 - percent_to_track) * nframes / 100
        n_levels = len(levels)
        inactive_ix = 0
        ix_count = 0
        for ix in np.arange(0, n_levels, 1):
            inactive_ix = inactive_ix + n_bins[ix]
            if inactive_ix > inactive_bins:
                break
            else:
                ix_count += 1
        if ix == 1:
            logging.warning("Counted every bin.........")
        elif ix == n_levels:
            raise ValueError("Generate_key_percent: no levels to count")
        expected = levels[max(1, ix_count)]

    # set new threshold conservatively to include more bins than desired
    used_thr_dB = expected

    # histogram should produce a two-peaked curve: thresh should be set in valley
    # between the two peaks, and set threshold a bit above that,
    # as it heads for main peak
    frame_index = np.nonzero(every_dB >= expected)[0]
    valid_frames = len(frame_index)
    key = np.zeros((1, valid_frames * winlen))[0]

    # convert frame numbers into indices for sig
    for ix in np.arange(valid_frames):
        meas_span = np.arange(
            (frame_index[ix] * winlen), (frame_index[ix] + 1) * winlen
        )
        key_span = np.arange(((ix) * winlen), (ix + 1) * winlen, 1)
        key[key_span] = meas_span
        key = key.flatten()

    return key, used_thr_dB


def measure_rms(signal, fs, dB_rel_rms, percent_to_track=None):
    """Measure rms.
    A sophisticated method of measuring RMS in a file. It splits the signal up into
    short windows, performs  a histogram of levels, calculates an approximate RMS,
    and then uses that RMS to calculate a threshold level in the histogram and then
    re-measures the RMS only using those durations whose individual RMS exceed that
    threshold.
    Args:
        signal (ndarray): the signal of which to measure the rms
        fs (float): sampling frequency
        dB_rel_rms (float): threshold for frames to track
        percent_to_track (float, optional): track percentage of frames,
            rather than threshold (default: {None})
    Returns:
        (tuple): tuple containing
        - rms (float): overall calculated rms (linear)
        - key (ndarray): "key" array of indices of samples used in rms calculation
        - rel_dB_thresh (float): fixed threshold value of -12 dB
        - active (float): proportion of values used in rms calculation
    """
    fs = int(fs)
    # first RMS is of all signal.
    first_stage_rms = np.sqrt(np.sum(np.power(signal, 2) / len(signal)))
    # use this RMS to generate key threshold to get more accurate RMS
    key_thr_dB = max(20 * np.log10(first_stage_rms) + dB_rel_rms, -80)

    # move key_thr_dB to account for noise less peakier than signal
    key, used_thr_dB = generate_key_percent(
        signal, key_thr_dB, round(WIN_SECS * fs), percent_to_track=percent_to_track
    )

    idx = key.astype(int)  # move into generate_key_percent
    # statistic to be reported later, BUT save for later
    # (for independent==1 loop where it sets a target for rms measure)
    active = 100 * len(key) / len(signal)
    rms = np.sqrt(np.sum(np.power(signal[idx], 2)) / len(key))
    rel_dB_thresh = used_thr_dB - 20 * np.log10(rms)

    return rms, idx, rel_dB_thresh, active


def pad(signal, length):
    """Zero pad signal to required length.

    Assumes required length is not less than input length.
    """
    assert length >= signal.shape[0]
    return np.pad(
        signal, [(0, length - signal.shape[0])] + [(0, 0)] * (len(signal.shape) - 1)
    )


def read_signal(filename, offset=0, nsamples=-1, nchannels=0, offset_is_samples=False):
    """Read a wavefile and return as numpy array of floats.
    Args:
        filename (string): Name of file to read
        offset (int, optional): Offset in samples or seconds (from start). Defaults to 0.
        nchannels: expected number of channel (default: 0 = any number OK)
        offset_is_samples (bool): measurement units for offset (default: False)
    Returns:
        ndarray: audio signal
    """
    try:
        wave_file = SoundFile(filename)
    except:  # noqa E722
        # Ensure incorrect error (24 bit) is not generated
        raise Exception(f"Unable to read {filename}.")

    if nchannels != 0 and wave_file.channels != nchannels:
        raise Exception(
            f"Wav file ({filename}) was expected to have {nchannels} channels."
        )

    if not offset_is_samples:  # Default behaviour
        offset = int(offset * wave_file.samplerate)

    if offset != 0:
        wave_file.seek(offset)

    x = wave_file.read(frames=nsamples)

    if wave_file.samplerate != MSBG_FS:
        x = scipy.signal.resample(x, int(MSBG_FS * x.shape[0] / wave_file.samplerate))

    return x


def write_signal(filename, x, fs, floating_point=True):
    """Write a signal as fixed or floating point wav file."""

    if fs != MSBG_FS:
        logging.warning(f"Sampling rate mismatch: {filename} with sr={fs}.")
        # raise ValueError("Sampling rate mismatch")

    if floating_point is False:
        if TEST_NBITS == 16:
            subtype = "PCM_16"
            # If signal is float and we want int16
            x *= 32768
            x = x.astype(np.dtype("int16"))
            assert np.max(x) <= 32767 and np.min(x) >= -32768
        elif TEST_NBITS == 24:
            subtype = "PCM_24"
    else:
        subtype = "FLOAT"

    soundfile.write(filename, x, fs, subtype=subtype)
