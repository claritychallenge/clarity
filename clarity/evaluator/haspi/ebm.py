from math import floor

import numpy as np
from scipy.signal import convolve, convolve2d


def EnvFilt(xdB, ydB, fcut, fsub, fsamp):
    """
    Function to lowpass filter and subsample the envelope in dB SL produced
    by the model of the auditory periphery. The LP filter uses a von Hann
    raised cosine window to ensure that there are no negative envelope values
    produced by the filtering operation.

    Args:
        xdB (np.ndarray): env in dB SL for the ref signal in each auditory band
        ydB (np.ndarray): env in dB SL for the degraded signal in each auditory band
        fcut ():  LP filter cutoff frequency for the filtered envelope, Hz
        fsub ():  subsampling frequency in Hz for the LP filtered envelopes
        fsamp ():  sampling rate in Hz for the signals xdB and ydB

    Returns:
        tuple: xLP - LP filtered and subsampled reference signal envelope
           Each frequency band is a separate column. yLP - LP filtered and subsampled
           degraded signal envelope


    James M. Kates, 12 September 2019.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Check the filter design parameters
    assert fsub <= fsamp, "Error in ebm.EnvFilt: Subsampling rate too high."
    assert fcut <= 0.5 * fsub, "Error in ebm.EnvFilt: LP cutoff frequency too high."

    # Check the data matrix orientation
    # Require each frequency band to be a separate column
    nrow = xdB.shape[0]  # number of rows
    ncol = xdB.shape[1]  # number of columnts
    if ncol > nrow:
        xdB = xdB.T
        ydB = ydB.T
    nsamp = xdB.shape[0]

    # Compute the lowpass filter length in samples to give -3 dB at fcut Hz
    tfilt = 1000 * (1 / fcut)  # filter length in ms
    tfilt = 0.7 * tfilt  # Empirical adjustment to the filter length
    nfilt = round(0.001 * tfilt * fsamp)  # Filter length in samples
    nhalf = floor(nfilt / 2)
    nfilt = int(2 * nhalf)  # Force an even filter length

    # Design the FIR LP filter using a von Hann window to ensure that there are no negative
    # envelope values. The MATLAB code uses the hanning() function, which returns the Hann
    # window without the first and last zero-weighted window samples,
    # unlike np.hann and scipy.signal.windows.hann; the code below replicates this behaviour
    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, nfilt / 2 + 1) / (nfilt + 1)))
    benv = np.concatenate((w, np.flip(w)))
    benv = benv / np.sum(benv)

    # LP filter for the envelopes at fsamp
    xenv = convolve2d(xdB, np.expand_dims(benv, 1), "full")  # 2D convolution
    xenv = xenv[nhalf : nhalf + nsamp, :]
    yenv = convolve2d(ydB, np.expand_dims(benv, 1), "full")
    yenv = yenv[nhalf : nhalf + nsamp, :]

    # Subsample the LP filtered envelopes
    space = floor(fsamp / fsub)
    index = np.arange(0, nsamp, space)
    xLP = xenv[index, :]
    yLP = yenv[index, :]

    return xLP, yLP


def CepCoef(xdB, ydB, thrCep, thrNerve, nbasis):
    """
    Function to compute the cepstral correlation coefficients between the
    reference signal and the distorted signal log envelopes. The silence
    portions of the signals are removed prior to the calculation based on the
    envelope of the reference signal. For each time sample, the log spectrum
    in dB SL is fitted with a set of half-cosine basis functions. The
    cepstral coefficients then form the input to the cepstral correlation
    calculation.

    Args:
        xdB ():      subsampled reference signal envelope in dB SL in each band
        ydB ():	    subsampled distorted output signal envelope
        thrCep ():   threshold in dB SPL to include sample in calculation
        thrNerve (): additive noise RMS for IHC firing (in dB)
        nbasis    number of cepstral basis functions to use

    Returns:
        tuple: xcep cepstral coefficient matrix for the ref signal (nsamp,nbasis) and ycep  cepstral coefficient matrix
               for the output signal (nsamp,nbasis) each column is a separate basis function, from low to high

    James M. Kates, 23 April 2015.
    Gammawarp version to fit the basis functions, 11 February 2019.
    Additive noise for IHC firing rates, 24 April 2019.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Processing parameters
    nbands = xdB.shape[1]

    # Mel cepstrum basis functions
    freq = np.arange(0, nbasis)
    k = np.arange(0, nbands)
    cepm = np.zeros((nbands, nbasis))

    for nb in range(nbasis):
        basis = np.cos(freq[nb] * np.pi * k / (nbands - 1))
        cepm[:, nb] = basis / np.sqrt(np.sum(basis**2))

    # Find the reference segments that lie sufficiently above the quiescent rate
    xLinear = 10 ** (xdB / 20)  # Convert envelope dB to linear (specific loudness)
    xsum = np.sum(xLinear, 1) / nbands  # Proportional to loudness in sones
    xsum = 20 * np.log10(xsum)  # Convert back to dB (loudness in phons)
    index = np.where(xsum > thrCep)[0]  # Identify those segments above threshold
    nsamp = len(index)  # Number of segments above threshold

    # Exit if not enough segments above zero
    assert nsamp > 1, "Function ebm_CepCoef: Signal below threshold"

    # Remove the silent samples
    xdB = xdB[index, :]
    ydB = ydB[index, :]

    # Add low-level noise to provide IHC firing jitter
    xdB = AddNoise(xdB, thrNerve)
    ydB = AddNoise(ydB, thrNerve)

    # Compute the mel cepstrum coefficients using only those samples above threshold
    xcep = xdB @ cepm
    ycep = ydB @ cepm

    # Remove the average value from the cepstral coefficients. The cepstral cross-correlation
    # will thus be a cross-covariance, and there is no effect of the absolute signal level in dB.
    for n in range(nbasis):
        x = xcep[:, n]
        x = x - np.mean(x)
        xcep[:, n] = x
        y = ycep[:, n]
        y = y - np.mean(y)
        ycep[:, n] = y

    return xcep, ycep


def AddNoise(ydB, thrdB):
    """
    Function to add independent random Gaussian noise to the subsampled
    signal envelope in each auditory frequency band.

    Args:
    ydB      subsampled envelope in dB re:auditory threshold
    thrdB    additive noise RMS level (in dB)
    Level1   an input having RMS=1 corresponds to Level1 dB SPL

    Returns:
    zdB      envelope with threshold noise added, in dB re:auditory threshold

    James M. Kates, 23 April 2019.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Additive noise sequence
    # rng = np.random.default_rng()
    noise = thrdB * np.random.standard_normal(
        ydB.shape
    )  # Gaussian noise with RMS=1, then scaled

    # Add the noise to the signal envelope
    zdB = ydB + noise
    return zdB


def ModFilt(Xenv, Yenv, fsub):
    """
    Function to apply an FIR modulation filterbank to the reference envelope
    signals contained in matrix Xenv and the processed signal envelope
    signals in matrix Yenv. Each column in Xenv and Yenv is a separate filter
    band or cepstral coefficient basis function. The modulation filters use a
    lowpass filter for the lowest modulation rate, and complex demodulation
    followed by a lowpass filter for the remaining bands. The onset and
    offset transients are removed from the FIR convolutions to temporally
    align the modulation filter outputs.

    Args:
        Xenv (np.ndarray) : matrix containing the subsampled reference envelope values. Each
             column is a different frequency band or cepstral basis function
             arranged from low to high.
        Yenv (np.ndarray): matrix containing the subsampled processed envelope values
        fsub (): envelope sub-sampling rate in Hz

    Returns:
        tuple: Xmod a cell array containing the reference signal output of the
             modulation filterbank. Xmod is of size {nchan,nmodfilt} where
             nchan is the number of frequency channels or cepstral basis
             functions in Xenv, and nmodfilt is the number of modulation
             filters used in the analysis. Each cell contains a column vector
             of length nsamp, where nsamp is the number of samples in each
             envelope sequence contained in the columns of Xenv.
             Ymod cell array containing the processed signal output of the
             modulation filterbank.
             cf vector of the modulation rate filter center frequencies

    James M. Kates, 14 February 2019.
    Two matrix version of gwarp_ModFiltWindow, 19 February 2019.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Input signal properties
    nsamp = Xenv.shape[0]
    nchan = Xenv.shape[1]

    # Modulation filter band cf and edges, 10 bands
    # Band spacing uses the factor of 1.6 used by Dau
    cf = np.array([2, 6, 10, 16, 25, 40, 64, 100, 160, 256])  # Band center frequencies
    nmod = len(cf)
    edge = np.zeros(nmod + 1)
    edge[0:3] = [0, 4, 8]
    for k in range(3, nmod + 1):
        edge[k] = (cf[k - 1] ** 2) / edge[k - 1]

    # Allowable filters based on envelope subsampling rate
    fNyq = 0.5 * fsub
    index = edge < fNyq
    edge = edge[index]  # Filter upper bands edges less than Nyquist rate
    nmod = len(edge) - 1
    cf = cf[:nmod]

    # Assign FIR filter lengths. Setting t0=0.2 gives a filter Q of about 1.25
    # to match Ewert et al. (2002), and t0=0.33 corresponds to Q=2 (Dau et al 1997a).
    # Moritz et al. (2015) used t0=0.29. General relation Q=6.25*t0, compromise with
    # t0=0.24 which gives Q=1.5

    t0 = 0.24  # Filter length in seconds for the lowest modulation frequency band
    t = np.zeros(nmod)
    t[0] = t0
    t[1] = t0
    t[2:nmod] = t0 * cf[2] / cf[2:nmod]  # Constant-Q filters above 10 Hz
    nfir = 2 * np.floor(t * fsub / 2)  # Force even filter lengths in samples
    nhalf = nfir / 2

    # Design the family of lowpass windows
    b = []  # Filter coefficients, one filter impulse response per list element
    for k in range(nmod):
        bk = np.hanning(nfir[k] + 1)
        bk = bk / np.sum(bk)
        b.append(bk)

    # Pre-compute the cosine and sine arrays
    co = []  # cosine array, one frequency per list element
    si = []  # sine array, one frequency per list element
    n = np.arange(1, nsamp + 1)
    for k in range(nmod):
        if k == 0:
            co.append(1)
            si.append(0)
        else:
            co.append(np.sqrt(2) * np.cos(np.pi * n * cf[k] / fNyq))
            si.append(np.sqrt(2) * np.sin(np.pi * n * cf[k] / fNyq))

    # Convolve the input and output envelopes with the modulation filters
    Xmod = np.zeros((nchan, nmod, nsamp))
    Ymod = np.zeros((nchan, nmod, nsamp))
    for k in range(nmod):  # Loop over the modulation filters
        bk = b[k]  # Extract the lowpass filter impulse response
        nh = int(nhalf[k])  # Transient duration for the filter
        c = co[k]  # Cosine and sine for complex modulation
        s = si[k]
        for m in range(nchan):  # Loop over the input signal vectors
            # Reference signal
            x = Xenv[:, m]  # Extract the frequency or cepstral coefficient band
            u = convolve(
                (x * c - 1j * x * s), bk
            )  # Complex demodulation, then LP filter
            u = u[nh : nh + nsamp]  # Truncate the filter transients
            xfilt = (
                np.real(u) * c - np.imag(u) * s
            )  # Modulate back up to the carrier freq
            Xmod[m, k, :] = xfilt  # Save the filtered signal

            # Processed signal
            y = Yenv[:, m]  # Extract the frequency or cepstral coefficient band
            v = convolve(
                (y * c - 1j * y * s), bk
            )  # Complex demodulation, then LP filter
            v = v[nh : nh + nsamp]  # Truncate the filter transients
            yfilt = (
                np.real(v) * c - np.imag(v) * s
            )  # Modulate back up to the carrier freq
            Ymod[m, k, :] = yfilt  # Save the filtered signal

    return Xmod, Ymod, cf


def ModCorr(Xmod, Ymod):
    """
    Function to compute the cross-correlations between the input signal
    time-frequency envelope and the distortion time-frequency envelope. The
    cepstral coefficients or envelopes in each frequency band have been
    passed through the modulation filterbank using function ebm_ModFilt.

    Args:
        Xmod (np.array): cell array containing the reference signal output of the
             modulation filterbank. Xmod is of size {nchan,nmodfilt} where
             nchan is the number of frequency channels or cepstral basis
             functions in Xenv, and nmodfilt is the number of modulation
             filters used in the analysis. Each cell contains a column vector
             of length nsamp, where nsamp is the number of samples in each
             envelope sequence contained in the columns of Xenv.
        Ymod subsampled distorted output signal envelope

    Output:
        float: aveCM modulation correlations averaged over basis functions 2-6
             vector of size nmodfilt

    James M. Kates, 21 February 2019.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Processing parameters
    nchan = Xmod.shape[0]  # Number of basis functions
    nmod = Xmod.shape[1]  # Number of modulation filters
    small = 1e-30  # Zero threshold

    # Compute the cross-covariance matrix
    CM = np.zeros((nchan, nmod))
    for m in range(nmod):
        for j in range(nchan):
            # Index j gives the input reference band
            xj = Xmod[j, m]  # Input freq band j, modulation freq m
            xj = xj - np.mean(xj)
            xsum = np.sum(xj**2)
            # Processed signal band
            yj = Ymod[j, m]  # Processed freq band j, modulation freq m
            yj = yj - np.mean(yj)
            ysum = np.sum(yj**2)
            # Cross-correlate the reference and processed signals
            if (xsum < small) or (ysum < small):
                CM[j, m] = 0
            else:
                CM[j, m] = np.abs(np.sum(xj * yj)) / np.sqrt(xsum * ysum)

    aveCM = np.mean(CM[1:6], 0)

    return aveCM
