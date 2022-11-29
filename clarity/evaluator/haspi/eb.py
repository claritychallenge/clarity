import numpy as np
from numba import jit
from scipy.signal import (
    butter,
    cheby2,
    convolve,
    correlate,
    firwin,
    group_delay,
    lfilter,
    resample_poly,
)

from clarity.enhancer.nalr import NALR


def EarModel(x, xsamp, y, ysamp, HL, itype, Level1):
    """
    Function to implement a cochlear model that includes the middle ear,
    auditory filter bank, OHC dynamic-range compression, and IHC attenuation.
    The inputs are the reference and processed signals that are to be
    compared. The reference x is at the reference intensity (e.g. 65 dB SPL
    or with NAL-R amplification) and has no other processing. The processed
    signal y is the hearing-aid output, and is assumed to have the same or
    greater group delay compared to the reference. The function outputs are
    the envelopes of the signals after OHC compression and IHC loss
    attenuation.

    Args:
    x        reference signal: should be adjusted to 65 dB SPL (itype=0 or 1)
               or to 65 dB SPL plus NAL-R gain (itype=2)
    xsamp    sampling rate for the reference signal, Hz
    y        processed signal (e.g. hearing-aid output) includes HA gain
    ysamp    sampling rate for the processed signal, Hz
    HL       audiogram giving the hearing loss in dB at six audiometric
               frequencies: [250, 500, 1000, 2000, 4000, 6000] Hz
    itype    purpose for the calculation:
             0=intelligibility: reference is nornal hearing and must not
               include NAL-R EQ
             1=quality: reference does not include NAL-R EQ
             2=quality: reference already has NAL-R EQ applied
    Level1   level calibration: signal RMS=1 corresponds to Level1 dB SPL

    Returns:
    xdB      envelope for the reference in each band
    xBM      BM motion for the reference in each band
    ydB      envelope for the processed signal in each band
    yBM      BM motion for the processed signal in each band
    xSL      compressed RMS average reference in each band converted to dB SL
    ySL      compressed RMS average output in each band converted to dB SL
    fsamp    sampling rate in Hz for the model outputs

    James M. Kates, 27 October 2011.
    BM motion added 30 Dec 2011.
    Revised 19 June 2012.
    Remove match of reference RMS level to processed 29 August 2012.
    IHC adaptation added 1 October 2012.
    BM envelope coverted to dB SL, 2 Oct 2012.
    Filterbank group delay corrected, 14 Dec 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    Updated by Gerardo Roa Dabike, September 2022.
    """

    # Processing parameters
    # OHC and IHC parameters for the hearing loss
    # Auditory filter center frequencies span 80 to 8000 Hz.
    nchan = 32  # Use 32 auditory frequency bands
    mdelay = 1  # Compensate for the gammatone group delay
    cfreq = CenterFreq(nchan)  # Center frequencies on an ERB scale

    # Cochlear model parameters for the processed signal
    attnOHCy, BWminy, lowkneey, CRy, attnIHCy = LossParameters(HL, cfreq)

    # The cochlear model parameters for the reference are the same as for the hearing
    # loss if calculating quality, but are for normal hearing if calculating intelligibility.
    if itype == 0:
        HLx = [0] * len(HL)
    else:
        HLx = HL
    [attnOHCx, BWminx, lowkneex, CRx, attnIHCx] = LossParameters(HLx, cfreq)

    # Parameters for the control filter bank
    HLmax = [100, 100, 100, 100, 100, 100]
    shift = 0.02  # Basal shift of 0.02 of the basilar membrane length
    cfreq1 = CenterFreq(nchan, shift)  # Center frequencies for the control
    _, BW1, _, _, _ = LossParameters(HLmax, cfreq1)
    # Maximum BW for the control

    # Input signal adjustments
    # Convert the signals to 24 kHz sampling rate.
    # Using 24 kHz guarantees that all of the cochlear filters have the same shape
    # independent of the incoming signal sampling rates
    x24, _ = Resamp24kHz(x, xsamp)
    y24, fsamp = Resamp24kHz(y, ysamp)

    # Check file sizes
    nxy = min(len(x24), len(y24))
    x24 = x24[:nxy]
    y24 = y24[:nxy]

    # Bulk broadband signal alignment
    x24, y24 = InputAlign(x24, y24)
    nsamp = len(x24)

    # For HASQI, here add NAL-R equalization if the quality reference doesn't already have it.
    if itype == 1:
        nfir = 140  # Length in samples of the FIR NAL-R EQ filter (24-kHz rate)
        enhancer = NALR(nfir, fsamp)
        aud = [250, 500, 1000, 2000, 4000, 6000]
        nalr_fir, _ = enhancer.build(HL, aud)
        x24 = convolve(x24, nalr_fir)  # Apply the NAL-R filter
        x24 = x24[nfir : nfir + nsamp]

    # Cochlear model
    # Middle ear
    xmid = MiddleEar(x24, fsamp)
    ymid = MiddleEar(y24, fsamp)

    # Initialize storage
    # Reference and processed envelopes and BM motion
    xdB = np.zeros((nchan, nsamp))
    ydB = np.zeros((nchan, nsamp))

    # Reference and processed average spectral values
    xave = np.zeros(nchan)
    yave = np.zeros(nchan)  # Processed
    xcave = np.zeros(nchan)  # Reference control
    ycave = np.zeros(nchan)  # Processed control

    # Filter bandwidths adjusted for intensity
    BWx = np.zeros(nchan)
    BWy = np.zeros(nchan)

    xb = np.zeros((nchan, nsamp))
    yb = np.zeros((nchan, nsamp))

    # Loop over each filter in the auditory filter bank
    for n in range(nchan):
        # Control signal envelopes for the reference and processed signals
        xcontrol, _, ycontrol, _ = GammatoneBM(
            xmid, BW1[n], ymid, BW1[n], fsamp, cfreq1[n]
        )

        # Adjust the auditory filter bandwidths for the average signal level
        BWx[n] = BWadjust(xcontrol, BWminx[n], BW1[n], Level1)  # Reference
        BWy[n] = BWadjust(ycontrol, BWminy[n], BW1[n], Level1)  # Processed

        # Envelopes and BM motion of the reference and processed signals
        xenv, xbm, yenv, ybm = GammatoneBM(xmid, BWx[n], ymid, BWy[n], fsamp, cfreq[n])

        # RMS levels of the ref and output envelopes for linear metric
        xave[n] = np.sqrt(np.mean(xenv**2))
        yave[n] = np.sqrt(np.mean(yenv**2))
        xcave[n] = np.sqrt(np.mean(xcontrol**2))
        ycave[n] = np.sqrt(np.mean(ycontrol**2))

        # Cochlear compression for the signal envelopes and BM motion
        xc, xb[n] = EnvCompressBM(
            xenv, xbm, xcontrol, attnOHCx[n], lowkneex[n], CRx[n], fsamp, Level1
        )
        yc, yb[n] = EnvCompressBM(
            yenv, ybm, ycontrol, attnOHCy[n], lowkneey[n], CRy[n], fsamp, Level1
        )

        # Correct for the delay between the reference and output
        yc = EnvAlign(xc, yc)  # Align processed envelope to reference
        yb[n] = EnvAlign(xb[n], yb[n])  # Align processed BM motion to reference

        # Convert the compressed envelopes and BM vibration envelopes to dB SPL
        xc, xb[n] = EnvSL(xc, xb[n], attnIHCx[n], Level1)
        yc, yb[n] = EnvSL(yc, yb[n], attnIHCy[n], Level1)

        # Apply the IHC rapid and short-term adaptation
        delta = 2  # Amount of overshoot
        xdB[n], xb[n] = IHCadapt(xc, xb[n], delta, fsamp)
        ydB[n], yb[n] = IHCadapt(yc, yb[n], delta, fsamp)

    # Additive noise level to give the auditory threshold
    IHCthr = -10  # Additive noise level, dB re: auditory threshold
    xBM = BMaddnoise(xb, IHCthr, Level1)
    yBM = BMaddnoise(yb, IHCthr, Level1)

    # Correct for the gammatone filterbank interchannel group delay.
    if mdelay > 0:
        xdB = GroupDelayComp(xdB, BWx, cfreq, fsamp)
        ydB = GroupDelayComp(ydB, BWx, cfreq, fsamp)
        xBM = GroupDelayComp(xBM, BWx, cfreq, fsamp)
        yBM = GroupDelayComp(yBM, BWx, cfreq, fsamp)

    # Convert average gammatone outputs to dB SPL
    xSL = aveSL(xave, xcave, attnOHCx, lowkneex, CRx, attnIHCx, Level1)
    ySL = aveSL(yave, ycave, attnOHCy, lowkneey, CRy, attnIHCy, Level1)

    return xdB, xBM, ydB, yBM, xSL, ySL, fsamp


def CenterFreq(nchan, shift=None):
    """
    Function to compute the ERB frequency spacing for the gammatone
    filter bank. The equation comes from Malcolm Slaney (1993).

    Calling variables
    nchan		number of filters in the filter bank
    shift     optional frequency shift of the filter bank specified as a
              fractional shift in distance along the BM. A positive shift
              is an increase in frequency (basal shift), and negative is
              a decrease in frequency (apical shift). The total length of
              the BM is normalized to 1. The frequency-to-distance map is
              from D.D. Greenwood (1990), JASA 87, 2592-2605, Eq (1).

    James M. Kates, 25 January 2007.
    Frequency shift added 22 August 2008.
    Lower and upper frequencies fixed at 80 and 8000 Hz, 19 June 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    lowFreq = 80
    highFreq = 8000

    # Moore and Glasberg ERB values
    EarQ = 9.26449
    minBW = 24.7

    # In the Matlab code, the loop below never evaluates
    # (but the current code was trained with this bug)
    shift = None  # This is to keep consistency with MATLAB code
    if shift is not None:
        k = 1
        A = 165.4
        a = 2.1  # shift specified as a fraction of the total length
        # Locations of the low and high frequencies on the BM between 0 and 1
        xLow = (1 / a) * np.log10(k + (lowFreq / A))
        xHigh = (1 / a) * np.log10(k + (highFreq / A))
        # Shift the locations
        xLow = xLow * (1 + shift)
        xHigh = xHigh * (1 + shift)
        # Compute the new frequency range
        lowFreq = A * (10 ** (a * xLow) - k)
        highFreq = A * (10 ** (a * xHigh) - k)

    # All of the following expressions are derived in Apple TR #35,
    # "An Efficient Implementation of the Patterson-Holdsworth Cochlear
    # Filter Bank" by Malcolm Slaney.
    cf = -(EarQ * minBW) + np.exp(
        np.arange(1, nchan)
        * (-np.log(highFreq + EarQ * minBW) + np.log(lowFreq + EarQ * minBW))
        / (nchan - 1)
    ) * (highFreq + EarQ * minBW)
    cf = np.insert(cf, 0, highFreq)  # Last center frequency is set to highFreq
    cf = np.flip(cf)
    return cf


def LossParameters(HL, cfreq):
    """
    Function to apportion the hearing loss to the outer hair cells (OHC)
    and the inner hair cells (IHC) and to increase the bandwidth of the
    cochlear filters in proportion to the OHC fraction of the total loss.

    Calling variables:
    HL		hearing loss at the 6 audiometric frequencies
    cfreq		array containing the center frequencies of the gammatone filters
                arranged from low to high

    Returns:
    attnOHC	attenuation in dB for the OHC gammatone filters
    BW		OHC filter bandwidth expressed in terms of normal
    lowknee	Lower kneepoint for the low-level linear amplification
    CR		Ranges from 1.4:1 at 150 Hz to 3.5:1 at 8 kHz for normal
                hearing. Reduced in proportion to the OHC loss to 1:1.
    attnIHC	attenuation in dB for the input to the IHC synapse

    James M. Kates, 25 January 2007.
    Version for loss in dB and match of OHC loss to CR, 9 March 2007.
    Low-frequency extent changed to 80 Hz, 27 Oct 2011.
    Lower kneepoint set to 30 dB, 19 June 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Audiometric frequencies in Hz
    aud = [250, 500, 1000, 2000, 4000, 6000]

    # Interpolation to give the loss at the gammatone center frequencies
    # Use linear interpolation in dB. The interpolation assumes that
    # cfreq[1] < aud[1] and cfreq[nfilt] > aud[6]
    nfilt = len(cfreq)
    fv = np.insert(aud, [0, len(aud)], [cfreq[0], cfreq[-1]])

    # Interpolated gain in dB
    loss = np.interp(cfreq, fv, np.insert(HL, [0, len(HL)], [HL[0], HL[-1]]))
    loss = np.maximum(loss, 0)
    # Make sure there are no negative losses

    # Compression ratio changes linearly with ERB rate from 1.25:1 in the 80-Hz
    # frequency band to 3.5:1 in the 8-kHz frequency band
    CR = 1.25 + 2.25 * np.arange(nfilt) / (nfilt - 1)

    # Maximum OHC sensitivity loss depends on the compression ratio.
    # The compression I/O curves assume linear below 30 and above 100 dB SPL in normal ears.
    maxOHC = 70 * (1 - (1 / CR))  # HC loss that results in 1:1 compression
    thrOHC = 1.25 * maxOHC  # Loss threshold for adjusting the OHC parameters

    # Apportion the loss in dB to the outer and inner hair cells based on the data of
    # Moore et al (1999), JASA 106, 2761-2778.

    # Reduce the CR towards 1:1 in proportion to the OHC loss.
    attnOHC = 0.8 * np.copy(loss)
    attnIHC = 0.2 * np.copy(loss)

    attnOHC[loss >= thrOHC] = 0.8 * thrOHC[loss >= thrOHC]
    attnIHC[loss >= thrOHC] = 0.2 * thrOHC[loss >= thrOHC] + (
        loss[loss >= thrOHC] - thrOHC[loss >= thrOHC]
    )

    # Adjust the OHC bandwidth in proportion to the OHC loss
    BW = np.ones(nfilt)
    BW = BW + (attnOHC / 50.0) + 2.0 * (attnOHC / 50.0) ** 6

    # Compute the compression lower kneepoint and compression ratio
    lowknee = attnOHC + 30
    upamp = 30 + 70 / CR  # Output level for an input of 100 dB SPL

    CR = (100 - lowknee) / (upamp + attnOHC - lowknee)  # OHC loss Compression ratio

    return attnOHC, BW, lowknee, CR, attnIHC


def Resamp24kHz(x, fsampx):
    """
    Function to resample the input signal at 24 kHz. The input sampling rate
    is rounded to the nearest kHz to comput the sampling rate conversion
    ratio.

    Calling variables:
    x         input signal
    fsampx    sampling rate for the input in Hz

    Returned argument:
    y         signal resampled at 24 kHz
    fsamp     output sampling rate in Kz

    James M. Kates, 20 June 2011.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Sampling rate information
    fsamp = 24000
    fy = round(fsamp / 1000)  # output rate to nearest kHz
    fx = round(fsampx / 1000)

    # Resample the signal
    if fx == fy:
        # No resampling performed if the rates match
        y = x
    elif fx < fy:
        # Resample for the input rate lower than the output
        y = resample_poly(x, fy, fx)

        # Match the RMS level of the resampled signal to that of the input
        xRMS = np.sqrt(np.mean(x**2))
        yRMS = np.sqrt(np.mean(y**2))
        y = (xRMS / yRMS) * y

    else:
        # Resample for the input rate higher than the output
        y = resample_poly(x, fy, fx)

        # Reduce the input signal bandwidth to 21 kHz (-10.5 to +10.5 kHz)
        # The power equalization is designed to match the signal intensities
        # over the frequency range spanned by the gammatone filter bank.
        # Chebyshev Type 2 LP
        order = 7
        atten = 30  # sidelobe attenuation in dB
        fcutx = 21 / fx
        bx, ax = cheby2(order, atten, fcutx)
        xfilt = lfilter(bx, ax, x, axis=0)

        # Reduce the resampled signal bandwisth to 21 kHz (-10.5 to +10.5 kHz)
        fcuty = 21 / fy
        by, ay = cheby2(order, atten, fcuty)
        yfilt = lfilter(by, ay, y, axis=0)

        # Compute the input and output RMS levels within the 21 kHz bandwidth and
        # match the output to the input
        xRMS = np.sqrt(np.mean(xfilt**2))
        yRMS = np.sqrt(np.mean(yfilt**2))
        y = (xRMS / yRMS) * y

    return y, fsamp


def InputAlign(x, y):
    """
    Function to provide approximate temporal alignment of the reference and
    processed output signals. Leading and trailing zeros are then pruned.
    The function assumes that the two sequences have the same sampling rate:
    call eb_Resamp24kHz for each sequence first, then call this function to
    align the signals.

    Calling variables:
    x       input reference sequence
    y       hearing-aid output sequence

    Returns:
    xp   pruned and shifted reference
    yp   pruned and shifted hearing-aid output

    James M. Kates, 12 July 2011.
    Match the length of the processed output to the reference for the
    purposes of computing the cross-covariance
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Match the length of the processed output to the reference for the purposes
    # of computing the cross-covariance
    nx = len(x)
    ny = len(y)
    nsamp = min(nx, ny)

    # Determine the delay of the output relative to the reference
    xy = correlate(
        x[:nsamp] - np.mean(x[:nsamp]), y[:nsamp] - np.mean(y[:nsamp]), "full"
    )  # Matlab code uses xcov thus the subtraction of mean
    index = np.argmax(np.abs(xy))
    delay = nsamp - index - 1

    # Back up 2 msec to allow for dispersion
    fsamp = 24000  # Cochlear model input sampling rate in Hz
    delay = round(delay - 2 * fsamp / 1000)  # Back up 2 ms

    # Align the output with the reference allowing for the dispersion
    if delay > 0:
        # Output delayed relative to the reference
        y = np.concatenate((y[delay:ny], np.zeros(delay)))
    else:
        # Output advanced relative to the reference
        y = np.concatenate((np.zeros(-delay), y[: ny + delay]))

    # Find the start and end of the noiseless reference sequence
    xabs = np.abs(x)
    xmax = np.max(xabs)
    xthr = 0.001 * xmax  # Zero detection threshold

    above_threshold = np.where(xabs > xthr)[0]
    nx0 = above_threshold[0]
    nx1 = above_threshold[-1]

    # Prune the sequences to remove the leading and trailing zeros
    nx1 = min(nx1, ny)
    xp = x[nx0 : nx1 + 1]
    yp = y[nx0 : nx1 + 1]

    return xp, yp


def MiddleEar(x, fsamp):
    """
    Function to design the middle ear filters and process the input
    through the cascade of filters. The middle ear model is a 2-pole HP
    filter at 350 Hz in series with a 1-pole LP filter at 5000 Hz. The
    result is a rough approximation to the equal-loudness contour at
    threshold.

    Calling variables:
        x		input signal
        fsamp	sampling rate in Hz

    Function output:
        xout	filtered output

    James M. Kates, 18 January 2007.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Design the 1-pole Butterworth LP using the bilinear transformation
    bLP, aLP = butter(1, 5000 / (0.5 * fsamp))

    # LP filter the input
    y = lfilter(bLP, aLP, x)

    # Design the 2-pole Butterworth HP using the bilinear transformation
    bHP, aHP = butter(2, 350 / (0.5 * fsamp), "high")

    # HP fitler the signal
    xout = lfilter(bHP, aHP, y)

    return xout


def GammatoneBM(x, BWx, y, BWy, fs, cf):
    """
    4th-order gammatone auditory filter. This implementation is based
    on the c program published on-line by Ning Ma, U. Sheffield, UK,
    that gives an implementation of the Martin Cooke (1991) filters:
    an impulse-invariant transformation of the gammatone filter. The
    signal is demodulated down to baseband using a complex exponential,
    and then passed through a cascade of four one-pole low-pass filters.

    This version filters two signals that have the same sampling rate and the
    same gammatone filter center frequencies. The lengths of the two signals
    should match; if they don't, the signals are truncated to the shorter of
    the two lengths.

    Calling variables:
    x			first sequence to be filtered
    BWx	    bandwidth for x relative to that of a normal ear
    y			second sequence to be filtered
    BWy	    bandwidth for x relative to that of a normal ear
    fs		sampling rate in Hz
    cf		filter center frequency in Hz

    Returns:
    envx      filter envelope output (modulated down to baseband) 1st signal
    BMx       BM motion for the first signal
    envy      filter envelope output (modulated down to baseband) 2nd signal
    BMy       BM motion for the second signal
    James M. Kates, 8 Jan 2007.
    Vectorized version for efficient MATLAB execution, 4 February 2007.
    Cosine and sine generation, 29 June 2011.
    Output sine and cosine sequences, 19 June 2012.
    Cosine/sine loop speed increased, 9 August 2013.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Filter ERB from Moore and Glasberg (1983)
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cf / earQ)

    # Check the lengths of the two signals
    nx = len(x)
    ny = len(y)
    nsamp = min(nx, ny)
    x = x[:nsamp]
    y = y[:nsamp]

    # Filter the first signal
    # Initialize the filter coefficients
    tpt = 2 * np.pi / fs
    tptBW = BWx * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    # Initialize the complex demodulation
    npts = len(x)
    sincf, coscf = GammatoneBW_demodulation(
        npts, tpt, cf, np.zeros(npts), np.zeros(npts)
    )

    # Filter the real and imaginary parts of the signal
    ureal = lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * coscf)
    uimag = lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], x * sincf)

    # Extract the BM velocity and the envelope
    BMx = gain * (ureal * coscf + uimag * sincf)
    envx = gain * np.sqrt(ureal * ureal + uimag * uimag)

    # Filter the second signal using the existing cosine and sine sequences
    tptBW = BWy * tpt * ERB * 1.019
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a
    gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)

    # Filter the real and imaginary parts of the signal
    ureal = lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * coscf)
    uimag = lfilter([1, a1, a5], [1, -a1, -a2, -a3, -a4], y * sincf)

    # Extract the BM velocity and the envelope
    BMy = gain * (ureal * coscf + uimag * sincf)
    envy = gain * np.sqrt(ureal * ureal + uimag * uimag)

    return envx, BMx, envy, BMy


@jit(nopython=True)
def GammatoneBW_demodulation(npts, tpt, cf, coscf, sincf):
    cn = np.cos(tpt * cf)
    sn = np.sin(tpt * cf)
    cold = 1
    sold = 0
    coscf[0] = cold
    sincf[0] = sold
    for n in range(1, npts):
        arg = cold * cn + sold * sn
        sold = sold * cn - cold * sn
        cold = arg
        coscf[n] = cold
        sincf[n] = sold

    return sincf, coscf


def BWadjust(control, BWmin, BWmax, Level1):
    """
    Function to compute the increase in auditory filter bandwidth in response
    to high signal levels.

    Args:
    control     envelope output in the control filter band
    BWmin       auditory filter bandwidth computed for the loss (or NH)
    BWmax       auditory filter bandwidth at maximum OHC damage
    Level1      RMS=1 corresponds to Level1 dB SPL

    Returned value:
    BW          filter bandwidth increased for high signal levels

    James M. Kates, 21 June 2011.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Compute the control signal level
    cRMS = np.sqrt(np.mean(control**2))
    cdB = 20 * np.log10(cRMS) + Level1

    # Adjust the auditory filter bandwidth
    if cdB < 50:
        # No BW adjustment for a signal below 50 dB SPL
        BW = BWmin
    elif cdB > 100:
        # Maximum BW if signal is above 100 dB SPL
        BW = BWmax
    else:
        # Linear interpolation between BW at 50 dB and max BW at 100 dB SPL
        BW = BWmin + ((cdB - 50) / 50) * (BWmax - BWmin)

    return BW


def EnvCompressBM(envsig, bm, control, attnOHC, thrLow, CR, fsamp, Level1):
    """
    Function to compute the cochlear compression in one auditory filter
    band. The gain is linear below the lower threshold, compressive with
    a compression ratio of CR:1 between the lower and upper thresholds,
    and reverts to linear above the upper threshold. The compressor
    assumes that auditory thresold is 0 dB SPL.

    Calling variables:
    envsig	analytic signal envelope (magnitude) returned by the
                gammatone filter bank
    bm        BM motion output by the filter bank
    control	analytic control envelope returned by the wide control
                path filter bank
    attnOHC	OHC attenuation at the input to the compressor
    thrLow	kneepoint for the low-level linear amplification
    CR		compression ratio
    fsamp		sampling rate in Hz
    Level1	dB reference level: a signal having an RMS value of 1 is
                assigned to Level1 dB SPL.

    Function outputs:
    y			compressed version of the signal envelope
    b         compressed version of the BM motion

    James M. Kates, 19 January 2007.
    LP filter added 15 Feb 2007 (Ref: Zhang et al., 2001)
    Version to compress the envelope, 20 Feb 2007.
    Change in the OHC I/O function, 9 March 2007.
    Two-tone suppression added 22 August 2008.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Initialize the compression parameters
    thrHigh = 100

    # Convert the control envelope to dB SPL
    small = 1e-30
    logenv = np.maximum(control, small)
    logenv = Level1 + 20 * np.log10(logenv)
    logenv = np.minimum(logenv, thrHigh)  # Clip signal levels above the upper threshold
    logenv = np.maximum(logenv, thrLow)  # Clip signal at the lower threshold

    # Compute the compression gain in dB
    gain = -attnOHC - (logenv - thrLow) * (1 - (1 / CR))

    # Convert the gain to linear and apply a LP filter to give a 0.2 ms delay
    gain = 10 ** (gain / 20)
    flp = 800
    b, a = butter(1, flp / (0.5 * fsamp))
    gain = lfilter(b, a, gain)

    # Apply the gain to the signals
    y = gain * envsig
    b = gain * bm

    return y, b


def EnvAlign(x, y):
    """
    Function to align the envelope of the processed signal to that of the
    reference signal.

    Args:
    x      envelope or BM motion of the reference signal
    y      envelope or BM motion of the output signal

    Returns:
    y      shifted output envelope to match the input

    James M. Kates, 28 October 2011.
    Absolute value of the cross-correlation peak removed, 22 June 2012.
    Cross-correlation range reduced, 13 August 2013.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # The MATLAB code limits the range of lags to search (to 100 ms) to save
    # computation time - no such option exists in numpy, but the code below limits the delay
    # to the same range as in Matlab, for consistent results
    fsamp = 24000
    corr_range = 100  # Range in msec for the correlation
    lags = round(0.001 * corr_range * fsamp)  # Range in samples
    npts = len(x)
    lags = min(lags, npts)

    xy = correlate(x, y, "full")
    location = np.argmax(xy[npts - lags : npts + lags])  # Limit the range in which
    delay = lags - location - 1

    # Time shift the output sequence
    if delay > 0:
        # Output delayed relative to the reference
        y = np.concatenate((y[delay:npts], np.zeros(delay)))
    elif delay < 0:
        # Output advanced relative to the reference
        y = np.concatenate((np.zeros(-delay), y[: npts + delay]))

    return y


def EnvSL(env, bm, attnIHC, Level1):
    """
    Function to convert the compressed envelope returned by
    cochlea_envcomp to dB SL.

    Args
    env			linear envelope after compression
    bm            linear basilar membrane vibration after compression
    attnIHC		IHC attenuation at the input to the synapse
    Level1		level in dB SPL corresponding to 1 RMS

    Return
    y				envelope in dB SL
    b             BM vibration with envelope converted to dB SL

    James M. Kates, 20 Feb 07.
    IHC attenuation added 9 March 2007.
    Basilar membrane vibration conversion added 2 October 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Convert the envelope to dB SL
    small = 1e-30
    y = Level1 - attnIHC + 20 * np.log10(env + small)
    y = np.maximum(y, 0)

    # Convert the linear BM motion to have a dB SL envelope
    gain = (y + small) / (env + small)
    b = gain * bm

    return y, b


@jit(nopython=True)
def IHCadapt(xdB, xBM, delta, fsamp):
    """
    Function to provide inner hair cell (IHC) adaptation. The adaptation is
    based on an equivalent RC circuit model, and the derivatives are mapped
    into 1st-order backward differences. Rapid and short-term adaptation are
    provided. The input is the signal envelope in dB SL, with IHC attenuation
    already applied to the envelope. The outputs are the envelope in dB SL
    with adaptation providing overshoot of the long-term output level, and
    the BM motion is multiplied by a gain vs. time function that reproduces
    the adaptation. IHC attenuation and additive noise for the equivalent
    auditory threshold are provided by a subsequent call to eb_BMatten.

    Calling variables:
    xdB      signal envelope in one frequency band in dB SL
             contains OHC compression and IHC attenuation
    xBM      basilar membrane vibration with OHC compression but no IHC atten
    delta    overshoot factor = delta x steady-state
    fsamp    sampling rate in Hz

    Returns:
    ydB      envelope in dB SL with IHC adaptation
    yBM      BM motion multiplied by the IHC adaptation gain function

    James M. Kates, 1 October 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Test the amount of overshoot
    dsmall = 1.0001
    delta = max(delta, dsmall)

    # Initialize adaptation time constants
    tau1 = 2  # Rapid adaptation in msec
    tau2 = 60  # Short-term adaptation in msec
    tau1 = 0.001 * tau1  # Convert to seconds
    tau2 = 0.001 * tau2

    # Equivalent circuit parameters
    T = 1 / fsamp
    R1 = 1 / delta
    R2 = 0.5 * (1 - R1)
    R3 = R2
    C1 = tau1 * (R1 + R2) / (R1 * R2)
    C2 = tau2 / ((R1 + R2) * R3)

    # Intermediate values used for the voltage update matrix inversion
    a11 = R1 + R2 + R1 * R2 * (C1 / T)
    a12 = -R1
    a21 = -R3
    a22 = R2 + R3 + R2 * R3 * (C2 / T)
    denom = 1 / (a11 * a22 - a21 * a12)

    # Additional intermediate values
    R1inv = 1 / R1
    R12C1 = R1 * R2 * (C1 / T)
    R23C2 = R2 * R3 * (C2 / T)

    # Initalize the outputs and state of the equivalent circuit
    nsamp = len(xdB)
    gain = np.ones_like(xdB)  # Gain vector to apply to the BM motion, default is 1
    ydB = np.zeros_like(xdB)
    V1 = 0
    V2 = 0
    small = 1e-30

    # Loop to process the envelope signal
    # The gain asymptote is 1 for an input envelope of 0 dB SPL
    for n in range(nsamp):
        V0 = xdB[n]
        b1 = V0 * R2 + R12C1 * V1
        b2 = R23C2 * V2
        V1 = denom * (a22 * b1 - a12 * b2)
        V2 = denom * (-a21 * b1 + a11 * b2)
        out = (V0 - V1) * R1inv
        ydB[n] = out

    ydB = np.maximum(ydB, 0)
    gain = (ydB + small) / (xdB + small)

    yBM = gain * xBM

    return ydB, yBM


def BMaddnoise(x, thr, Level1):
    """
    Function to apply the IHC attenuation to the BM motion and to add a
    low-level Gaussian noise to give the auditory threshold.

    Args:
    x         BM motion to be attenuated
    thr       additive noise level in dB re:auditory threshold
    Level1    an input having RMS=1 corresponds to Leve1 dB SPL

    Returns:
    y         attenuated signal with threhsold noise added

    James M. Kates, 19 June 2012.
    Just additive noise, 2 Oct 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    gn = 10 ** ((thr - Level1) / 20)  # Linear gain for the noise

    # rng = np.random.default_rng()
    noise = gn * np.random.standard_normal(x.shape)  # Gaussian RMS=1, then attenuated
    y = x + noise

    return y


def GroupDelayComp(xenv, BW, cfreq, fsamp):
    """
    Function to compensate for the group delay of the gammatone filter bank.
    The group delay is computed for each filter at its center frequency. The
    firing rate output of the IHC model is then adjusted so that all outputs
    have the same group delay.

    Calling variables:
        xenv (np.ndarray): matrix of signal envelopes or BM motion
        BW (): gammatone filter bandwidths adjusted for loss
        cfreq (): center frequencies of the bands
        fsamp (): sampling rate for the input signal in Hz (e.g. 24,000 Hz)

    Returns:
        yenv    envelopes or BM motion compensated for the group delay

    James M. Kates, 28 October 2011.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Processing parameters
    nchan = len(BW)

    # Filter ERB from Moore and Glasberg (1983)
    earQ = 9.26449
    minBW = 24.7
    ERB = minBW + (cfreq / earQ)

    # Initialize the gamatone filter coefficients
    tpt = 2 * np.pi / fsamp
    tptBW = tpt * 1.019 * BW * ERB
    a = np.exp(-tptBW)
    a1 = 4.0 * a
    a2 = -6.0 * a * a
    a3 = 4.0 * a * a * a
    a4 = -a * a * a * a
    a5 = 4.0 * a * a

    # Compute the group delay in samples at fsamp for each filter
    gd = np.zeros(nchan)
    for n in range(nchan):
        _, gd[n] = group_delay(
            ([1, a1[n], a5[n]], [1, -a1[n], -a2[n], -a3[n], -a4[n]]), 1
        )
    gd = np.round(gd).astype("int")  # convert to integer samples

    # Compute the delay correlation
    gmin = np.min(gd)
    gd = gd - gmin  # Remove the minimum delay from all the over values
    gmax = np.max(gd)
    correct = gmax - gd  # Samples delay needed to add to give alignment

    # Add delay correction to each frequency band
    yenv = np.zeros(xenv.shape)
    for n in range(nchan):
        r = xenv[n]
        npts = len(r)
        yenv[n] = np.concatenate((np.zeros(correct[n]), r[: npts - correct[n]]))

    return yenv


def aveSL(env, control, attnOHC, thrLow, CR, attnIHC, Level1):
    """
    Function to covert the RMS average output of the gammatone filter bank
    into dB SL. The gain is linear below the lower threshold, compressive
    with a compression ratio of CR:1 between the lower and upper thresholds,
    and reverts to linear above the upper threshold. The compressor
    assumes that auditory thresold is 0 dB SPL.

    Calling variables:
    env		analytic signal envelope (magnitude) returned by the
                gammatone filter bank, RMS average level
    control   control signal envelope
    attnOHC	OHC attenuation at the input to the compressor
    thrLow	kneepoint for the low-level linear amplification
    CR		compression ratio
    attnIHC	IHC attenuation at the input to the synapse
    Level1	dB reference level: a signal having an RMS value of 1 is
                assigned to Level1 dB SPL.

    Function output:
    xdB		compressed output in dB above the impaired threshold

    James M. Kates, 6 August 2007.
    Version for two-tone suppression, 29 August 2008.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Initialize the compression parameters
    thrHigh = 100  # Upper compression threshold

    # Convert the control to dB SPL
    small = 1e-30
    logenv = np.maximum(control, small)
    logenv = Level1 + 20 * np.log10(logenv)
    logenv = np.minimum(logenv, thrHigh)
    logenv = np.maximum(logenv, thrLow)

    # Compute compression gain in dB
    gain = -attnOHC - (logenv - thrLow) * (1 - (1 / CR))

    # Convert the signal envelope to dB SPL
    logenv = np.maximum(env, small)
    logenv = Level1 + 20 * np.log10(logenv)
    logenv = np.maximum(logenv, 0)
    xdB = logenv + gain - attnIHC
    xdB = np.maximum(xdB, 0)

    return xdB


def env_smooth(env, segsize, fsamp):
    """
    Function to smooth the envelope returned by the cochlear model. The
    envelope is divided into segments having a 50% overlap. Each segment is
    windowed, summed, and divided by the window sum to produce the average.
    A raised cosine window is used. The envelope sub-sampling frequency is
    2*(1000/segsize).

    Arguments:
        xenv: matrix of envelopes in each of the auditory bands
        segsize: averaging segment size in msec
        fsamp: input envelope sampling rate in Hz

    Returns:
        smooth: matrix of subsampled windowed averages in each band

    James M. Kates, 26 January 2007.
    Final half segment added 27 August 2012.

    Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Compute the window
    nwin = int(np.around(segsize * (0.001 * fsamp)))  # Segment size in samples
    test = nwin - 2 * np.floor(nwin / 2)  # 0=even, 1=odd
    if test > 0:
        # Force window length to be even
        nwin = nwin + 1
    window = np.hanning(nwin)  # Raised cosine von Hann window
    wsum = np.sum(window)  # Sum for normalization

    #  The first segment has a half window
    nhalf = int(nwin / 2)
    halfwindow = window[nhalf:nwin]
    halfsum = np.sum(halfwindow)

    # Number of segments and assign the matrix storage
    nchan = np.size(env, 0)
    npts = np.size(env, 1)
    nseg = int(1 + np.floor(npts / nwin) + np.floor((npts - nwin / 2) / nwin))
    smooth = np.zeros((nchan, nseg))

    #  Loop to compute the envelope in each frequency band
    for k in range(nchan):
        # Extract the envelope in the frequency band
        r = env[k, :]

        # The first (half) windowed segment
        nstart = 0
        smooth[k, 0] = np.sum(r[nstart:nhalf] * halfwindow.conj().transpose()) / halfsum

        # Loop over the remaining full segments, 50% overlap
        for n in range(1, nseg - 1):
            nstart = int(nstart + nhalf)
            nstop = int(nstart + nwin)
            smooth[k, n] = sum(r[nstart:nstop] * window.conj().transpose()) / wsum

        # The last (half) windowed segment
        nstart = nstart + nhalf
        nstop = nstart + nhalf
        smooth[k, nseg - 1] = (
            np.sum(r[nstart:nstop] * window[:nhalf].conj().transpose()) / halfsum
        )

    return smooth


def melcor(x, y, thr, addnoise):
    """
    Function to compute the cross-correlations between the input signal
    time-frequency envelope and the distortion time-frequency envelope. For
    each time interval, the log spectrum is fitted with a set of half-cosine
    basis functions. The spectrum weighted by the basis functions corresponds
    to mel cepstral coefficients computed in the frequency domain. The
    amplitude-normalized cross-covariance between the time-varying basis
    functions for the input and output signals is then computed.

    Arguments:
        x : subsampled input signal envelope in dB SL in each critical band
        y : subsampled distorted output signal envelope
        thr : threshold in dB SPL to include segment in calculation
        addnoise : additive Gaussian noise to ensure 0 cross-corr at low levels

    Returns:
        m1 : average cepstral correlation 2-6, input vs output
        xy : individual cepstral correlations, input vs output

    James M. Kates, 24 October 2006.
    Difference signal removed for cochlear model, 31 January 2007.
    Absolute value added 13 May 2011.
    Changed to loudness criterion for silence threhsold, 28 August 2012.

    Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Processing parameters
    nbands = x.shape[0]

    # Mel cepstrum basis functions (mel cepstrum because of auditory bands)
    nbasis = 6  # Number of cepstral coefficients to be used
    freq = np.arange(nbasis)
    k = np.arange(nbands)
    cepm = np.zeros((nbands, nbasis))
    for nb in range(nbasis):
        basis = np.cos(k * float(freq[nb]) * np.pi / float((nbands - 1)))
        cepm[:, nb] = basis / np.linalg.norm(basis)

    # Find the segments that lie sufficiently above the quiescent rate
    xLinear = 10 ** (x / 20)  # Convert envelope dB to linear (specific loudness)
    xsum = np.sum(xLinear, 0) / nbands  # Proportional to loudness in sones
    xsum = 20 * np.log10(xsum)  # Convert back to dB (loudness in phons)
    index = np.where(xsum > thr)[0]  # Identify those segments above threshold
    nsamp = index.shape[0]  # Number of segments above threshold

    # Exit if not enough segments above zero
    m1 = 0
    xy = 0
    if nsamp <= 1:
        print("Function eb.melcor: Signal below threshold, outputs set to 0.")
        return m1, xy

    # Remove the silent intervals
    x = x[:, index]
    y = y[:, index]

    # Add the low-level noise to the envelopes
    x = x + addnoise * np.random.standard_normal(x.shape)
    y = y + addnoise * np.random.standard_normal(y.shape)

    # Compute the mel cepstrum coefficients using only those segments
    # above threshold
    xcep = np.zeros((nbasis, nsamp))  # Input
    ycep = np.zeros((nbasis, nsamp))  # Output
    for n in range(nsamp):
        for k in range(nbasis):
            xcep[k, n] = np.sum(x[:, n] * cepm[:, k])
            ycep[k, n] = np.sum(y[:, n] * cepm[:, k])

    # Remove the average value from the cepstral coefficients. The
    # cross-correlation thus becomes a cross-covariance, and there
    # is no effect of the absolute signal level in dB.
    for k in range(nbasis):
        xcep[k, :] = xcep[k, :] - np.mean(xcep[k, :], axis=0)
        ycep[k, :] = ycep[k, :] - np.mean(ycep[k, :], axis=0)

    # Normalized cross-correlations between the time-varying cepstral coeff
    xy = np.zeros(nbasis)  # Input vs output
    small = 1.0e-30
    for k in range(nbasis):
        xsum = np.sum(xcep[k, :] ** 2)
        ysum = np.sum(ycep[k, :] ** 2)
        if (xsum < small) or (ysum < small):
            xy[k] = 0.0
        else:
            xy[k] = np.abs(np.sum(xcep[k, :] * ycep[k, :]) / np.sqrt(xsum * ysum))

    #
    # % Figure of merit is the average of the cepstral correlations, ignoring
    # % the first (average spectrum level).
    m1 = np.sum(xy[1:nbasis]) / (nbasis - 1)
    return m1, xy


def melcor9(x, y, thr, addnoise, segsize):
    """
    Function to compute the cross-correlations between the input signal
    time-frequency envelope and the distortion time-frequency envelope. For
    each time interval, the log spectrum is fitted with a set of half-cosine
    basis functions. The spectrum weighted by the basis functions corresponds
    to mel cepstral coefficients computed in the frequency domain. The
    amplitude-normalized cross-covariance between the time-varying basis
    functions for the input and output signals is then computed for each of
    the 8 modulation frequencies.

    Arguments:
        x : subsampled input signal envelope in dB SL in each critical band
        y : subsampled distorted output signal envelope
        thr : threshold in dB SPL to include segment in calculation
        addnoise : additive Gaussian noise to ensure 0 cross-corr at low levels
        segsize : segment size in ms used for the envelope LP filter (8 msec)

    Returns:
        CMave : average of the modulation correlations across analysis frequency
            bands and modulation frequency bands, basis functions 2 -6
        CMlow : average over the four lower mod freq bands, 0 - 20 Hz
        CMhigh : average over the four higher mod freq bands, 20 - 125 Hz
        CMmod : vector of cross-correlations by modulation frequency,
            averaged over ananlysis frequency band

    James M. Kates, 24 October 2006.
    Difference signal removed for cochlear model, 31 January 2007.
    Absolute value added 13 May 2011.
    Changed to loudness criterion for silence threshold, 28 August 2012.
    Version using envelope modulation filters, 15 July 2014.
    Modulation frequency vector output added 27 August 2014.

    Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Processing parameters
    nbands = x.shape[0]

    # Mel cepstrum basis functions (mel cepstrum because of auditory bands)
    nbasis = 6  # Number of cepstral coefficients to be used
    freq = np.arange(nbasis)
    k = np.arange(nbands)
    cepm = np.zeros((nbands, nbasis))
    for nb in range(nbasis):
        basis = np.cos(k * float(freq[nb]) * np.pi / float((nbands - 1)))
        cepm[:, nb] = basis / np.linalg.norm(basis)

    # Find the segments that lie sufficiently above the quiescent rate
    xLinear = 10 ** (x / 20)  # Convert envelope dB to linear (specific loudness)
    xsum = np.sum(xLinear, 0) / nbands  # Proportional to loudness in sones
    xsum = 20 * np.log10(xsum)  # Convert back to dB (loudness in phons)
    index = np.where(xsum > thr)[0]  # Identify those segments above threshold
    nsamp = index.shape[0]  # Number of segments above threshold

    # Modulation filter bands, segment size is 8 msec
    edge = [4.0, 8.0, 12.5, 20.0, 32.0, 50.0, 80.0]  # 8 bands covering 0 to 125 Hz
    nmod = 1 + len(edge)  # Number of modulation filter bands

    # Exit if not enough segments above zero
    CMave = 0
    CMlow = 0
    CMhigh = 0
    CMmod = np.zeros(nmod)
    if nsamp <= 1:
        print("Function eb.melcor9: Signal below threshold, outputs set to 0.")
        return CMave, CMlow, CMhigh, CMmod

    # Remove the silent intervals
    x = x[:, index]
    y = y[:, index]

    # Add the low-level noise to the envelopes
    x = x + addnoise * np.random.standard_normal(x.shape)
    y = y + addnoise * np.random.standard_normal(y.shape)

    # Compute the mel cepstrum coefficients using only those segments
    # above threshold
    xcep = np.zeros((nbasis, nsamp))  # Input
    ycep = np.zeros((nbasis, nsamp))  # Output
    for n in range(nsamp):
        for k in range(nbasis):
            xcep[k, n] = np.sum(x[:, n] * cepm[:, k])
            ycep[k, n] = np.sum(y[:, n] * cepm[:, k])

    # Remove the average value from the cepstral coefficients. The
    # cross-correlation thus becomes a cross-covariance, and there
    # is no effect of the absolute signal level in dB.
    for k in range(nbasis):
        xcep[k, :] = xcep[k, :] - np.mean(xcep[k, :], axis=0)
        ycep[k, :] = ycep[k, :] - np.mean(ycep[k, :], axis=0)

    # Envelope sampling parameters
    fsub = 1000.0 / (0.5 * segsize)  # Envelope sampling frequency in Hz
    fnyq = 0.5 * fsub  # Envelope Nyquist frequency

    # Design the linear-phase envelope modulation filters
    nfir = np.around(128 * (fnyq / 125))  # Adjust filter length to sampling rate
    nfir = int(2 * np.floor(nfir / 2))  # Force an even filter length
    b = np.zeros((nmod, nfir + 1))

    b[0, :] = firwin(
        nfir + 1, edge[0] / fnyq, window="hann", pass_zero="lowpass"
    )  # LP filter 0-4 Hz
    b[nmod - 1, :] = firwin(
        nfir + 1, edge[nmod - 2] / fnyq, window="hann", pass_zero="highpass"
    )  # HP 80-125 Hz
    for m in range(1, nmod - 1):
        b[m, :] = firwin(
            nfir + 1,
            [edge[m - 1] / fnyq, edge[m] / fnyq],
            window="hann",
            pass_zero="bandpass",
        )  # Bandpass filter

    CM = melcor9_crosscovmatrix(b, nmod, nbasis, nsamp, nfir, xcep, ycep)

    # Average over the  modulation filters and basis functions 2 - 6
    for m in range(nmod):
        for j in range(1, nbasis):
            CMave += CM[m, j]

    CMave = CMave / (nmod * (nbasis - 1))

    # Average over the four lower modulation filters
    for m in range(4):
        for j in range(1, nbasis):
            CMlow += CM[m, j]

    CMlow = CMlow / (4 * (nbasis - 1))

    #  Average over the four upper modulation filters
    for m in range(4, 8):
        for j in range(1, nbasis):
            CMhigh += CM[m, j]

    CMhigh = CMhigh / (4 * (nbasis - 1))

    # Average each modulation frequency over the basis functions
    for m in range(nmod):
        ave = 0
        for j in range(1, nbasis):
            ave += CM[m, j]

        CMmod[m] = ave / (nbasis - 1)

    return CMave, CMlow, CMhigh, CMmod


def melcor9_crosscovmatrix(b, nmod, nbasis, nsamp, nfir, xcep, ycep):
    """Compute the cross-covariance matrix."""
    small = 1.0e-30
    nfir2 = nfir / 2
    # Convolve the input and output envelopes with the modulation filters
    X = np.zeros((nmod, nbasis, nsamp))
    Y = np.zeros((nmod, nbasis, nsamp))
    for m in range(nmod):
        for j in range(nbasis):
            c = convolve(b[m], xcep[j, :], mode="full")
            X[m, j, :] = c[int(nfir2) : int(nfir2 + nsamp)]  # Remove the transients
            c = convolve(b[m], ycep[j, :], mode="full")
            Y[m, j, :] = c[int(nfir2) : int(nfir2 + nsamp)]  # Remove the transients

    # Compute the cross-covariance matrix
    CM = np.zeros((nmod, nbasis))
    for m in range(nmod):
        for j in range(nbasis):
            #  Index j gives the input reference band
            xj = X[m, j]  # Input freq band j, modulation freq m
            xj = xj - np.mean(xj)
            xsum = np.sum(xj**2)

            # Processed signal band
            yj = Y[m, j]  # Input freq band j, modulation freq m
            yj = yj - np.mean(yj)
            ysum = np.sum(yj**2)

            # Cross-correlate the reference and processed signals
            if (xsum < small) or (ysum < small):
                CM[m, j] = 0
            else:
                CM[m, j] = np.abs(np.sum(xj * yj)) / np.sqrt(xsum * ysum)
    return CM


def spect_diff(xSL, ySL):
    """
    Function to compute changes in the long-term spectrum and spectral slope.
    The metric is based on the spectral distortion metric of Moore and Tan
    (JAES, Vol 52, pp 900-914). The log envelopes in dB SL are converted to
    linear to approximate specific loudness. The outputs are the sum of the
    absolute differences, the standard deviation of the differences, and the
    maximum absolute difference. The same three outputs are provided for the
    normalized spectral difference and for the slope. The output is
    calibrated so that a processed signal having 0 amplitude produces a
    value of 1 for the spectrum difference.

    Abs diff: weight all deviations uniformly
    Std diff: weight larger deviations more than smaller deviations
    Max diff: only weight the largest deviation

    Arguments:
        xSL : reference signal spectrum in dB SL
        ySL : degraded signal spectrum in dB SL

    Returns:
        dloud (np.array) : [sum abs diff, std dev diff, max diff] spectra
        dnorm (np.array) : [sum abs diff, std dev diff, max diff] norm spectra
        dslope (np.array) : [sum abs diff, std dev diff, max diff] slope

    James M. Kates, 28 June 2012.

    Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Convert the dB SL to linear magnitude values. Because of the auditory
    # filter bank, the OHC compression, and auditory threshold, the linear
    # values are closely related to specific loudness.
    nbands = xSL.shape[0]
    x = 10 ** (xSL / 20)
    y = 10 ** (ySL / 20)

    # Normalize the level of the reference and degraded signals to have the
    # same loudness. Thus overall level is ignored while differences in
    # spectral shape are measured.
    xsum = np.sum(x)
    x /= xsum  # Loudness sum = 1 (arbitrary amplitude, proportional to sones)
    ysum = np.sum(y)
    y /= ysum

    # Compute the spectrum difference
    dloud = np.zeros(3)
    d = x - y  # Difference in specific loudness in each band
    dloud[0] = np.sum(np.abs(d))
    dloud[1] = nbands * np.std(d)  # Biased std: second moment
    dloud[2] = np.max(np.abs(d))

    # Compute the normalized spectrum difference
    dnorm = np.zeros(3)
    d = (x - y) / (x + y)  # Relative difference in specific loudness
    dnorm[0] = np.sum(np.abs(d))
    dnorm[1] = nbands * np.std(d)
    dnorm[2] = np.max(np.abs(d))

    # Compute the slope difference
    dslope = np.zeros(3)
    dx = x[1:nbands] - x[0 : nbands - 1]
    dy = y[1:nbands] - y[0 : nbands - 1]
    d = dx - dy  # Slope difference
    dslope[0] = np.sum(np.abs(d))
    dslope[1] = nbands * np.std(d)
    dslope[2] = np.max(np.abs(d))

    return dloud, dnorm, dslope


def bm_covary(xBM, yBM, segsize, fsamp):
    """
    Function to compute the cross-covariance (normalized cross-correlation)
    between the reference and processed signals in each auditory band. The
    signals are divided into segments having 50% overlap.

    Arguments:
        xBM : BM movement, reference signal
        yBM : BM movement, processed signal
        segsize : signal segment size, msec
        fsamp : sampling rate in Hz

    Returns:
        sigcov (np.array) : [nchan,nseg] of cross-covariance values
        sigMSx (np.array) : [nchan,nseg] of MS input signal energy values
        sigMSy (np.array) : [nchan,nseg] of MS processed signal energy values

    James M. Kates, 28 August 2012.
    Output amplitude adjustment added, 30 october 2012.

    Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Initialize parameters
    small = 1.0e-30

    # Lag for computing the cross-covariance
    lagsize = 1.0  # Lag (+/-) in msec
    maxlag = np.around(lagsize * (0.001 * fsamp))  # Lag in samples

    # Compute the segment window
    nwin = int(np.around(segsize * (0.001 * fsamp)))  # Segment size in samples
    test = nwin - 2 * np.floor(nwin / 2)  # 0=even, 1=odd
    if test > 0:
        nwin = nwin + 1  # Force window length to be even

    window = np.hanning(nwin).conj().transpose()  # Raised cosine von Hann window
    wincorr = correlate(window, window, "full")  # Window autocorrelation, inverted
    wincorr = wincorr[int(len(window) - 1 - maxlag) : int(maxlag + len(window))]
    wincorr = 1 / wincorr
    winsum2 = 1.0 / np.sum(window**2)  # Window power, inverted

    # The first segment has a half window
    nhalf = int(nwin / 2)
    halfwindow = window[nhalf:nwin]
    halfcorr = correlate(halfwindow, halfwindow, "full")
    halfcorr = halfcorr[
        int(len(halfwindow) - 1 - maxlag) : int(maxlag + len(halfwindow))
    ]
    halfcorr = 1 / halfcorr
    halfsum2 = 1.0 / np.sum(halfwindow**2)  # MS sum normalization, first segment

    # Number of segments
    nchan = xBM.shape[0]
    npts = xBM.shape[1]
    nseg = int(1 + np.floor(npts / nwin) + np.floor((npts - nwin / 2) / nwin))
    sigMSx = np.zeros((nchan, nseg))
    sigMSy = np.zeros((nchan, nseg))
    sigcov = np.zeros((nchan, nseg))

    # Loop to compute the signal mean-squared level in each band for each
    # segment and to compute the cross-corvariances.
    for k in range(nchan):
        # Extract the BM motion in the frequency band
        x = xBM[k, :]
        y = yBM[k, :]

        # The first (half) windowed segment
        nstart = 0
        segx = x[nstart:nhalf] * halfwindow  # Window the reference
        segy = y[nstart:nhalf] * halfwindow  # Window the processed signal
        segx = segx - np.mean(segx)  # Make 0-mean
        segy = segy - np.mean(segy)
        MSx = np.sum(segx**2) * halfsum2  # Normalize signal MS value by the window
        MSy = np.sum(segy**2) * halfsum2
        c = correlate(segx, segy, "full")
        c = c[int(len(segx) - 1 - maxlag) : int(maxlag + len(segx))]
        Mxy = np.max(np.abs(c * halfcorr))  # Unbiased cross-correlation
        if (MSx > small) and (MSy > small):
            sigcov[k, 0] = Mxy / np.sqrt(MSx * MSy)  # Normalized cross-covariance
        else:
            sigcov[k, 0] = 0.0

        sigMSx[k, 0] = MSx  # Save the reference MS level
        sigMSy[k, 0] = MSy

        # Loop over the remaining full segments, 50% overlap
        for n in range(1, nseg - 1):
            nstart = nstart + nhalf
            nstop = nstart + nwin
            segx = x[nstart:nstop] * window  # Window the reference
            segy = y[nstart:nstop] * window  # Window the processed signal
            segx = segx - np.mean(segx)  # Make 0-mean
            segy = segy - np.mean(segy)
            MSx = np.sum(segx**2) * winsum2  # Normalize signal MS value by the window
            MSy = np.sum(segy**2) * winsum2
            c = correlate(segx, segy, "full")
            c = c[int(len(segx) - 1 - maxlag) : int(maxlag + len(segx))]
            Mxy = np.max(np.abs(c * wincorr))  # Unbiased cross-corr
            if (MSx > small) and (MSy > small):
                sigcov[k, n] = Mxy / np.sqrt(MSx * MSy)  # Normalized cross-covariance
            else:
                sigcov[k, n] = 0.0

            sigMSx[k, n] = MSx  # Save the reference MS level
            sigMSy[k, n] = MSy  # Save the reference MS level

        # The last (half) windowed segment
        nstart = nstart + nhalf
        nstop = nstart + nhalf
        segx = x[nstart:nstop] * window[0:nhalf]  # Window the reference
        segy = y[nstart:nstop] * window[0:nhalf]  # Window the processed signal
        segx = segx - np.mean(segx)  # Make 0-mean
        segy = segy - np.mean(segy)
        MSx = np.sum(segx**2) * halfsum2  # Normalize signal MS value by the window
        MSy = np.sum(segy**2) * halfsum2

        c = np.correlate(segx, segy, "full")
        c = c[int(len(segx) - 1 - maxlag) : int(maxlag + len(segx))]

        Mxy = np.max(np.abs(c * halfcorr))  # Unbiased cross-correlation
        if (MSx > small) and (MSy > small):
            sigcov[k, nseg - 1] = Mxy / np.sqrt(
                MSx * MSy
            )  # Normalized cross-covariance
        else:
            sigcov[k, nseg - 1] = 0.0

        sigMSx[k, nseg - 1] = MSx  # Save the reference MS level
        sigMSy[k, nseg - 1] = MSy  # Save the reference MS level

    # Limit the cross-covariance to lie between 0 and 1
    sigcov = np.clip(sigcov, 0, 1)

    # Adjust the BM magnitude to correspond to the envelope in dB SL
    sigMSx = 2.0 * sigMSx
    sigMSy = 2.0 * sigMSy

    return sigcov, sigMSx, sigMSy


def ave_covary2(sigcov, sigMSx, thr):
    """
    Function to compute the average cross-covariance between the reference
    and processed signals in each auditory band. The silent time-frequency
    tiles are removed from consideration. The cross-covariance is computed
    for each segment in each frequency band. The values are weighted by 1
    for inclusion or 0 if the tile is below threshold. The sum of the
    covariance values across time and frequency are then divided by the total
    number of tiles above thresold. The calculation is a modification of
    Tan et al. (JAES 2004). The cross-covariance is also output with a
    frequency weighting that reflects the loss of IHC synchronization at high
    frequencies (Johnson, JASA 1980).

    Arguments:
        sigcov (np.array) : [nchan,nseg] of cross-covariance values
        sigMSx (np.array) : [nchan,nseg] of reference signal MS values
        thr : threshold in dB SL to include segment ave over freq in average

    Returns:
        avecov : cross-covariance in segments averaged over time and frequency
        syncov : cross-coraviance array, 6 different weightings for loss of
              IHC synchronization at high frequencies:
              LP Filter Order     Cutoff Freq, kHz
                1              1.5
                3              2.0
                5              2.5, 3.0, 3.5, 4.0

    James M. Kates, 28 August 2012.
    Adjusted for BM vibration in dB SL, 30 October 2012.
    Threshold for including time-freq tile modified, 30 January 2013.
    Version for different sync loss, 15 February 2013.

    Translated from MATLAB to Python by Gerardo Roa Dabike, September 2022.
    """

    # Array dimensions
    nchan = sigcov.shape[0]

    # Initialize the LP filter for loss of IHC synchronization
    cfreq = CenterFreq(nchan)  # Center frequencies in Hz on an ERB scale
    p = np.array([1, 3, 5, 5, 5, 5])  # LP filter order
    fcut = 1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])  # Cutoff frequencies in Hz
    fsync = np.zeros((6, nchan))  # Array of filter freq resp vs band center freq
    for n in range(6):
        fc2p = fcut[n] ** (2 * p[n])
        freq2p = cfreq ** (2 * p[n])
        fsync[n, :] = np.sqrt(fc2p / (fc2p + freq2p))

    # Find the segments that lie sufficiently above the threshold.
    sigRMS = np.sqrt(sigMSx)  # Convert squared amplitude to dB envelope
    sigLinear = 10 ** (sigRMS / 20)  # Linear amplitude (specific loudness)
    xsum = np.sum(sigLinear, 0) / nchan  # Intensity averaged over frequency bands
    xsum = 20 * np.log10(xsum)  # Convert back to dB (loudness in phons)
    index = np.argwhere(xsum > thr).T  # Identify those segments above threshold
    if index.size != 1:
        index = index.squeeze()
    nseg = index.shape[0]  # Number of segments above threshold

    # Exit if not enough segments above zero
    if nseg <= 1:
        print("Function eb.AveCovary: Ave signal below threshold, outputs set to 0.")
        avecov = 0
        # syncov = 0
        syncov = [0] * 6
        return avecov, syncov

    # Remove the silent segments
    sigcov = sigcov[:, index]
    sigRMS = sigRMS[:, index]

    # Compute the time-frequency weights. The weight=1 if a segment in a
    # frequency band is above threshold, and weight=0 if below threshold.
    weight = np.zeros((nchan, nseg))  # No IHC synchronization roll-off
    wsync1 = np.zeros((nchan, nseg))  # Loss of IHC synchronization at high frequencies
    wsync2 = np.zeros((nchan, nseg))
    wsync3 = np.zeros((nchan, nseg))
    wsync4 = np.zeros((nchan, nseg))
    wsync5 = np.zeros((nchan, nseg))
    wsync6 = np.zeros((nchan, nseg))
    for k in range(nchan):
        for n in range(nseg):
            if sigRMS[k, n] > thr:  # Thresh in dB SL for including time-freq tile
                weight[k, n] = 1
                wsync1[k, n] = fsync[0, k]
                wsync2[k, n] = fsync[1, k]
                wsync3[k, n] = fsync[2, k]
                wsync4[k, n] = fsync[3, k]
                wsync5[k, n] = fsync[4, k]
                wsync6[k, n] = fsync[5, k]

    # Sum the weighted covariance values
    csum = np.sum(np.sum(weight * sigcov))  # Sum of weighted time-freq tiles
    wsum = np.sum(np.sum(weight))  # Total number of tiles above thresold
    fsum = np.zeros(6)
    ssum = np.zeros(6)
    fsum[0] = np.sum(np.sum(wsync1 * sigcov))  # Sum of weighted time-freq tiles
    ssum[0] = np.sum(np.sum(wsync1))  # Total number of tiles above thresold
    fsum[1] = np.sum(np.sum(wsync2 * sigcov))  # Sum of weighted time-freq tiles
    ssum[1] = np.sum(np.sum(wsync2))  # Total number of tiles above thresold
    fsum[2] = np.sum(np.sum(wsync3 * sigcov))  # Sum of weighted time-freq tiles
    ssum[2] = np.sum(np.sum(wsync3))  # Total number of tiles above thresold
    fsum[3] = np.sum(np.sum(wsync4 * sigcov))  # Sum of weighted time-freq tiles
    ssum[3] = np.sum(np.sum(wsync4))  # Total number of tiles above thresold
    fsum[4] = np.sum(np.sum(wsync5 * sigcov))  # Sum of weighted time-freq tiles
    ssum[4] = np.sum(np.sum(wsync5))  # Total number of tiles above thresold
    fsum[5] = np.sum(np.sum(wsync6 * sigcov))  # Sum of weighted time-freq tiles
    ssum[5] = np.sum(np.sum(wsync6))  # Total number of tiles above thresold

    # Exit if not enough segments above zero
    if wsum < 1:
        avecov = 0
        print("Function eb.AveCovary: Signal tiles below threshold, outputs set to 0.")
    else:
        avecov = csum / wsum
    syncov = fsum / ssum

    return avecov, syncov
