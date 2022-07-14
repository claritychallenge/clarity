import numpy as np
from numba import jit
from scipy.signal import butter, cheby2, correlate, group_delay, lfilter, resample_poly


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

    Calling arguments:
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

    Returned values:
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
    """

    # Processing parameters
    # OHC and IHC parameters for the hearing loss
    # Auditory filter center frequencies span 80 to 8000 Hz.
    nchan = 32  # Use 32 auditory frequency bands
    mdelay = 1  # Compensate for the gammatone group delay
    cfreq = CenterFreq(nchan)  # Center frequencies on an ERB scale

    # Cochlear model parameters for the processed signal
    attnOHCy, BWminy, lowkneey, CRy, attnIHCy = LossParameters(HL, cfreq)

    # The cochlear model parameters for the reference are the same as for the hearing loss if calculating quality,
    # but are for normal hearing if calculating intelligibility.
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
    # Using 24 kHz guarantees that all of the cochlear filters have the same shape independent of the incoming signal sampling rates
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
    # if itype == 1:
    #     pass

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

    # In the Matlab code, the loop below never evaluates (but the current code was trained with this bug)
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

    # All of the following expressions are derived in Apple TR #35, "An Efficient Implementation of the Patterson-Holdsworth Cochlear
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

    Returned values:
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
    # Use linear interpolation in dB. The interpolation assumes that  cfreq[1] < aud[1] and cfreq[nfilt] > aud[6]
    nfilt = len(cfreq)
    fv = np.insert(aud, [0, len(aud)], [cfreq[0], cfreq[-1]])

    # Interpolated gain in dB
    loss = np.interp(cfreq, fv, np.insert(HL, [0, len(HL)], [HL[0], HL[-1]]))
    loss = np.maximum(loss, 0)
    # Make sure there are no negative losses

    # Compression ratio changes linearly with ERB rate from 1.25:1 in the 80-Hz frequency band to 3.5:1 in the 8-kHz frequency band
    CR = 1.25 + 2.25 * np.arange(nfilt) / (nfilt - 1)

    # Maximum OHC sensitivity loss depends on the compression ratio.
    # The compression I/O curves assume linear below 30 and above 100 dB SPL in normal ears.
    maxOHC = 70 * (1 - (1 / CR))  # HC loss that results in 1:1 compression
    thrOHC = 1.25 * maxOHC  # Loss threshold for adjusting the OHC parameters

    # Apportion the loss in dB to the outer and inner hair cells based on the data of Moore et al (1999), JASA 106, 2761-2778.
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

        # Compute the input and output RMS levels within the 21 kHz bandwidth and match the output to the input
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

    Returned values:
    xp   pruned and shifted reference
    yp   pruned and shifted hearing-aid output

    James M. Kates, 12 July 2011.
    Match the length of the processed output to the reference for the
    purposes of computing the cross-covariance
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # Match the length of the processed output to the reference for the purposes of computing the cross-covariance
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
    if nx1 > ny:
        nx1 = ny
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

    Returned values:
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

    Calling arguments:
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

    Calling arguments:
    x      envelope or BM motion of the reference signal
    y      envelope or BM motion of the output signal

    Returned values:
    y      shifted output envelope to match the input

    James M. Kates, 28 October 2011.
    Absolute value of the cross-correlation peak removed, 22 June 2012.
    Cross-correlation range reduced, 13 August 2013.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """

    # The MATLAB code limits the range of lags to search (to 100 ms) to save computation time - no such option exists in numpy,
    # but the code below limits the delay to the same range as in Matlab, for consistent results
    fsamp = 24000
    range = 100  # Range in msec for the correlation
    lags = round(0.001 * range * fsamp)  # Range in samples
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

    Calling arguments
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

    Returned values:
    ydB      envelope in dB SL with IHC adaptation
    yBM      BM motion multiplied by the IHC adaptation gain function

    James M. Kates, 1 October 2012.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Test the amount of overshoot
    dsmall = 1.0001
    if delta < dsmall:
        delta = dsmall

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

    Calling arguments:
    x         BM motion to be attenuated
    thr       additive noise level in dB re:auditory threshold
    Level1    an input having RMS=1 corresponds to Leve1 dB SPL

    Returned values:
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
    xenv     matrix of signal envelopes or BM motion
    BW       gammatone filter bandwidths adjusted for loss
    cfreq    center frequencies of the bands
    fsamp    sampling rate for the input signal in Hz (e.g. 24,000 Hz)

    Returned values:
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
