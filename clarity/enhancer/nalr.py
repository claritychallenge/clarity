import math
from typing import Optional, Tuple, Union

import numpy as np
import scipy
import scipy.signal


def firwin2(
    n: int,
    f: np.ndarray,
    a: np.ndarray,
    window: Optional[Union[str, float]] = None,
    antisymmetric: Optional[bool] = None,
) -> np.ndarray:
    """FIR filter design using the window method.
    Partial implementation of scipy firwin2 but using our own MATLAB-derived fir2.

    Args:
        n (int): The number of taps in the FIR filter.
        f (ndarray): The frequency sampling points. 0.0 to 1.0 with 1.0 being Nyquist.
        a (ndarray): The filter gains at the frequency sampling points.
        window (string or (string, float), optional): See scipy.firwin2 (default: (None))
        _antisymmetric (bool, optional): Unused but present to main compatability
            with scipy firwin2.
    Returns:
        np.ndarray:  The filter coefficients of the FIR filter, as a 1-D array of length n.
    """
    window_shape = None
    window_type: Union[str, float, None]
    window_param: Union[str, float]
    if isinstance(window, tuple):
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


def fir2(
    nn: int, ff: np.ndarray, aa: np.ndarray, npt: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """FIR arbitrary shape filter design using the frequency sampling method.
    Translation of MATLAB fir2.

    Args:
        nn (int): Order
        ff (np.ndarray): Frequency breakpoints (0 < F < 1) where 1 is Nyquist rate.
                        First and last elements must be 0 and 1 respectively
        aa (np.ndarray): Magnitude breakpoints
        npt (int, optional): Number of points for freq response interpolation
            (default: max (smallest power of 2 greater than nn, 512))

    Returns:
        np.ndarray: nn + 1 filter coefficients
    """
    # Work with filter length instead of filter order
    nn += 1

    if npt is None:
        npt = 2.0 ** np.ceil(math.log(nn) / math.log(2)) if nn >= 1024 else 512
        wind = scipy.signal.hamming(nn)
    else:
        wind = npt
        npt = 2.0 ** np.ceil(math.log(nn) / math.log(2)) if nn >= 1024 else 512
    lap = int(np.fix(npt / 25))

    nbrk = max(len(ff), len(aa))

    ff[0] = 0
    ff[nbrk - 1] = 1

    H = np.zeros(npt + 1)
    nint: int = nbrk - 1
    df = np.diff(ff, n=1)

    npt += 1
    nb: int = 0
    H[0] = aa[0]

    for i in np.arange(nint):
        if df[i] == 0:
            nb = int(np.ceil(nb - lap / 2))
            ne = nb + lap - 1
        else:
            ne = int(np.fix(ff[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)
        inc: Union[float, np.ndarray] = 0.0 if nb == ne else (j - nb) / (ne - nb)
        H[nb : (ne + 1)] = (inc * aa[i + 1]) + ((1 - inc) * aa[i])
        nb = ne + 1

    dt: float = 0.5 * (nn - 1)
    rad = -dt * 1j * math.pi * np.arange(0, npt) / (npt - 1)
    H = H * np.exp(rad)

    H = np.concatenate((H, H[npt - 2 : 0 : -1].conj()))
    ht = np.real(np.fft.ifft(H))

    b = ht[0:nn] * wind

    return b, 1


class NALR:
    def __init__(self, nfir: int, fs: int):
        """
        Args:
            nfir: Order of the NAL-R EQ filter and the matching delay
            fs: Sampling rate in Hz
        """
        self.nfir = nfir
        # Processing parameters
        self.fmax = 0.5 * fs

        # Audiometric frequencies
        self.aud = np.array([250, 500, 1000, 2000, 4000, 6000])

        # Design a flat filter having the same delay as the NAL-R filter
        self.delay = np.zeros(nfir + 1)
        self.delay[nfir // 2] = 1.0

    def hl_interp(self, hl: np.ndarray, cfs: np.ndarray):
        try:
            hl_interpf = scipy.interpolate.interp1d(cfs, hl)
        except ValueError:
            raise ValueError(
                "Hearing losses (hl) and center frequencies (cfs) don't match!"
            )
        return hl_interpf(self.aud)

    def build(
        self,
        hl: np.ndarray,
        cfs: np.ndarray = None,
    ):
        """
        Args:
            hl: hearing thresholds at [250, 500, 1000, 2000, 4000, 6000] Hz
            cfs: center frequencies of the hearing thresholds. If None, the default
                values are used.
        Returns:
            NAL-R FIR filter
            delay
        """
        if cfs is None:
            cfs = np.array([250, 500, 1000, 2000, 3000, 6000])

        hl = self.hl_interp(np.array(hl), np.array(cfs))
        mloss = np.max(hl)

        if mloss > 0:
            # Compute the NAL-R frequency response at the audiometric frequencies
            bias = np.array([-17, -8, 1, -1, -2, -2])
            t3 = hl[1] + hl[2] + hl[3]
            if t3 <= 180:
                xave = 0.05 * t3
            else:
                xave = 9.0 + 0.116 * (t3 - 180)
            gdB = xave + 0.31 * hl + bias
            gdB = np.clip(gdB, a_min=0, a_max=None)

            # Design the linear-phase FIR filter
            fv = np.append(
                np.append(0, self.aud), self.fmax
            )  # Frequency vector for the interpolation
            cfreq = (
                np.linspace(0, self.nfir, self.nfir + 1) / self.nfir
            )  # Uniform frequency spacing from 0 to 1
            gdBv = np.append(
                np.append(gdB[0], gdB), gdB[-1]
            )  # gdB vector for the interpolation
            interpf = scipy.interpolate.interp1d(fv, gdBv)
            gain = interpf(self.fmax * cfreq)
            glin = np.power(10, gain / 20.0)
            nalr = firwin2(self.nfir + 1, cfreq, glin)
        else:
            nalr = self.delay.copy()
        return nalr, self.delay

    def apply(self, nalr: np.ndarray, wav: np.ndarray):
        """
        Args:
            nalr: built NAL-R FIR filter
            wav: one dimensional wav signal

        Returns:
            amplified signal
        """
        return np.convolve(wav, nalr)
