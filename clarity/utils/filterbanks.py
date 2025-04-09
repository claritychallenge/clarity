"""Module for generating filter banks for audio processing."""

from collections import OrderedDict

import numpy as np
from numba import njit
from numpy import ndarray
from scipy.signal import lfilter


class Gammatone:
    """Gammatone Filter"""

    def __init__(
        self,
        center_freq: float,
        sample_rate: float,
        ear_q: float = 9.264,
        min_bw: float = 24.7,
    ):
        """
        Initialize the Gammatone filter.

        Args:
            center_freq (float): The center frequency of the Gammatone filter.
            sample_rate (float): The sample rate
            ear_q (float): Quality factor of the filter.
            min_bw (float): Minimum bandwidth of the filter.
        """
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.ear_q = ear_q
        self.min_bw = min_bw

        # Filter Equivalent Rectangular Bandwidth from Moore and Glasberg (1983)
        # doi: 10.1121/1.389861
        self.erb = self.min_bw + (self.center_freq / self.ear_q)
        self.tpt = 2 * np.pi / self.sample_rate

        self.center_freq_sin = None
        self.center_freq_cos = None

    def __call__(self, signal: ndarray, bandwidth: float) -> ndarray:
        """Applies the filtered signal using the Gammatone filter.

        Args:
            signal (ndarray): The signal to be filtered.
            bandwidth (float): The bandwidth of the filter.
        """

        # Compute the center frequency sin and cos for current sample rate and
        # center frequency specified
        if self.center_freq_sin is None or self.center_freq_cos is None:
            npts = len(signal)
            self.center_freq_sin, self.center_freq_cos = (
                gammatone_bandwidth_demodulation(npts, self.tpt, self.center_freq)
            )

        tpt_bw = bandwidth * self.tpt * self.erb * 1.019
        a = np.exp(-tpt_bw)
        a_1 = 4.0 * a
        a_2 = -6.0 * a * a
        a_3 = 4.0 * a * a * a
        a_4 = -a * a * a * a
        a_5 = 4.0 * a * a
        gain = 2.0 * (1 - a_1 - a_2 - a_3 - a_4) / (1 + a_1 + a_5)

        ureal = lfilter(
            [1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], signal * self.center_freq_cos
        )
        uimag = lfilter(
            [1, a_1, a_5], [1, -a_1, -a_2, -a_3, -a_4], signal * self.center_freq_sin
        )

        basilar_membrane = gain * (
            ureal * self.center_freq_cos + uimag * self.center_freq_sin
        )
        envelope = gain * np.sqrt(ureal * ureal + uimag * uimag)

        return basilar_membrane, envelope


class Filterbank:
    """A filterbank using different filter"""

    def __init__(
        self,
        center_frequencies: list | float,  # type: ignore
        sample_rate: float,
        filter_type: str = "gammatone",
    ):
        """
        Constructor filterbanks. Class allows different kind of filters but
        it is fixed to GAMMATONE for now

        Args:
            center_frequencies: list with center frequency of the filterbank in Hz
            sample_rate: sampling rate of the signals
            filter_type: the filter to use, One of ['gammatone']
        """

        if isinstance(center_frequencies, float):
            center_frequencies = [center_frequencies]

        self.n_filter = len(center_frequencies)
        self.filters = OrderedDict()

        for cf in sorted(center_frequencies):
            if filter_type == "gammatone":
                self.filters[cf] = Gammatone(cf, sample_rate)
            else:
                raise Exception(f"Filter {filter_type} not supported.")

    def __call__(
        self, signal: ndarray, bandwidth: float | list  # type: ignore
    ) -> tuple[ndarray, ndarray]:
        """Compute the filterbank.

        Args:
            signal (ndarray): The signal to be filtered.
            bandwidth (float): The bandwidth of the filters. If a list, it has to
            be the same length of center frequencies
        """

        if isinstance(bandwidth, float):
            bandwidth = [bandwidth] * self.n_filter

        bank_env = []
        bank_bm = []
        for idx, item in enumerate(self.filters.items()):
            cf, filter = item
            bm, env = filter(signal, bandwidth[idx])
            bank_env.append(env)
            bank_bm.append(bm)

        return np.array(bank_bm), np.array(bank_env)


@njit
def gammatone_bandwidth_demodulation(
    npts: int,
    tpt: float,
    center_freq: float,
    center_freq_cos: ndarray | None = None,  # type: ignore
    center_freq_sin: ndarray | None = None,  # type: ignore
) -> tuple[ndarray, ndarray]:
    """Create the carriers for demodulaton, using the 2d Rotation method from
      https://ccrma.stanford.edu/~jos/pasp/Digital_Sinusoid_Generators.html
    to generate the sin and cos components.  More efficient, perhaps, than
    calculating the sin and cos at each point in time.

    Arguments:
        npts (): How many points are needed.
        tpt (): Phase change (2pi/T) due to each sample time.
        center_freq (): The carrier frequency
        center_freq_cos (ndarray | None): Array to overwrite for the output.
        center_freq_sin (ndarray | None): Array to overwrite for the output.

    Returns:
        sincf (): Samples of the carrier frequency in sin phase.
        coscf (): Samples of the carrier frequency in cos phase.
    """
    if center_freq_cos is None or center_freq_sin is None:
        center_freq_cos = np.zeros(npts)
        center_freq_sin = np.zeros(npts)

    cos_n = np.cos(tpt * center_freq)
    sin_n = np.sin(tpt * center_freq)
    cold = 1.0
    sold = 0.0
    center_freq_cos[0] = cold
    center_freq_sin[0] = sold
    for n in range(1, npts):
        arg = cold * cos_n + sold * sin_n
        sold = sold * cos_n - cold * sin_n
        cold = arg
        center_freq_cos[n] = cold
        center_freq_sin[n] = sold

    return center_freq_sin, center_freq_cos
