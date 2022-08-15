"""Gammatone filterbank simulation of the Cochlea."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import scipy
from scipy import signal

from clarity.evaluator.msbg.audiogram import Audiogram
from clarity.evaluator.msbg.msbg_utils import read_gtf_file
from clarity.evaluator.msbg.smearing import Smearer

# TODO: Fix power overflow error when (expansion_ratios[ixch] - 1) < 0


@dataclass
class FilterBank:
    """Holds the numerators and demoninators of an IIR filter bank."""

    nums: np.ndarray
    denoms: np.ndarray


# Parameters for smearing and gammatone filtering according to degree of loss
HL_PARAMS: Dict[str, Dict[str, Union[str, tuple]]] = {
    "SEVERE": {
        "gtfbank_file": "GT4FBank_Brd3.0E_Spaced2.3E_44100Fs",
        "smear_params": (4, 2),  # asymmetric severe smearing
    },
    "MODERATE": {
        "gtfbank_file": "GT4FBank_Brd2.0E_Spaced1.5E_44100Fs",
        "smear_params": (2.4, 1.6),  # asymmetric moderate smearing
    },
    "MILD": {
        "gtfbank_file": "GT4FBank_Brd1.5E_Spaced1.1E_44100Fs",
        "smear_params": (1.6, 1.1),  # asymmetric mild smearing
    },
    "NOTHING": {
        "gtfbank_file": "GT4FBank_Brd1.5E_Spaced1.1E_44100Fs",
        "smear_params": (1.0, 1.0),  # No smearing
    },
}


def compute_recruitment_parameters(
    gtn_cf: np.ndarray, audiogram: Audiogram, catch_up: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute parameters to be used in recruitment model.

    Computes expansion ratios for each gammatone filterbank channel and
    the equal loudness catch up level per channel (currently this is
    a fixed value per channel)

    Args:
        gtn_cf (ndarray): gammatone filterbank centre frequencies
        audiogram (Audiogram): the audiogram to employ
        catch_up (float): level in dB at which catches up with NH

    Returns:
        ndarray: expansion ratio for each gammatone filterbank channel

    """
    cf_expansion = np.zeros(gtn_cf.shape)  # expansion ratios

    for ix_cf in np.arange(0, gtn_cf.shape[0]):
        if gtn_cf[ix_cf] < audiogram.cfs[0]:
            # Extend audiogram, flat below lowest freq measured
            cf_expansion[ix_cf] = catch_up / (catch_up - audiogram.levels[0])
        elif gtn_cf[ix_cf] > audiogram.cfs[-1]:
            # Extend audiogram, flat above highest freq measured
            cf_expansion[ix_cf] = catch_up / (catch_up - audiogram.levels[-1])
        else:
            # In the sensible region
            cf_to_level_fn = scipy.interpolate.interp1d(
                audiogram.cfs, audiogram.levels, kind="linear"
            )
            # Assumes catch-up at catch_up dB (typ 100-105)
            cf_expansion[ix_cf] = catch_up / (catch_up - cf_to_level_fn(gtn_cf[ix_cf]))

    #  Loudness catch-up levels - currently same for all channels
    eq_loud_db_catch_up = catch_up * np.ones(gtn_cf.shape)

    return cf_expansion, eq_loud_db_catch_up


def gammatone_filterbank(
    x: np.ndarray,
    ngamma: int,
    gtn_filters: FilterBank,
    gtn_delays: np.ndarray,
    start2poleHP: int,
    hp_filters: FilterBank,
) -> np.ndarray:
    """Pass signal through gammatone filterbank.

    Args:
        x (ndarray): input signal
        ngamma (int): 4, gammatone order
        gtn_filters (FilterBank): set of gammatone filters
        gtn_delays (ndarray): gammatone filter delays
        start2poleHP (int): parameter value from gtfbank_file
        hp_filter (FilterBarker): set of highpass filters

    Returns:
        ndarray: cochleagram with gtn_denoms.shape[0] channels of len(x)

    """
    n_chans = gtn_filters.denoms.shape[0]
    n_samples = len(x)

    cochleagram = np.zeros((n_chans, n_samples))

    # Implement Nth order Gammatone filterbank
    for ixch in np.arange(0, n_chans):
        pass_n = x
        for __ in np.arange(0, ngamma):
            pass_n = signal.lfilter(
                gtn_filters.nums[ixch, :], gtn_filters.denoms[ixch, :], pass_n
            )

        dly = gtn_delays[ixch]
        # Time-advance signal to compensate for IIR shift, zero-pad remainder
        pass_n[np.arange(0, (n_samples - dly))] = pass_n[np.arange(dly, n_samples)]
        pass_n[np.arange(n_samples - dly, n_samples)] = 0

        # ...possibly control lf tails with second order highpass
        if ixch >= (start2poleHP - 1):
            ix_hp = ixch - start2poleHP + 1
            pass_n = signal.lfilter(
                hp_filters.nums[ix_hp, :], hp_filters.denoms[ix_hp, :], pass_n
            )
        cochleagram[ixch, :] = pass_n
    return cochleagram


def compute_envelope(
    coch_sig: np.ndarray, erbn_cf: np.ndarray, fs: float
) -> np.ndarray:
    """Obtain signal envelope.

    Envelope computed using full-wave rectification and low-pass filter

    Args:
        coch_sig (ndarray): input signal
        erbn_cf (ndarray): ERB centre frequencies
        fs (float): sampling frequency

    Returns:
        ndarray: signal envelope

    """

    envelope = np.zeros(coch_sig.shape)
    n_chans = coch_sig.shape[0]
    for ixch in np.arange(0, n_chans):
        # Extract envelope; channel envelope lpf is NOT fixed for all channels,
        # tracks slightly above ERBn
        # Don't need to worry about phase since using bidirectional filters
        # Put limit on bandwidth for very high freq channels, (30/40) == -3dB at fc,
        fc_envlp = (30 / 40) * min(100, erbn_cf[ixch])

        # Reduced rate of cut(4-pole), but improved time response of filter.
        chan_lpfB, chan_lpfA = signal.ellip(2, 0.25, 35, fc_envlp / (fs / 2))

        # Full-wave rectify (take absolute values of raw signal coch_sig) and
        # low-pass to get envelope
        padlen = 3 * (max(len(chan_lpfA), len(chan_lpfB))) - 1
        envelope[ixch, :] = signal.filtfilt(
            chan_lpfB, chan_lpfA, np.abs(coch_sig[ixch, :]), padlen=padlen
        )
    return envelope


def recruitment(
    coch_sig: np.ndarray,
    envelope: np.ndarray,
    SPL_equiv_0dB: float,
    expansion_ratios: np.ndarray,
    eq_loud_db: np.ndarray,
) -> np.ndarray:
    """Simulate loudness recruitment.

    Args:
        coch_sig (ndarray): input signal
        envelope (ndarray): signal envelope
        SPL_equiv_0dB (float): equivalent level in dB SPL of 0 dB Full Scale
        expansion_ratios (ndarray): expansion ratios for expanding channel
            signals
        eq_loud_db (ndarray): loudness catch-up level in dB

    Returns:
        ndarray: cochlear output signal

    """

    n_chans = coch_sig.shape[0]
    for ixch in np.arange(0, n_chans):
        # Limit max envelope, and hence maximum gain
        envlp_out = envelope[ixch, :]
        envlp_max = np.power(10, 0.05 * (eq_loud_db[ixch] - SPL_equiv_0dB))
        envlp_out[envlp_out > envlp_max] = envlp_max
        envlp_out[envlp_out < 1e-9] = 1e-9

        gain = (envlp_out / envlp_max) ** (expansion_ratios[ixch] - 1)
        coch_sig[ixch, :] = coch_sig[ixch, :] * gain

    return coch_sig


class Cochlea:
    """Simulate the cochlea.

    Includes simulation of effects of bandwidth broadening (smearing) and recruitment.
    Implements 3 different degrees of impairment which affect the degree of smearing.
    Recruitment currently always with x2 broadening.

    Degree of hearing impairment used to control the following filterbank variables:
    BROADEN, SPACING, NGAMMA, Fs, N_Chans,
    ERBn_CentFrq, GTn_CentFrq, GTnDelays,
    GTn_denoms, GTn_nums, Start2PoleHP, HP_FCorner, HP_denoms,
    HP_nums, HP_Delays,  Recombination_dB
    """

    def __init__(
        self, audiogram: Audiogram, catch_up_level: float = 105.0, fs: float = 44100.0
    ):
        """Cochlea constructor.

        Args:
            audiogram (Audiogram): Audiogram characterising hearing loss
            catch_up_level (float, optional): loudness catch-up level in dB (default: {105})
            fs (float, optional): sampling frequency

        """
        self.fs = fs
        self.audiogram = audiogram  # Audiogram to employ
        self.catch_up_level = catch_up_level  # Level in dB at which catches up with NH

        # Compute severity level and set parameters accordingly
        severity_level = self.audiogram.severity
        self.gtfbank_params = read_gtf_file(
            f"msbg_hparams/{HL_PARAMS[severity_level]['gtfbank_file']}.json"
        )
        self.cf_expansion, self.eq_loud_db_catch_up = compute_recruitment_parameters(
            self.gtfbank_params["GTn_CentFrq"], audiogram, catch_up_level
        )

        # Set-up the smearer
        self.smearer = None
        if severity_level != "NOTHING":
            smear_params = HL_PARAMS[severity_level]["smear_params"]
            self.smearer = Smearer(smear_params[0], smear_params[1], fs)

        logging.info("Severity level - %s", severity_level)

    def simulate(self, coch_sig: np.ndarray, equiv_0dB_file_SPL: float) -> np.ndarray:
        """Pass a signal through the cochlea.

        Args:
            coch_sig (ndarray): input signal
            equiv_0dB_file_SPL (float): equivalent level in dB SPL of 0 dB Full Scale

        Returns:
            ndarray: cochlear output signal

        """
        if self.smearer is not None:
            coch_sig = self.smearer.smear(coch_sig)

        # Filter with gammatone filterbank
        coch_sig_out = gammatone_filterbank(
            coch_sig,
            self.gtfbank_params["NGAMMA"],
            FilterBank(
                self.gtfbank_params["GTn_nums"], self.gtfbank_params["GTn_denoms"]
            ),
            self.gtfbank_params["GTnDelays"],
            self.gtfbank_params["Start2PoleHP"],
            FilterBank(
                self.gtfbank_params["HP_nums"], self.gtfbank_params["HP_denoms"]
            ),
        )

        # Simulate loudness recruitment
        envelope = compute_envelope(
            coch_sig_out, np.array(self.gtfbank_params["ERBn_CentFrq"]), self.fs
        )

        coch_sig_out = recruitment(
            coch_sig_out,
            envelope,
            equiv_0dB_file_SPL,
            self.cf_expansion,
            self.eq_loud_db_catch_up,
        )

        # Recombine channels to obtain output signal
        coch_sig_out = np.sum(coch_sig_out, axis=0) * np.power(
            10, -0.05 * self.gtfbank_params["Recombination_dB"]
        )

        return coch_sig_out
