"""Implementation of the MSBG hearing loss model."""
from __future__ import annotations

import logging
import math
from typing import Final

import numpy as np
import scipy
from numpy import ndarray
from scipy.signal import firwin, lfilter

from clarity.evaluator.msbg.cochlea import Cochlea
from clarity.evaluator.msbg.msbg_utils import (
    DF_ED,
    FF_ED,
    HZ,
    ITU_ERP_DRP,
    ITU_HZ,
    MIDEAR,
    firwin2,
    gen_eh2008_speech_noise,
    gen_tone,
    measure_rms,
)
from clarity.utils.audiogram import Audiogram

# Cut off frequency of low-pass filter at end of simulations:
# prevents possible excessive processing noise at high frequencies.
UPPER_CUTOFF_HZ: Final = 18000


class Ear:
    """Representation of a pairs of ears."""

    def __init__(
        self,
        src_pos: str = "ff",
        sample_rate: float = 44100.0,
        equiv_0db_spl: float = 100.0,
        ahr: float = 20.0,
    ) -> None:
        """
        Constructor for the Ear class.
        Args:
            src_pos (str): Position of the source.
            sample_rate (float): sample frequency.
            equiv_0db_spl (): ???
            ahr (): ???
        """
        self.sample_rate = sample_rate
        self.src_correction = self.get_src_correction(src_pos)
        self.equiv_0db_spl = equiv_0db_spl
        self.ahr = ahr
        self.cochlea: Cochlea | None = None

    def set_audiogram(self, audiogram: Audiogram) -> None:
        """Set the audiogram to be used."""
        if np.max(audiogram.levels[audiogram.levels is not None]) > 80:
            logging.warning(
                "Impairment too severe: Suggest you limit audiogram max to"
                "80-90 dB HL, otherwise things go wrong/unrealistic."
            )
        self.cochlea = Cochlea(audiogram=audiogram)

    @staticmethod
    def get_src_correction(src_pos: str) -> ndarray:
        """Select relevant external field to eardrum correction.

        Args:
            src_pos (str): Position of src. One of ff, df or ITU

        """
        if src_pos == "ff":
            src_correction = FF_ED
        elif src_pos == "df":
            src_correction = DF_ED
        elif src_pos == "ITU":  # transfer to same grid
            field = scipy.interpolate.interp1d(ITU_HZ, ITU_ERP_DRP, kind="linear")
            src_correction = field(HZ)
        else:
            logging.error(
                f"Invalid src position ({src_pos}). Must be one of ff, df or ITU"
            )
            raise ValueError("Invalid src position")
        return src_correction

    @staticmethod
    def src_to_cochlea_filt(
        input_signal: ndarray,
        src_correction: ndarray,
        sample_rate: float,
        backward: bool = False,
    ) -> ndarray:
        """Simulate middle and outer ear transfer functions.

        Made more general, Mar2012, to include diffuse field as well as ITU reference
        points, that were included in DOS-versions of recruitment simulator, released
        ca 1999-2001, and on hearing group website, Mar2012 variable [src_pos] takes one
        of 3 values: 'ff', 'df' and 'ITU' free-field to cochlea filter forwards or
        backward direction, depends on 'backward' switch. NO LONGER via 2 steps. ff to
        eardrum and then via middle ear: use same length FIR 5-12-97.

        Args:
            input_signal (ndarray): signal to process
            src_correction (np.ndarray): correction to make for src position as an array
                returned by get_src_correction(src_pos) where src_pos is one of ff, df
                or ITU
            sample_rate (int): sampling frequency
            backward (bool, optional): if true then cochlea to src (default: False)

        Returns:
            np.ndarray: the processed signal

        """
        logging.info("performing outer/middle ear corrections")

        # make sure that response goes only up to sample_frequency/2
        nyquist = int(sample_rate / 2.0)
        ixf_useful = np.nonzero(HZ < nyquist)

        hz_used = HZ[ixf_useful]
        hz_used = np.append(hz_used, nyquist)

        # sig from free field to cochlea: 0 dB at 1kHz
        correction = src_correction - MIDEAR
        field = scipy.interpolate.interp1d(HZ, correction, kind="linear")
        last_correction = field(nyquist)  # generate synthetic response at Nyquist

        correction_used = np.append(correction[ixf_useful], last_correction)
        if backward:  # ie. coch->src rather than src->coch
            correction_used = -correction_used
        correction_used = np.power(10, (0.05 * correction_used))

        correction_used = correction_used.flatten()
        # Create filter with 23 msec window to do reasonable job down to about 100 Hz
        # Scales with fs, fails with longer windows in fir2 in original MATLAB version
        n_wdw = 2 * math.floor((sample_rate / 16e3) * 368 / 2)
        hz_used = hz_used / nyquist

        b = firwin2(n_wdw + 1, hz_used.flatten(), correction_used, window=("kaiser", 4))
        output_signal = scipy.signal.lfilter(b, 1, input_signal)

        return output_signal

    def make_calibration_signal(
        self, ref_rms_db: float, n_channels: int = 1
    ) -> tuple[ndarray, ndarray]:
        """Add the calibration signal to the start of the signal.

        Args:
            ref_rms_db (float): reference rms level in dB

        Returns:
            tuple[ndarray, ndarray] - pre and post calibration signals

        """
        # Calibration noise and tone with same RMS as original speech,
        # Tone at nearest channel centre frequency to 500 Hz
        # For testing, ref_rms_dB must be equal to -31.2

        noise_burst = gen_eh2008_speech_noise(
            duration=2, sample_rate=self.sample_rate, level=ref_rms_db
        )
        tone_burst = gen_tone(
            freq=520,
            duration=0.5,
            sample_rate=self.sample_rate,
            level=ref_rms_db,
        )
        silence = np.zeros(int(0.05 * self.sample_rate))  # 50 ms duration

        pre_calibration = np.concatenate(
            (silence, tone_burst, silence, noise_burst, silence)
        )

        # Repeat signals for the desired number of channels
        post_calibration = np.tile(silence[np.newaxis, ...], (n_channels, 1))
        pre_calibration = np.tile(pre_calibration[np.newaxis, ...], (n_channels, 1))

        return (pre_calibration, post_calibration)

    def process(self, signal: ndarray, add_calibration: bool = False) -> list[ndarray]:
        """Run the hearing loss simulation.

        Args:
            signal (ndarray): signal to process, shape either N, Nx1, Nx2
            add_calibration (bool): prepend calibration tone and speech-shaped noise
                (default: False)

        Returns:
            np.ndarray: the processed signal

        """

        signal = signal.T  # signals as rows
        if len(signal.shape) == 1:
            signal = signal[np.newaxis, ...]
        sample_rate = 44100  # This is the only sampling frequency that can be used
        if sample_rate != self.sample_rate:
            logging.error(
                "Warning: only a sampling frequency of 44.1kHz can be used by MSBG."
            )
            raise ValueError("Invalid sampling frequency, valid value is 44100")

        logging.info("Processing {len(chans)} samples")

        # Need to know file RMS, and then call that a certain level in SPL:
        # needs some form of pre-measuring.
        signal_rms_level_db = 10 * np.log10(np.mean(np.array(signal) ** 2))

        equiv_0db_spl = self.equiv_0db_spl + self.ahr

        level_db_spl = equiv_0db_spl + signal_rms_level_db
        calib_db_spl = level_db_spl
        target_spl = level_db_spl
        ref_rms_db = calib_db_spl - equiv_0db_spl

        # Measure RMS where 3rd arg is dB_rel_rms (how far below)
        calculated_rms, idx, _rel_db_thresh, _active = measure_rms(
            signal[0], sample_rate, -12
        )

        # Rescale input data and check level after rescaling
        # This is to ensure that the following processing steps are applied correctly
        change_db = target_spl - (equiv_0db_spl + 20 * np.log10(calculated_rms))
        signal = signal * np.power(10, 0.05 * change_db)
        new_rms_db = equiv_0db_spl + 10 * np.log10(
            np.mean(np.power(signal[0][idx], 2.0))
        )
        logging.info(
            "Rescaling: "
            f"leveldBSPL was {level_db_spl:3.1f} dB SPL, now {new_rms_db:3.1f} dB SPL. "
            f" Target SPL is {target_spl:3.1f} dB SPL."
        )

        # Add calibration signal at target SPL dB
        if add_calibration is True:
            pre_calibration, post_calibration = self.make_calibration_signal(
                ref_rms_db, n_channels=signal.shape[0]
            )
            # signal = [
            #    np.concatenate((calibration_signal[0], x, #calibration_signal[1]))
            #    for x in signal
            # ]
            signal = np.concatenate((pre_calibration, signal, post_calibration), axis=1)

        # Transform from src pos to cochlea, simulate cochlea, transform back to src pos
        signal = Ear.src_to_cochlea_filt(signal, self.src_correction, sample_rate)
        if self.cochlea is not None:
            signal = np.array([self.cochlea.simulate(x, equiv_0db_spl) for x in signal])
        signal = Ear.src_to_cochlea_filt(
            signal, self.src_correction, sample_rate, backward=True
        )

        # Implement low-pass filter at top end of audio range: flat to Cutoff freq,
        # tails below -80 dB. Suitable lpf for signals later converted to MP3, flat to
        # 15 kHz. Small window to design low-pass FIR, to cut off high freq processing
        # noise low-pass to something sensible, prevents exaggeration of > 15 kHz
        winlen = 2 * math.floor(0.0015 * sample_rate) + 1
        lpf44d1 = firwin(
            winlen, UPPER_CUTOFF_HZ / int(sample_rate / 2), window=("kaiser", 8)
        )
        signal_list = [lfilter(lpf44d1, 1, x) for x in signal]

        return signal_list
