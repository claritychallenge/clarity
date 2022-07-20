import logging
import math

import numpy as np
from scipy import interpolate, signal
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

# Cut off frequency of low-pass filter at end of simulations:
# prevents possible excessive processing noise at high frequencies.
UPPER_CUTOFF_HZ = 18000


class Ear:
    """Representation of a pairs of ears."""

    def __init__(self, src_pos="ff", fs=44100, equiv0dBSPL=100, ahr=20):
        """
        Constructor for the Ear class.
        Args:
            src_pos (str): Position of the source
        """
        self.fs = fs
        self.calibration_signal = None
        self.src_correction = self.get_src_correction(src_pos)
        self.equiv0dBSPL = equiv0dBSPL
        self.ahr = ahr
        self.cochlea = None

    def set_audiogram(self, audiogram):
        if np.max(audiogram.levels[audiogram.levels is not None]) > 80:
            logging.warning(
                "Impairment too severe: Suggest you limit audiogram max to 80-90 dB HL, \
                otherwise things go wrong/unrealistic."
            )
        self.cochlea = Cochlea(audiogram=audiogram)

    @staticmethod
    def get_src_correction(src_pos):
        """Select relevant external field to eardrum correction.

        Args:
            src_pos (str): Position of src. One of ff, df or ITU

        """
        if src_pos == "ff":
            src_correction = FF_ED
        elif src_pos == "df":
            src_correction = DF_ED
        elif src_pos == "ITU":  # transfer to same grid
            f = interpolate.interp1d(ITU_HZ, ITU_ERP_DRP, kind="linear")
            src_correction = f(HZ)
        else:
            logging.error(
                "Invalid src position (%s). Must be one of ff, df or ITU", src_pos
            )
            raise ValueError("Invalid src position")
        return src_correction

    @staticmethod
    def src_to_cochlea_filt(ip_sig, src_correction, fs, backward=False):
        """Simulate middle and outer ear transfer functions.

        Made more general, Mar2012, to include diffuse field as well as ITU reference points,
        that were included in DOS-versions of recruitment simulator, released ca 1999-2001,
        and on hearing group website, Mar2012 variable [src_pos] takes one of 3
        values: 'ff', 'df' and 'ITU' free-field to cochlea filter forwards or backward direction,
        depends on 'backward' switch. NO LONGER via 2 steps. ff to eardrum and then via middle ear:
        use same length FIR 5-12-97.

        Args:
            ip_sig (ndarray): signal to process
            src_correction (str): correction to make for src position
            fs (int): sampling frequency
            backward (bool, optional): if true then cochlea to src (default: False)

        Returns:
            ndarray: the processed signal

        """
        logging.info("performing outer/middle ear corrections")

        # make sure that response goes only up to fs/2
        nyquist = int(fs / 2)
        ixf_useful = np.nonzero(HZ < nyquist)

        hz_used = HZ[ixf_useful]
        hz_used = np.append(hz_used, nyquist)

        # sig from free field to cochlea: 0 dB at 1kHz
        correction = src_correction - MIDEAR
        f = interpolate.interp1d(HZ, correction, kind="linear")
        last_correction = f(nyquist)  # generate synthetic response at Nyquist

        correction_used = np.append(correction[ixf_useful], last_correction)
        if backward:  # ie. coch->src rather than src->coch
            correction_used = -correction_used
        correction_used = np.power(10, (0.05 * correction_used))

        correction_used = correction_used.flatten()
        # Create filter with 23 msec window to do reasonable job down to about 100 Hz
        # Scales with fs, falls over with longer windows in fir2 in original MATLAB version
        n_wdw = 2 * math.floor((fs / 16e3) * 368 / 2)
        hz_used = hz_used / nyquist

        b = firwin2(n_wdw + 1, hz_used.flatten(), correction_used, window=("kaiser", 4))
        op_sig = signal.lfilter(b, 1, ip_sig)

        return op_sig

    def make_calibration_signal(self, REF_RMS_DB):
        """Add the calibration signal to the start of the signal.

        Args:
            signal (ndarray): input signal

        Returns:
            ndarray: the processed signal

        """
        # Calibration noise and tone with same RMS as original speech,
        # Tone at nearest channel centre frequency to 500 Hz
        # For testing, ref_rms_dB must be equal to -31.2

        noise_burst = gen_eh2008_speech_noise(duration=2, fs=self.fs, level=REF_RMS_DB)
        tone_burst = gen_tone(freq=520, duration=0.5, fs=self.fs, level=REF_RMS_DB)
        silence = np.zeros(int(0.05 * self.fs))  # 50 ms duration
        return (
            np.concatenate((silence, tone_burst, silence, noise_burst, silence)),
            silence,
        )

    @staticmethod
    def array_to_list(chans):
        """Convert ndarray into a list of 1-D arrays."""
        if len(chans.shape) == 1:
            chans = chans[..., np.newaxis]
        return [chans[:, i] for i in range(chans.shape[1])]

    def process(self, chans, add_calibration=False):
        """Run the hearing loss simulation.

        Args:
            chans (ndarray): signal to process, shape either N, Nx1, Nx2
            add_calibration (bool): prepend calibration tone and speech-shaped noise
                (default: False)

        Returns:
            ndarray: the processed signal

        """

        fs = 44100  # This is the only sampling frequency that can be used
        if fs != self.fs:
            logging.error(
                "Warning: only a sampling frequency of 44.1kHz can be used by MSBG."
            )

        logging.info("Processing %s samples", len(chans))

        # Get single channel array and convert to list
        chans = Ear.array_to_list(chans)

        # Need to know file RMS, and then call that a certain level in SPL:
        # needs some form of pre-measuring.
        levelreFS = 10 * np.log10(np.mean(np.array(chans) ** 2))

        equiv_0dB_SPL = self.equiv0dBSPL + self.ahr

        leveldBSPL = equiv_0dB_SPL + levelreFS
        CALIB_DB_SPL = leveldBSPL
        TARGET_SPL = leveldBSPL
        REF_RMS_DB = CALIB_DB_SPL - equiv_0dB_SPL

        # Measure RMS where 3rd arg is dB_rel_rms (how far below)
        calculated_rms, idx, _rel_dB_thresh, _active = measure_rms(chans[0], fs, -12)

        # Rescale input data and check level after rescaling
        # This is to ensure that the following processing steps are applied correctly
        change_dB = TARGET_SPL - (equiv_0dB_SPL + 20 * np.log10(calculated_rms))
        chans = [x * np.power(10, 0.05 * change_dB) for x in chans]
        new_rms_db = equiv_0dB_SPL + 10 * np.log10(
            np.mean(np.power(chans[0][idx], 2.0))
        )
        logging.info(
            "Rescaling: leveldBSPL was %3.1f dB SPL, now %3.1f dB SPL. Target SPL is %3.1f dB SPL.",
            leveldBSPL,
            new_rms_db,
            TARGET_SPL,
        )

        # Add calibration signal at target SPL dB
        if add_calibration is True:
            if self.calibration_signal is None:
                self.calibration_signal = self.make_calibration_signal(REF_RMS_DB)
            chans = [
                np.concatenate(
                    (self.calibration_signal[0], x, self.calibration_signal[1])
                )
                for x in chans
            ]

        # Transform from src pos to cochlea, simulate cochlea, transform back to src pos
        chans = [Ear.src_to_cochlea_filt(x, self.src_correction, fs) for x in chans]
        chans = [self.cochlea.simulate(x, equiv_0dB_SPL) for x in chans]
        chans = [
            Ear.src_to_cochlea_filt(x, self.src_correction, fs, backward=True)
            for x in chans
        ]

        # Implement low-pass filter at top end of audio range: flat to Cutoff freq, tails
        # below -80 dB. Suitable lpf for signals later converted to MP3, flat to 15 kHz.
        # Small window to design low-pass FIR, to cut off high freq processing noise
        # low-pass to something sensible, prevents exaggeration of > 15 kHz
        winlen = 2 * math.floor(0.0015 * fs) + 1
        lpf44d1 = firwin(winlen, UPPER_CUTOFF_HZ / int(fs / 2), window=("kaiser", 8))
        chans = [lfilter(lpf44d1, 1, x) for x in chans]

        return chans
