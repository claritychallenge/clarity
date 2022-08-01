import logging
import math
import os

import numpy as np
import soundfile
from scipy.signal import convolve
from soundfile import SoundFile

from clarity.data.utils import better_ear_speechweighted_snr, pad, sum_signals


class Renderer:
    """
    SceneGenerator of CEC1 training and development sets. The render() function generates all
    simulated signals for each scene given the parameters specified in the
    metadata/scenes.train.json or metadata/scenes.dev.json file.
    """

    def __init__(
        self,
        input_path,
        output_path,
        num_channels=1,
        fs=44100,
        ramp_duration=0.5,
        tail_duration=0.2,
        pre_duration=2.0,
        post_duration=1.0,
        test_nbits=16,
    ):

        self.input_path = input_path
        self.output_path = output_path
        self.fs = fs
        self.ramp_duration = ramp_duration
        self.n_tail = int(tail_duration * fs)
        self.pre_duration = pre_duration
        self.post_duration = post_duration
        self.test_nbits = test_nbits

        if num_channels == 0:
            # This will only generate the initial target, masker and anechoic target signal
            self.channels = []
        else:
            # ... as above plus N hearing aid input channels plus 'channel 0' (the eardrum signal).
            # e.g. num_channel = 2  => channels [1, 2, 0]
            self.channels = list(range(1, num_channels + 1)) + [0]

    def read_signal(
        self, filename, offset=0, nsamples=-1, nchannels=0, offset_is_samples=False
    ):
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
        except Exception as e:
            # Ensure incorrect error (24 bit) is not generated
            raise Exception(f"Unable to read {filename}.") from e

        if nchannels not in (0, wave_file.channels):
            raise Exception(
                f"Wav file ({filename}) was expected to have {nchannels} channels."
            )

        if wave_file.samplerate != self.fs:
            raise Exception(f"Sampling rate is not {self.fs} for filename {filename}.")

        if not offset_is_samples:  # Default behaviour
            offset = int(offset * wave_file.samplerate)

        if offset != 0:
            wave_file.seek(offset)

        x = wave_file.read(frames=nsamples)
        return x

    def write_signal(self, filename, x, fs, floating_point=True):
        """Write a signal as fixed or floating point wav file."""

        if fs != self.fs:
            logging.warning("Sampling rate mismatch: %s with sr=%s.", filename, fs)
            # raise ValueError("Sampling rate mismatch")

        if floating_point is False:
            if self.test_nbits == 16:
                subtype = "PCM_16"
                # If signal is float and we want int16
                x *= 32768
                x = x.astype(np.dtype("int16"))
                assert np.max(x) <= 32767 and np.min(x) >= -32768
            elif self.test_nbits == 24:
                subtype = "PCM_24"
        else:
            subtype = "FLOAT"

        soundfile.write(filename, x, fs, subtype=subtype)

    def apply_ramp(self, x, dur):
        """Apply half cosine ramp into and out of signal

        dur - ramp duration in seconds
        """
        ramp = np.cos(np.linspace(math.pi, 2 * math.pi, int(self.fs * dur)))
        ramp = (ramp + 1) / 2
        y = np.array(x)
        y[0 : len(ramp)] *= ramp
        y[-len(ramp) :] *= ramp[::-1]
        return y

    def apply_brir(self, signal, brir):
        """Convolve a signal with a BRIR.

        Args:
            signal (ndarray): The mono or stereo signal stored as array of floats
            brir (ndarray): The binaural room impulse response stored a 2xN array of floats
            n_tail (int): Truncate output to input signal length + n_tail

        Returns:
            ndarray: The convolved signals

        """
        output_len = len(signal) + self.n_tail
        brir = np.squeeze(brir)

        if len(np.shape(signal)) == 1 and len(np.shape(brir)) == 2:
            signal_l = convolve(signal, brir[:, 0], mode="full", method="fft")
            signal_r = convolve(signal, brir[:, 1], mode="full", method="fft")
        elif len(np.shape(signal)) == 2 and len(np.shape(brir)) == 2:
            signal_l = convolve(signal[:, 0], brir[:, 0], mode="full", method="fft")
            signal_r = convolve(signal[:, 1], brir[:, 1], mode="full", method="fft")
        else:
            logging.error("Signal does not have the required shape.")
        output = np.vstack([signal_l, signal_r]).T
        return output[0:output_len, :]

    def compute_snr(self, target, noise, pre_samples=0, post_samples=-1):
        """Return the SNR.
        Take the overlapping segment of the noise and get the speech-weighted
        better ear SNR. (Note, SNR is a ratio -- not in dB.)
        """

        pre_samples = int(self.fs * self.pre_duration)
        post_samples = int(self.fs * self.post_duration)

        segment_target = target[pre_samples:-post_samples]
        segment_noise = noise[pre_samples:-post_samples]
        assert len(segment_target) == len(segment_noise)

        snr = better_ear_speechweighted_snr(segment_target, segment_noise)

        return snr

    def render(
        self,
        target,
        noise_type,
        interferer,
        room,
        scene,
        offset,
        snr_dB,
        dataset,
        pre_samples=88200,
        post_samples=44100,
    ):

        brir_stem = f"{self.input_path}/{dataset}/rooms/brir/brir_{room}"
        anechoic_brir_stem = f"{self.input_path}/{dataset}/rooms/brir/anech_brir_{room}"
        target_fn = f"{self.input_path}/{dataset}/targets/{target}.wav"
        interferer_fn = (
            f"{self.input_path}/{dataset}/interferers/{noise_type}/{interferer}.wav"
        )

        target = self.read_signal(target_fn)
        target = np.pad(target, [(pre_samples, post_samples)])

        interferer = self.read_signal(
            interferer_fn, offset=offset, nsamples=len(target), offset_is_samples=True
        )

        if len(target) != len(interferer):
            logging.debug("Target and interferer have different lengths")

        # Apply 500ms half-cosine ramp
        interferer = self.apply_ramp(interferer, dur=self.ramp_duration)

        prefix = f"{self.output_path}/{scene}"
        outputs = [
            (f"{prefix}_target.wav", target),
            (f"{prefix}_interferer.wav", interferer),
        ]

        snr_ref = None
        for channel in self.channels:
            # Load scene BRIRs
            target_brir_fn = f"{brir_stem}_t_CH{channel}.wav"
            interferer_brir_fn = f"{brir_stem}_i1_CH{channel}.wav"
            target_brir = self.read_signal(target_brir_fn)
            interferer_brir = self.read_signal(interferer_brir_fn)

            # Apply the BRIRs
            target_at_ear = self.apply_brir(target, target_brir)
            interferer_at_ear = self.apply_brir(interferer, interferer_brir)

            # Scale interferer to obtain SNR specified in scene description
            logging.info("Scaling interferer to obtain mixture SNR = %s dB.", snr_dB)

            if snr_ref is None:
                # snr_ref computed for first channel in the list and then
                # same scaling applied to all
                snr_ref = self.compute_snr(
                    target_at_ear,
                    interferer_at_ear,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                )
                logging.debug("Using channel %s as reference.", channel)

            # Apply snr_ref reference scaling to get 0 dB and then scale to target snr_dB
            interferer_at_ear = interferer_at_ear * snr_ref
            interferer_at_ear = interferer_at_ear * 10 ** ((-snr_dB) / 20)

            # Sum target and scaled and ramped interferer
            signal_at_ear = sum_signals([target_at_ear, interferer_at_ear])
            outputs.extend(
                [
                    (f"{prefix}_mixed_CH{channel}.wav", signal_at_ear),
                    (f"{prefix}_target_CH{channel}.wav", target_at_ear),
                    (f"{prefix}_interferer_CH{channel}.wav", interferer_at_ear),
                ]
            )

        if self.channels == []:
            target_brir_fn = f"{brir_stem}_t_CH0.wav"
            target_brir = self.read_signal(target_brir_fn)

        # Construct the anechoic target reference signal
        anechoic_brir_fn = (
            f"{anechoic_brir_stem}_t_CH1.wav"  # CH1 used for the anechoic signal
        )
        anechoic_brir = self.read_signal(anechoic_brir_fn)
        # Padding the anechoic brir very inefficient but keeps it simple
        anechoic_brir_pad = pad(anechoic_brir, len(target_brir))
        target_anechoic = self.apply_brir(target, anechoic_brir_pad)

        outputs.append((f"{prefix}_target_anechoic.wav", target_anechoic))

        # Write all output files
        for (filename, signal) in outputs:
            self.write_signal(filename, signal, self.fs)


def check_scene_exists(scene, output_path, num_channels):
    """Checks correct dataset directory for full set of pre-existing files.

    Args:
        scene (dict): dictionary defining the scene to be generated.

    Returns:
        status: boolean value indicating whether scene signals exist
            or do not exist.

    """
    channels = []
    if num_channels == 0:
        # This will only generate the initial target, masker and anechoic target signal
        pass
    else:
        # ... as above plus N hearing aid input channels plus 'channel 0' (the eardrum signal).
        # e.g. num_channel = 2  => channels [1, 2, 0]
        channels = list(range(1, num_channels + 1)) + [0]

    pattern = f"{output_path}/{scene['scene']}"
    files_to_check = [
        f"{pattern}_target.wav",
        f"{pattern}_target_anechoic.wav",
        f"{pattern}_interferer.wav",
    ]
    for ch in channels:
        files_to_check.extend(
            [
                f"{pattern}_mixed_CH{ch}.wav",
                f"{pattern}_interferer_CH{ch}.wav",
                f"{pattern}_target_CH{ch}.wav",
            ]
        )

    scene_exists = True
    for filename in files_to_check:
        scene_exists = scene_exists and os.path.exists(filename)
    return scene_exists
