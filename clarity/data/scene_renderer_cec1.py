"""Scene rendering for CEC1 challenge."""
import logging
import math
from pathlib import Path

import numpy as np
from scipy.signal import convolve

from clarity.data.utils import better_ear_speechweighted_snr, pad, sum_signals
from clarity.utils.file_io import read_signal, write_signal


class Renderer:
    """
    SceneGenerator of CEC1 training and development sets. The render() function
    generates all simulated signals for each scene given the parameters specified in the
    metadata/scenes.train.json or metadata/scenes.dev.json file.
    """

    def __init__(
        self,
        input_path,
        output_path,
        num_channels=1,
        sample_rate=44100,
        ramp_duration=0.5,
        tail_duration=0.2,
        pre_duration=2.0,
        post_duration=1.0,
        test_nbits=16,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.ramp_duration = ramp_duration
        self.n_tail = int(tail_duration * sample_rate)
        self.pre_duration = pre_duration
        self.post_duration = post_duration
        self.test_nbits = test_nbits

        if num_channels == 0:
            # Generate just the initial target, masker and anechoic target signal
            self.channels = []
        else:
            # ... as above plus N hearing aid input channels plus 'channel 0'
            # (the eardrum signal). e.g. num_channel = 2  => channels [1, 2, 0]
            self.channels = list(range(1, num_channels + 1)) + [0]

    def apply_ramp(self, signal, ramp_duration):
        """Apply half cosine ramp into and out of signal.

        Args:
            signal (np.ndarray): signal to be ramped.
            ramp_duration (int): ramp duration in seconds.

        Returns:
            np.ndarray: Signal ramped into and out of by cosine function.
        """
        ramp = np.cos(
            np.linspace(math.pi, 2 * math.pi, int(self.sample_rate * ramp_duration))
        )
        ramp = (ramp + 1) / 2
        signal_ramped = np.array(signal)
        signal_ramped[0 : len(ramp)] *= ramp
        signal_ramped[-len(ramp) :] *= ramp[::-1]
        return signal_ramped

    def apply_brir(self, signal, brir):
        """Convolve a signal with a binaural room impulse response (BRIR).

        Args:
            signal (ndarray): The mono or stereo signal stored as array of floats
            brir (ndarray): The BRIR stored a 2xN array of floats
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

    def compute_snr(
        self, target: np.ndarray, noise: np.ndarray, pre_samples=0, post_samples=-1
    ):
        """Return the Signal Noise Ratio (SNR).

        Take the overlapping segment of the noise and get the speech-weighted
        better ear Signal Noise Ratio. (Note, SNR is a ratio -- not in dB.)

        Args:
            target (np.ndarray): Target signal.
            noise (np.ndarray): Noise (should be same length as target)

        Returns:
            float: signal_noise_ratio for better ear.
        """

        pre_samples = int(self.sample_rate * self.pre_duration)
        post_samples = int(self.sample_rate * self.post_duration)

        segment_target = target[pre_samples:-post_samples]
        segment_noise = noise[pre_samples:-post_samples]
        try:
            assert len(segment_target) == len(segment_noise)
        except AssertionError as e:
            raise ValueError(
                f"Target ({len(segment_target)}) "
                f"differs in length from Noise ({len(segment_noise)})"
            ) from e

        snr = better_ear_speechweighted_snr(segment_target, segment_noise)
        return snr

    def render(
        self,
        target_id: str,
        noise_type: str,
        interferer_id: str,
        room: str,
        scene: str,
        offset,
        snr_dB: int,
        dataset,
        pre_samples=88200,
        post_samples=44100,
    ):
        brir_stem = f"{self.input_path}/{dataset}/rooms/brir/brir_{room}"
        anechoic_brir_stem = f"{self.input_path}/{dataset}/rooms/brir/anech_brir_{room}"
        target_fn = f"{self.input_path}/{dataset}/targets/{target_id}.wav"
        interferer_fn = (
            f"{self.input_path}/{dataset}/interferers/{noise_type}/{interferer_id}.wav"
        )

        target = read_signal(
            target_fn, sample_rate=self.sample_rate, allow_resample=False
        )
        target = np.pad(target, [(pre_samples, post_samples)])

        interferer_signal = read_signal(
            interferer_fn,
            sample_rate=self.sample_rate,
            offset=offset,
            n_samples=len(target),
            offset_is_samples=True,
            allow_resample=False,
        )

        if len(interferer_signal) != len(target):
            logging.error("Interferer signal too short")
            raise ValueError(f"Interferer signal too short: {interferer_fn}")

        # Apply 500ms half-cosine ramp
        interferer_signal = self.apply_ramp(
            interferer_signal, ramp_duration=self.ramp_duration
        )

        prefix = f"{self.output_path}/{scene}"
        outputs = [
            (f"{prefix}_target.wav", target),
            (f"{prefix}_interferer.wav", interferer_signal),
        ]

        snr_ref = None
        for channel in self.channels:
            # Load scene BRIRs
            target_brir_fn = f"{brir_stem}_t_CH{channel}.wav"
            interferer_brir_fn = f"{brir_stem}_i1_CH{channel}.wav"
            target_brir = read_signal(
                target_brir_fn, sample_rate=self.sample_rate, allow_resample=False
            )
            interferer_brir = read_signal(
                interferer_brir_fn, sample_rate=self.sample_rate, allow_resample=False
            )

            # Apply the BRIRs
            target_at_ear = self.apply_brir(target, target_brir)
            interferer_at_ear = self.apply_brir(interferer_signal, interferer_brir)

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

            # Apply snr_ref reference scaling to get 0 dB, then scale to target snr_dB
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
            target_brir = read_signal(
                target_brir_fn, sample_rate=self.sample_rate, allow_resample=False
            )

        # Construct the anechoic target reference signal
        anechoic_brir_fn = (
            f"{anechoic_brir_stem}_t_CH1.wav"  # CH1 used for the anechoic signal
        )
        anechoic_brir = read_signal(
            anechoic_brir_fn, sample_rate=self.sample_rate, allow_resample=False
        )
        # Padding the anechoic brir very inefficient but keeps it simple
        anechoic_brir_pad = pad(anechoic_brir, len(target_brir))
        target_anechoic = self.apply_brir(target, anechoic_brir_pad)

        outputs.append((f"{prefix}_target_anechoic.wav", target_anechoic))

        # Write all output files
        for filename, signal in outputs:
            write_signal(filename, signal, self.sample_rate, strict=True)


def check_scene_exists(scene: dict, output_path: str, num_channels: int) -> bool:
    """Checks correct dataset directory for full set of pre-existing files.

    Args:
        scene (dict): dictionary defining the scene to be generated.
        output_path (str): Path files should be saved to.
        num_channels (int): Number of channels

    Returns:
        status: boolean value indicating whether scene signals exist
            or do not exist.

    """
    channels = []
    if num_channels == 0:
        # This will only generate the initial target, masker and anechoic target signal
        pass
    else:
        # ... as above plus N hearing aid input channels plus 'channel 0' (the eardrum
        # signal), e.g., num_channel = 2  => channels [1, 2, 0]
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

    # Return True if all files exist, False otherwise
    return all(Path(filename).exists() for filename in files_to_check)
