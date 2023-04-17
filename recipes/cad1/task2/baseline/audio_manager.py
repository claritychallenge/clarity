"""A utility class for managing audio files."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from scipy.io import wavfile

logger = logging.getLogger(__name__)


class AudioManager:
    """A utility class for managing audio files."""

    def __init__(
        self,
        sample_rate: int = 44100,
        output_audio_path: str | Path = "",
        soft_clip: bool = False,
    ):
        """Initialize the AudioManager instance."""
        self.audios_to_save: dict[str, np.ndarray] = {}
        self.sample_rate = sample_rate
        self.soft_clip = soft_clip
        self.output_audio_path = Path(output_audio_path)
        self.output_audio_path.mkdir(exist_ok=True, parents=True)
        self.level_meter = pyln.Meter(self.sample_rate)

    def add_audios_to_save(self, file_name: str, waveform: np.ndarray) -> None:
        """Add a waveform to the list of audios to save.

        Args:
            file_name (str): The name of the track.
            waveform (np.ndarray): The track to save.
        """
        self.audios_to_save[file_name] = waveform.copy()

    def save_audios(self) -> None:
        """Save the audios to the given path.

        Args:
            output_audio_path (str): The path to save the audios to.
        """
        for file_name, waveform in self.audios_to_save.items():
            self._save_audio(file_name, waveform)

    def _save_audio(self, file_name: str, waveform: np.ndarray) -> None:
        """Save the audio to the given path.
        It always save in Int16 format.

        Args:
            file_name (str): The name of the track.
            waveform (np.ndarray): The track to save.
            output_audio_path (str): The path to save the audio to.
            sample_rate (int): The sample rate of the audio.
        """
        waveform = waveform.T if waveform.shape[0] == 2 else waveform

        n_clipped, waveform = self.clip_audio(waveform)
        if n_clipped > 0:
            logger.warning(
                f"Writing {self.output_audio_path / file_name}: {n_clipped} "
                "samples clipped"
            )

        waveform = (32768.0 * waveform).astype(np.int16)

        wavfile.write(
            self.output_audio_path / f"{file_name}.wav",
            self.sample_rate,
            waveform,
        )

    def clip_audio(
        self, signal: np.ndarray, min_val: float = -1, max_val: float = 1
    ) -> tuple[int, np.ndarray]:
        """Clip a WAV file to the given range.

        Args:
            signal (np.ndarray): The WAV file to clip.
            min_val (float): The minimum value to clip to. Defaults to -1.
            max_val (float): The maximum value to clip to. Defaults to 1.

        Returns:
            Tuple[int, np.ndarray]: Number of samples clipped and the clipped signal.
        """
        if self.soft_clip:
            signal = np.tanh(signal)
        n_clipped = np.sum(np.abs(signal) > 1.0)
        return int(n_clipped), np.clip(signal, min_val, max_val)

    def get_lufs_level(self, signal: np.ndarray) -> float:
        """Get the LUFS level of the signal.

        Args:
            signal (np.ndarray): The signal to get the LUFS level of.

        Returns:
            float: The LUFS level of the signal.
        """
        return self.level_meter.integrated_loudness(signal)

    def scale_to_lufs(self, signal: np.ndarray, target_lufs: float) -> np.ndarray:
        """Scale the signal to the given LUFS level.

        Args:
            signal (np.ndarray): The signal to scale.
            target_lufs (float): The target LUFS level.

        Returns:
            np.ndarray: The scaled signal.
        """
        current_lufs = self.get_lufs_level(signal)
        with warnings.catch_warnings(record=True):
            scaled_signal = pyln.normalize.loudness(signal, current_lufs, target_lufs).T
        return scaled_signal
