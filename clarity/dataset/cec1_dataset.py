import json
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.signal import firwin, lfilter
from soundfile import read
from torch.utils import data

logger = logging.getLogger(__name__)


def read_wavfile(path):
    wav, _ = read(path)
    return wav.transpose()


class CEC1Dataset(data.Dataset):
    def __init__(
        self,
        scenes_folder,
        scenes_file,
        sample_rate,
        downsample_factor,
        wav_sample_len=None,
        wav_silence_len=2,
        num_channels=6,
        norm=False,
        testing=False,
    ):
        self.scenes_folder = scenes_folder
        self.sample_rate = sample_rate
        self.downsample_factor = downsample_factor
        self.wav_sample_len = wav_sample_len
        self.wav_silence_len = wav_silence_len
        self.num_channels = num_channels
        self.norm = norm
        self.testing = testing

        self.scene_list = []
        with open(scenes_file, encoding="utf-8") as fp:
            scene_json = json.load(fp)
            if not testing:
                for scene in scene_json:
                    self.scene_list.append(scene["scene"])
            else:
                for scene in scene_json.keys():
                    self.scene_list.append(scene)

        if self.num_channels == 2:
            self.mixed_suffix = "_mixed_CH1.wav"
            self.target_suffix = "_target_anechoic.wav"
        elif self.num_channels == 6:
            self.mixed_suffix = ["_mixed_CH1.wav", "_mixed_CH2.wav", "_mixed_CH3.wav"]
            self.target_suffix = "_target_anechoic.wav"
        else:
            raise NotImplementedError

        self.lowpass_filter = firwin(
            1025,
            self.sample_rate // (2 * self.downsample_factor),
            pass_zero="lowpass",
            fs=self.sample_rate,
        )

    def wav_sample(self, x, y):
        """
        A 2 second silence is in the beginning of clarity data
        Get rid of the silence segment in the beginning & sample a
        constant wav length for training.
        """
        silence_len = int(self.wav_silence_len * self.sample_rate)
        x = x[:, silence_len:]
        y = y[:, silence_len:]

        wav_len = x.shape[1]
        sample_len = int(self.wav_sample_len * self.sample_rate)
        if wav_len > sample_len:
            start = np.random.randint(wav_len - sample_len)
            end = start + sample_len
            x = x[:, start:end]
            y = y[:, start:end]
        elif wav_len < sample_len:
            x = np.append(
                x, np.zeros([x.shape[1], sample_len - wav_len], dtype=np.float32)
            )
            y = np.append(
                y, np.zeros([x.shape[1], sample_len - wav_len], dtype=np.float32)
            )

        return x, y

    def lowpass_filtering(self, x):
        return lfilter(self.lowpass_filter, 1, x)

    def __getitem__(self, item):
        scenes_folder = Path(self.scenes_folder)
        if self.num_channels == 2:
            mixed = read_wavfile(
                scenes_folder / (self.scene_list[item] + self.mixed_suffix)
            )
        elif self.num_channels == 6:
            mixed = []
            for suffix in self.mixed_suffix:
                mixed.append(
                    read_wavfile(scenes_folder / (self.scene_list[item] + suffix))
                )
            mixed = np.concatenate(mixed, axis=0)
        else:
            raise NotImplementedError
        target = None
        if not self.testing:
            target = read_wavfile(
                scenes_folder / (self.scene_list[item] + self.target_suffix)
            )
            if target.shape[1] > mixed.shape[1]:
                logging.warning(
                    "Target length is longer than mixed length. Truncating target."
                )
                target = target[:, : mixed.shape[1]]
            elif target.shape[1] < mixed.shape[1]:
                logging.warning(
                    "Target length is shorter than mixed length. Padding target."
                )
                target = np.pad(
                    target,
                    ((0, 0), (0, mixed.shape[1] - target.shape[1])),
                    mode="constant",
                )

        if self.sample_rate != 44100:
            mixed_resampled, target_resampled = [], []
            for i in range(mixed.shape[0]):
                mixed_resampled.append(
                    librosa.resample(
                        mixed[i], target_sr=44100, orig_sr=self.sample_rate
                    )
                )
            mixed = np.array(mixed_resampled)
            if target is not None:
                for i in range(target.shape[0]):
                    target_resampled.append(
                        librosa.resample(
                            target[i], target_sr=44100, orig_sr=self.sample_rate
                        )
                    )
                target = np.array(target_resampled)

        if self.wav_sample_len is not None:
            mixed, target = self.wav_sample(mixed, target)

        if self.norm:
            mixed_max = np.max(np.abs(mixed))
            mixed = mixed / mixed_max
            if target is not None:
                target = target / mixed_max

        if not self.testing:
            return_data = (
                torch.tensor(mixed, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
            )
        else:
            return_data = (
                torch.tensor(mixed, dtype=torch.float32),
                self.scene_list[item],
            )

        return return_data

    def __len__(self):
        return len(self.scene_list)
