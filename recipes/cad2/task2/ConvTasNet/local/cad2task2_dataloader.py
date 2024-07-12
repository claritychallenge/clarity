from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils import data


class Compose:
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio


def augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain to each source between `low` and `high`"""
    if torch.FloatTensor(1).uniform_() < 0.5:
        gain_1 = low + torch.rand(1) * (high - low)
        gain_2 = low + torch.rand(1) * (high - low)
        return torch.stack([audio[0] * gain_1, audio[1] * gain_2])

    return audio


def augment_channelswap(audio):
    """Randomly swap channels of stereo sources"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])

    return audio


def get_audio_durations(track_path: Path | str) -> float:
    if isinstance(track_path, str):
        track_path = Path(track_path)
    return librosa.get_duration(path=track_path.as_posix())


class RebalanceMusicDataset(data.Dataset):
    """
    Dataset to process EnsembleSet and CadenzaWoodwind datasets for CAD2 Task2 baseline
    The dataset is composed of a target source and a random number of
    accompaniment sources.

    Args:
        root_path (str): Path to the root directory of the dataset
        music_tracks_file (str): Path to the json file containing the music tracks
        target (str): Target source to be extracted
        samples_per_track (int): Number of samples to extract from each track
        segment_length (float): Length of the segment to extract
        random_segments (bool): If True, extract random segments from the tracks
        random_track_mix (bool): If True, mix random accompaniment tracks
        split (str): Split of the dataset to use
        sample_rate (int): Sample rate of the audio files

    """

    dataset_name = "EnsembleSet & CadenzaWoodwind"

    def __init__(
        self,
        root_path: Path | str,
        music_tracks_file: Path | str,
        target: str,
        samples_per_track: int = 1,
        segment_length: float | None = 5.0,
        random_segments=False,
        random_track_mix=False,
        split: str = "train",
        source_augmentations=lambda audio: audio,
        sample_rate: int = 44100,
    ):
        self.instruments = [
            "Bassoon",
            "Cello",
            "Clarinet",
            "Flute",
            "Oboe",
            "Sax",
            "Viola",
            "Violin",
        ]
        self.repeated_instruments: list[str] = []  # ["Violin", "Viola", "Flute", "Sax"]

        if isinstance(root_path, str):
            root_path = Path(root_path)

        self.root_path: Path = root_path
        if split == "train":
            self.root_path = root_path / "train"

        self.music_tracks_file = Path(music_tracks_file)

        self.target = target

        self.samples_per_track = samples_per_track
        self.segment_length = segment_length
        self.random_segments = random_segments
        self.random_track_mix = random_track_mix
        self.split = split
        self.sample_rate = sample_rate
        self.source_augmentations = source_augmentations

        with open(self.music_tracks_file) as f:
            self.tracks = json.load(f)
        self.tracks_list = list(self.tracks.keys())

        self.target_tracks = []
        for k, v in self.tracks.items():
            for type_source, source in v.items():
                if type_source == "mixture":
                    continue

                if self.target in source["instrument"]:
                    self.target_tracks.append(k)

        if self.split == "train":
            self.min_src = 1
            self.max_src = 3

            self.accompaniment_tracks : dict = {
                i: [] for i in self.instruments if i != self.target
            }
            for k, v in self.tracks.items():
                for type_source, source in v.items():
                    if type_source == "mixture":
                        continue

                    if self.target not in source["instrument"]:
                        self.accompaniment_tracks[
                            source["instrument"].split("_")[0]
                        ].append(k)

    def __len__(self):
        return len(self.target_tracks) * self.samples_per_track

    def __getitem__(self, idx):
        # assemble the mixture of target and interferers
        audio_sources = {}
        track_idx = idx // self.samples_per_track

        # Load the target source
        target_name = self.target_tracks[track_idx]

        if self.split == "test":
            self.track_name = target_name

        accompaniment_tracks = []
        target_track = None
        for type_source, source in self.tracks[target_name].items():
            if type_source == "mixture":
                continue

            if self.target in source["instrument"]:
                target_track = source
            else:
                accompaniment_tracks.append(source)

        segment_length = self.segment_length
        if self.segment_length is None:
            segment_length = target_track["duration"]

        target_duration = target_track["duration"]
        start = target_track["start"]

        if self.random_segments:
            start = np.floor(
                np.random.uniform(0, int(target_duration) - segment_length)
            )

        segment_length = self.segment_length
        if self.segment_length is None:
            segment_length = target_track["duration"]

        start_sample = int(start * self.sample_rate)
        stop_sample = int((start + segment_length) * self.sample_rate)

        target_signal, _ = sf.read(
            self.root_path / target_track["track"],
            always_2d=True,
            start=start_sample,
            stop=stop_sample,
        )

        # convert to torch tensor
        target_signal = torch.tensor(target_signal.T, dtype=torch.float)
        # target_signal = F.pad(target_signal, (target_len - target_signal.size(1), 0))
        target_signal = self.source_augmentations(target_signal)
        target_len = int(segment_length * self.sample_rate)
        audio_sources["target"] = torch.zeros(2, target_len)
        audio_sources["target"] = target_signal
        audio_sources["accompaniment"] = torch.zeros_like(target_signal)

        # ***************************************
        # Load accompaniments
        # If random mixture, change `accompaniment_tracks` variable with
        # random instruments and tracks

        if self.random_track_mix:
            num_accomp = np.random.randint(self.min_src, self.max_src + 1)
            accompaniment_instruments = np.random.choice(
                [
                    x
                    for x in self.instruments + self.repeated_instruments
                    if self.target not in x
                ],
                num_accomp,
                replace=False,
            )
            accompaniment_tracks = []
            for accomp_instrument in accompaniment_instruments:
                a_t = np.random.choice(self.accompaniment_tracks[accomp_instrument])
                for item in self.tracks[a_t].values():
                    if accomp_instrument in item["instrument"]:
                        accompaniment_tracks.append(item)

        # Load accompaniment files
        for accomp_track in accompaniment_tracks:
            accomp_duration = accomp_track["duration"]

            # new random segment
            start = accomp_track["start"]
            if self.random_segments:
                start = np.floor(np.random.uniform(0, accomp_duration - segment_length))

            start_sample = int(start * self.sample_rate)
            stop_sample = int((start + segment_length) * self.sample_rate)

            try:
                accomp_signal, _ = sf.read(
                    self.root_path / accomp_track["track"],
                    always_2d=True,
                    start=start_sample,
                    stop=stop_sample,
                )

                accomp_signal = torch.tensor(accomp_signal.T, dtype=torch.float)

                accomp_signal = self.source_augmentations(accomp_signal)

                audio_sources["accompaniment"] += accomp_signal
            except Exception as err:
                print(accomp_track, flush=True)
                print(f"start {start}", flush=True)
                print(f"len {segment_length}", flush=True)
                print(err)

        # prepare mixture
        audio_mix = torch.stack(list(audio_sources.values())).sum(0)

        audio_sources = torch.stack(
            [audio_sources["target"], audio_sources["accompaniment"]],
            dim=0,
        )
        return audio_mix, audio_sources

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "separation"
        infos["licenses"] = [ensembleset_license, woodwing_licence]
        return infos


ensembleset_license = dict(
    title="EnsembleSet",
    title_link="https://zenodo.org/records/6519024",
    author="Saurjya Sarkar, Emmanouil Benetos, Mark Sandler",
    licence="CC BY-NC 4.0",
    licence_link="https://creativecommons.org/licenses/by-nc/4.0/",
    non_commercial=False,
)

woodwing_licence = dict(
    title="CadenzaWoodwind",
    title_link="",
    author="Alex Miller, Trevor Cox, Gerardo Roa Dabike",
    licence="CC NC 4.0",
    licence_link="https://creativecommons.org/licenses/by/4.0/",
    non_commercial=False,
)


if __name__ == "__main__":
    _root_path = Path(
        "/media/gerardoroadabike/Extreme"
        " SSD1/Challenges/CAD2/cadenza_data/cad2/task2/audio"
    )
    _music_tracks_file = Path(
        "/media/gerardoroadabike/Extreme"
        " SSD1/Challenges/CAD2/cadenza_data/cad2/task2/metadata/music.valid.json"
    )
    for target_source in [
        "Bassoon",
        # "Cello",
        # "Clarinet",
        # "Flute",
        # "Oboe",
        # "Sax",
        # "Viola",
        # "Violin",
    ]:
        dataset = RebalanceMusicDataset(
            _root_path,
            _music_tracks_file,
            target_source,
            split="valid",
            random_track_mix=False,
            random_segments=False,
            segment_length=None,
        )
        for x, y in dataset:
            print(x.shape, y.shape)
