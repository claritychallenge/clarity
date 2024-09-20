# Generate music dataset for the ICASSP 2024 Cadenza Challenge

The ICASSP 2024 Cadenza Challenge music dataset is based on the MUSDB18-HQ dataset.

Steps:

1. Download `cadenza_cad1_task1_core_musdb18hq.tar.gz` and `cadenza_cad1_task1_core_metadata.tar.gz`
packages from the [Cadenza Challenge website](https://cadenza-challenge.github.io/).
2. Unpack packages under the same root directory.
3. Run the script

## Unpack the data

To unpack the data run:

```bash
tar -xvzf <PACKAGE_NAME>
```

## Generate the dataset

To generate the dataset, set the `path.root` parameter in the `generate_dataset/config.yaml`
to where you unpacked the data. Then run:

```bash
python generates_at_mic_musdb18.py
```

or, run the script with the `path.root` parameter:

```bash
python generates_at_mic_musdb18.py path.root <PATH_TO_UNPACKED_DATA>
```

The script will generate the dataset in the `path.root` directory.

The script should create the `at_mic_microphone` where all music samples
picked up by the microphones (at the mic) are saved.

In the next example, `A Classic Education - NightOwl-hp_0103` corresponds to the
song `A Classic Education - NightOwl` with the `hp_0103` head position.

```text
cadenza_data
├───audio
|   ├───at_mic_music
|   |   └───train (80.8 GB)
|   |       ├───A Classic Education - NightOwl-hp_0103
|   |       |   |  bass.wav
|   |       |   |  drums.wav
|   |       |   |  other.wav
|   |       |   |  vocals.wav
|   |       |   |  mixture.wav
|   |       |
|   |       ├───A Classic Education - NightOwl-hp_0138
|   |       |   ....
|   |
|   ├───hrtf (336 kB)
|   |
|   └───music
|       └───train (20.2 GB)
|
└───metadata  (328 kB)
|  gains.json
    |  at_mic_music.train.json
    |  ...
```
