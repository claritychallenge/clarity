# The Second Cadenza Challenge (CAD2) - Task 1: Lyrics Intelligibility

Cadenza code for the Second Cadenza Challenge (CAD2) Task1.

For more information please visit the [challenge website](https://cadenzachallenge.org/docs/cadenza2/intro).

## 1. Data structure

The Second Cadenza Challenge - task 1 is using the transcribed version of the MUSDB18-HQ dataset.
This extension comprises 96 manual transcriptions of English songs by
non-native English speakers, following the same split for training, validation and test as museval,
totalling 366 minutes of audio.

To download the data, please visit [here](https://forms.gle/BzGrtdzqLvdjH6ja8).
The data is contained in the package `cadenza_cad2_task1_train.v1_0.tar.gz`.
The package contains the transcribe version of the MUSDB18-HQ dataset in FLAC format
and the metadata for the systems.

Unpack packages under the same root directory using

```bash
tar -xvzf <PACKAGE_NAME>
```

### 1.1 Necessary data

- **Music** contains the MUSDB18-HQ music dataset for training, validation and evaluation.

```text
cadenza_data
└───cad2
    └───task1
        └───audio
            └───musdb18hq
                └───train
                     ├───audio
                     └───lyrics
```

- **Metadata** contains the metadata for the systems.

```text
cadenza_data
└───cad2
    └───task1
        └───metadata
            ├───alpha.json
            ├───compressor_params.train.json
            ├───compressor_params.valid.json
            ├───listeners.train.json
            ├───listeners.valid.json
            ├───music.train.json
            ├───music.valid.json
            ├───scene.train.json
            ├───scene.valid.json
            ├───scene_listeners.train.json
            └───scene_listeners.valid.json
```

## 2. Baseline

In the `baseline/` folder, we provide code for running the baseline enhancement system
and performing the objective evaluation.
Note that we use [hydra](https://hydra.cc/docs/intro/) for config handling.

### 2.1 Enhancement

The enhancement system uses an audio source separation model to separate the vocals from the music.
Next, using the alpha parameter, it changes the levels of the vocals and music to
simulate different mixing scenarios. Then, uses a multiband dynamic range compressor to
amplify the audio signal. The final signals is saved in FLAC, 44100 Hz, 16-bit format.

The audio source separation model correspond to a ConvTasNet model trained on the MUSDB18-HQ dataset
to separate the vocals from the background.

To run the baseline enhancement system first, make sure that `paths.root` in `config.yaml` points to
where you have installed the Cadenza data. You can also define your own `path.exp_folder`
to store the enhanced signals and evaluated results.

You can modify the code to add your own enhancement system or your own remixing strategy
using the alpha parameter.

Then run:

```bash
python enhance.py
```

Alternatively, you can provide the root variable on the command line, e.g.,

```bash
python enhance.py path.root=/full/path/to/my/cadenza_data
```

To get a full list of the parameters, run:

```bash
python enhance.py --help
```

The folder `enhanced_signals` will appear in the `exp` folder.

### 2.2 Evaluation

A csv file containing the HAAQI scores and Whisper correct transcribed words will be generated in the `path.exp_folder`.

To check the HAAQI code, see [here](../../../../clarity/evaluator/haaqi).

Please note: you will not get identical HAAQI scores for the same signals if the random seed is not defined.
This is due to the  random noises generated within HAAQI, but the differences should be sufficiently small.
For reproducibility, in the given recipe, the random seed for each signal is set as the last eight digits
of the song md5.

### 2.3 ConvTasNet Source Separation

In directory `ConvTasNet/` we provide the code to train the ConvTasNet model
on the MUSDB18-HQ dataset.

This recipe is based on the stereo modification of the ConvTasNet done by Alexandre Defossez
and available in Demucs' branch [`v1`](https://github.com/facebookresearch/demucs/blob/110f8fee0815d4c0d4ed3e2d478e37df247cd269/demucs/tasnet.py)

The training is based in an Asteroid recipe.
