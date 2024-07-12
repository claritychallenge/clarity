# The Second Cadenza Challenge (CAD2) - Task 2: Rebalancing Classical Music

Cadenza challenge code for the Second Cadenza Challenge (CAD2) Task2.

For more information please visit the [challenge website](https://cadenzachallenge.org/docs/cadenza2/intro).

## 1. Data structure

The Second Cadenza Challenge - task 2 is using two datasets for training:

- EnsembleSet: This contains 80 pieces of synthesised classical music.
- CadenzaWoodwind: This contains 19 pieces of woodwind quartets.

To download the data, please visit [here](https://forms.gle/taYK6MfBeW9sQk5PA).
The data is contained in several packages:

- CadenzaWoodwind.zip: containing the CadenzaWoodwind dataset.
- EnsembleSet_Mix_1.zip: containing the EnsembleSet dataset for microphone `MIX_1`.
- metadata.zip: containing the metadata for the systems.

### 1.1 Generate data directory

The baseline system expects the data in a specific directory structure.
To generate this structure:

1. Save the packages under the same directory.
2. Run the scrip `process_dataset/process_zenodo_download.py`
3. in config.yaml, set:
   4. `path.zenodo_download_path` to the path where you saved the packages.
   5. `path.root` to the path where you want to save the data.

Note that the script will unzip the packages if they are not already unzipped.

### 1.2 Necessary data

The script `process_dataset/process_zenodo_download.py` will generate the following structure:

- **audio** directory containing the audio files.

```text
cadenza_data
└───cad2
    └───task2
        └───audio
            ├───train
            └───valid
```

- **Metadata** contains the metadata for the systems.

```text
cadenza_data
└───cad2
    └───task1
        └───metadata
            ├───compressor_params.train.json
            ├───compressor_params.valid.json
            ├───gains.json
            ├───gains_meta.json
            ├───listeners.train.json
            ├───listeners.valid.json
            ├───music.train.json
            ├───music.valid.json
            ├───music.valid.to_generate.json
            ├───music_tracks.train.json
            ├───music_tracks.valid.json
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

The enhancement system uses an audio source separation model to separate the different
sources of the mixture. What instruments are contained in the mixture is given information.

Next, using the requested gains, it changes the levels of the different sources and
downmix to stereo. Then, uses a multiband dynamic range compressor to compress the audio signal.
The final signals is saved in FLAC, 44100 Hz, 16-bit format.

The audio source separation model correspond to 8 ConvTasNet models trained on EnsembleSet and CadenzaWoodwind.
Each model is trained to separate a single target instrument from the music.

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

A csv file containing the eight HAAQI scores and the combined score will be generated in the `path.exp_folder`.

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

## 3. Results

| System    | HAAQI  |
|:----------|:------:|
| Causal    |   -    |
| NonCausal | 0.4594 |
