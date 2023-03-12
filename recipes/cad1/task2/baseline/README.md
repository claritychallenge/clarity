# The First Cadenza Challenge (CAD1) - Task 2: Listening music in a car

Cadenza challenge code for the First Cadenza Challenge (CAD1) Task2.

For more information please visit the [challenge website](https://cadenzachallenge.org/docs/cadenza1/cc1_intro).

## 1. Data structure

The dataset for the First Cadenza Challenge - task 2 is based on the small subset of FMA (FMA-small) and
the MTG-Jamendo dataset.

The dataset is split into 3 subsets: `train`, `valid` and `test`, totalling 10,000 songs with a distribution of 80% / 10% / 10%.

To download the data, please visit [here](https://forms.gle/9L5ncYKe2YhD5c828).

[//]: # (The data is split into `cadenza_cad1_task1_core_musdb18hq.tar.gz` &#40;containing the MUSDB18-HQ dataset&#41; and)

[//]: # (`cadenza_cad1_task1_core_metadata.tar.gz` &#40;containing the list of songs and listeners' characteristics per split&#41;.)

[//]: # (Alternatively, you can download the MUSDB18-HQ dataset from the official [SigSep website]&#40;https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav&#41;.)

[//]: # (If you opt for this alternative, be sure to download the uncompressed wav version. Note that you will need both packages to run the baseline system.)

[//]: # (If you need additional music data for training your model, please restrict to the use of [MedleyDB]&#40;https://medleydb.weebly.com/&#41; [4][5],)

[//]: # ([BACH10]&#40;https://labsites.rochester.edu/air/resource.html&#41; [6] and [FMA-small]&#40;https://github.com/mdeff/fma&#41; [7].)

[//]: # (Theses are shared as `cadenza_cad1_task1_augmentation_medleydb.tar.gz`, `cadenza_cad1_task1_augmentation_bach10.tar.gz`)

[//]: # (and `cadenza_cad1_task1_augmentation_fma_small.tar.gz`.)

[//]: # (**Keeping the augmentation data restricted to these datasets will ensure that the evaluation is fair for all participants**.)

Unpack packages under the same root directory using


```bash
tar -xvzf <PACKAGE_NAME>
```

### 1.1 Mandatory data

* **Music** contains the MUSDB18-HQ music dataset for training, validation and evaluation.

```text
cadenza_data
└───task1
    └───audio
        └───musdb18hq
            ├───train
            └───test
```

* **Metadata** contains the metadata for the systems.

```text
cadenza_data
└───task1
    └───metadata
        └───musdb18hq
            ├───listeners.train.json
            ├───listeners.valid.json
            ├───musdb18.train.json
            ├───musdb18.valid.json
            └───musdb18.test.json
```

### 1.2 Demo data

To help you to start with the challenge, we provide a small subset of the data.
The `demo_data` folder contains a single song and two listeners from the validation set.

To use the demo data, simply download the package `cadenza_data_demo.tar.xz`
from [here](https://drive.google.com/drive/folders/1Yxo_R-yPByEUvX5O5lhsHk3tW1ek5qKW?usp=share_link)
and unpack it under `recipes/cad1/task1/`, i.e., one level above the baseline directory.
Note that the `root.path` variable in `config.yaml` is already set to the demo data by default.

To unpack the demo data, run:

```bash
tar -xvf cadenza_data_demo.tar.xz
```


## 2. Baseline

In the `baseline/` folder, we provide code for running the baseline enhancement system and performing
the objective evaluation. Note that we use [hydra](https://hydra.cc/docs/intro/) for config handling.

### 2.1 Enhancement

The objective of the enhancement stage is takes a song and optimise it to a listener hearing characteristics
knowing metadata information about the car noise scenario (tou won't have access to noise signal), head
rotation of the listener and the SNR of the enhanced music and the noise at the hearing aid microphones.

In the baseline, we simply attenuate the songs in 5 dB LUFS and save it in 16-bit PCM WAV format. The
level was selected so fewer samples from the hearing aid output signal are clipped.

To run the baseline enhancement system first, make sure that `paths.root` in `config.yaml` points to
where you have installed the Cadenza data. This parameter defaults to the working directory.
You can also define your own `path.exp_folder` to store enhanced
signals and evaluated results.

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

The `evaluate.py` module takes the enhanced signals and adds the room impulses and the car noise using
the expected SNR. It then pass that signal through a fixed hearing aid. The hearing aid output and
the reference song (scaled to the hearing aid output LUFS) are used to compute the HAAQI score.

To run the evaluation stage, make sure that `path.root` is set in the `config.yaml` file and then run

```bash
python evaluate.py
```

A csv file containing the HAAQI scores for each ear and the average of both will be generated in
the `path.exp_folder`.

To check the HAAQI code, see [here](../../../../clarity/evaluator/haaqi).

Please note: you will not get identical HAAQI scores for the same signals if the random seed is not defined
(in the given recipe, the random seed for each signal is set as the last eight digits of the song md5).
As there are random noises generated within HAAQI, but the differences should be sufficiently small.

**Baseline performance on the validation set will be updated soon.**

## References

* [2] Byrne, Denis, and Harvey Dillon. "The National Acoustic Laboratories'(NAL) new procedure for selecting the gain and frequency response of a hearing aid." Ear and hearing 7.4 (1986): 257-265. [doi:10.1097/00003446-198608000-00007](https://doi.org/10.1097/00003446-198608000-00007)
* [3] Kates J M, Arehart K H. "The Hearing-Aid Audio Quality Index (HAAQI)". IEEE/ACM transactions on audio, speech, and language processing, 24(2), 354–365. [doi:10.1109/TASLP.2015.2507858](https://doi.org/10.1109%2FTASLP.2015.2507858)
* [7] Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2016). "FMA: A dataset for music analysis". arXiv preprint arXiv:1612.01840. [doi:10.48550/arXiv.1612.01840](https://doi.org/10.48550/arXiv.1612.01840)