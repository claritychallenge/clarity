# The First Cadenza Challenge (CAD1) - Task 2: Listening music in a car

Cadenza challenge code for the First Cadenza Challenge (CAD1) Task2.

For more information please visit the [challenge website](https://cadenzachallenge.org/docs/cadenza1/cc1_intro).

## 1. Data structure

### 1.1 Obtaining the CAD1 - Task2 data

The music dataset for the First Cadenza Challenge - Task 2 is based on the small subset of the FMA [2] dataset
(FMA-small) and the MTG-Jamendo dataset [4]. The dataset contains 1000 samples from seven musical genres,
totalling 7000 songs with a distribution of 80% / 10% / 10% for `train`, `valid` and `test`.

From FMA small:

* Hip-Hop
* Instrumental
* International
* Pop
* Rock

From MTG-Jamendo:

* Classical
* Orchestral

The HRTFs data is based on the eBrIRD - ELOSPHERES binaural room impulse response database.

To download the data, please visit [here](https://forms.gle/9L5ncYKe2YhD5c828).

The data will download into a package file called `cadenza_cad1_task2_core.v1_1.tar.gz`.

Unpack this package using

```bash
tar -xvzf cadenza_cad1_task2_core.v1_1.tar.gz
```

Once unpacked the directory structure will be as follows

**cadenza_cad1_task2_core.v1.0** contains the training and validation data:

```bash
clarity_CPC2_data
└── cadenza_data
   └── cad1  # The hearing aid output signals
        └── taks2
            ├── audio
            |   ├── eBrird  # HRTFs directory
            |   └── music
            |       ├── training
            |       └── validation
            ├── metadata  # Metadata
            └── manifest  # Lists the package contents
```

### 1.2 Demo data

To help you to start with the challenge, we provide a small subset of the data.
The `demo_data` folder contains a single song and two listeners from the validation set.

To use the demo data, simply download the package `cadenza_task2_data_demo.tar.tar.xz`
from [here](https://drive.google.com/drive/folders/1On5Bv7Sd6zLZWfA76jdkM-FmGS61Mbi-?usp=share_link)
and unpack it under `recipes/cad1/task2/`, i.e., one level above the baseline directory.
Note that the `root.path` variable in `config.yaml` is already set to the demo data by default.

To unpack the demo data, run:

```bash
tar -xvf cadenza_data_demo.tar.xz
```

## 2. Baseline

In the `baseline/` folder, we provide code for running the baseline enhancement system and performing
the objective evaluation. Note that we use [hydra](https://hydra.cc/docs/intro/) for config handling.

The baseline uses librosa to read the MP3 audio files. Librosa will raise error is libsoundfile and ffmpeg are not installed.
If you have an Anaconda or Miniconda environment, you can install them as:

* conda install -c conda-forge ffmpeg
* conda install -c conda-forge libsndfile

```bash

### 2.1 Enhancement

The objective of the enhancement stage is takes a song and optimise it to a listener hearing characteristics
knowing metadata information about the car noise scenario (you won't have access to noise signal), head
rotation of the listener and the SNR of the enhanced music and the noise at the hearing aid microphones.

In the baseline, we simply attenuate the song according to the average hearing loss and save it in 16-bit PCM WAV format.
This attenuation prevents some clipping in the hearing aid output signal.

To run the baseline enhancement system first, make sure that `paths.root` in `config.yaml` points to
where you have installed the Cadenza data foer the task2. This parameter defaults to one level above the recipe
for the demo data. You can also define your own `path.exp_folder` to store enhanced and evaluated signal results.

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
the expected SNR. It then passes that signal through a fixed hearing aid. The hearing aid output and
the reference song are used to compute the HAAQI [2] score.

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

The overall HAAQI score for baseline is 0.1248.

## References

* [1] Byrne, Denis, and Harvey Dillon. "The National Acoustic Laboratories'(NAL) new procedure for selecting the gain and frequency response of a hearing aid." Ear and hearing 7.4 (1986): 257-265. [doi:10.1097/00003446-198608000-00007](https://doi.org/10.1097/00003446-198608000-00007)
* [2] Kates J M, Arehart K H. "The Hearing-Aid Audio Quality Index (HAAQI)". IEEE/ACM transactions on audio, speech, and language processing, 24(2), 354–365. [doi:10.1109/TASLP.2015.2507858](https://doi.org/10.1109%2FTASLP.2015.2507858)
* [3] Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2016). "FMA: A dataset for music analysis". arXiv preprint arXiv:1612.01840. [doi:10.48550/arXiv.1612.01840](https://doi.org/10.48550/arXiv.1612.01840)
* [4] Bogdanov, D., Won, M., Tovstogan, P., Porter, A., & Serra, X. (2019). The MTG-Jamendo dataset for automatic music tagging. ICML.
