# The First Cadenza Challenge (CAD1) - Task 1: Listening music via headphones


Cadenza challenge code for the First Cadenza Challenge (CAD1) Task1.

For more information please visit the [challenge website](https://cadenzachallenge.org/docs/cadenza1/cc1_intro).

## 1. Data structure
The First Cadenza Challenge - task 1 is using the MUSDB18-HQ dataset.
The data is split into train, validation and test following the same split from museval.
I.e., 86 songs are for training, [16 for validation](validation_tracks:) and 50 for evaluation.

To download the data, please visit [here](). The data is split into `cadenza_cad1_task1_music.tar.gz` (containing the MUSDB18-HQ dataset) and
`cadenza_cad1_task1_metadata.tar.gz` (containing the necessary metadata). Alternatively, you can download the MUSDB18-HQ dataset from the official [SigSep website](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav).
If you opt for this alternative, be sure to download the uncompressed wav version. Note that you will need both packages to run the baseline system.

If you need additional music data for training your model, please restrict to the use of [MedleyDB](https://medleydb.weebly.com/) [4][5],
[BACH10](https://labsites.rochester.edu/air/resource.html) [6] and [FMA-small](https://github.com/mdeff/fma) [7].
Theses are shared as `cadenza_cad1_task1_medleydb.tar.gz`, `cadenza_cad1_task1_bach10.tar.gz` and `cadenza_cad1_task1_fma_small.tar.gz`.
**Keeping the augmentation data restricted to these datasets will ensure that the evaluation is fair for all participants**.

Unpack packages under the same root directory using
```bash
tar -xvzf <PACKAGE_NAME>
```

### 1.1 Necessary data
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
            ├───listeners.json
            ├───musdb18_train.json
            ├───musdb18_valid.json
            └───musdb18_test.json
```

### 1.2 Additional optional data
* **MedleyDB** contains both MedleyDB versions 1 [4] and 2 [5] datasets.
MedleyDB dataset is not used in the evaluation and is your responsibility to exclude any song that is already in the training set.
```text
cadenza_data
└───task1
    └───audio
        └───MedleyDB
            ├───Audio
            └───Metadata
```

* **BACH10** contains the BACH10 dataset [6].
BACH10 dataset is not used in the evaluation.
```text
cadenza_data
└───task1
    └───audio
        └───fma_small
            ├───000
            ├───001
            ├───...
```

* **FMA Small** contains the FMA small subset of the FMA dataset [7].
FMA small dataset is not used in the evaluation. This dataset does not provide the independent stems, but only the full mix.
However, it can be used for any unsupervised learning task.
```text
cadenza_data
└───task1
    └───audio
        └───fma_small
            ├───000
            ├───001
            ├───...
```

## 2. Baseline

In the `baseline/` folder, we provide code for running the baseline enhancement system and performing the objective evaluation.
Note that we use [hydra](https://hydra.cc/docs/intro/) for config handling.

### 2.1 Enhancement

The baseline enhance simply takes the out-of-the-box [Hybrid Demucs](https://github.com/facebookresearch/demucs)[1]
source separation model distributed on [TorchAudio](https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html)
and applies a simple NAL-R [2] fitting amplification and a simple automatic gain compressor to each
VDBO (`vocals`, `drums`, `bass` and `others`) stem.

The remixing is performed by summing the amplified VDBO stems.

The baseline will generate a left and right signal for each VDBO stem and a remixed signal, totalling 9 signals per song-listener.

To run the baseline enhancement system, firstly specify `paths.root` in `config.yaml` to point to
where you have installed the Cadenza data. You can also define your own `path.exp_folder` to store enhanced
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

The `evaluate.py` simply takes the signals stored in `enhanced_signals` and computes the HAAQI [3] score
for each of the eight left and right VDBO stems.
The average of these eight scores is computed and returned for each signal.

To run the evaluation stage, make sure that `path.root` is set in the config.yaml file and then run
```bash
python evaluate.py
```
A csv file containing the eight HAAQI scores and the combined score will be generated in the `path.exp_folder`.

To check the HAAQI code, see [here](../../../../clarity/evaluator/haaqi).

Please note: you will not get identical HAAQI scores for the same signals if the random seed is not defined
(in the given recipe, the random seed for each signal is set as the last eight digits of the song md5).
As there are random noises generated within HAAQI, but the differences should be sufficiently small.

We ran evaluation for the baseline for five times, and the average overall score is XXX +/- XXXX.

## References

* [1] Défossez, A. "Hybrid Spectrogram and Waveform Source Separation". Proceedings of the ISMIR 2021 Workshop on Music Source Separation. [doi:10.48550/arXiv.2111.03600](https://arxiv.org/abs/2111.03600)
--


]
* [2] Byrne, Denis, and Harvey Dillon. "The National Acoustic Laboratories'(NAL) new procedure for selecting the gain and frequency response of a hearing aid." Ear and hearing 7.4 (1986): 257-265.
* [3] Kates J M, Arehart K H. "The Hearing-Aid Audio Quality Index (HAAQI)". IEEE/ACM transactions on audio, speech, and language processing, 24(2), 354–365.
* [4] R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam and J. P. Bello, "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research", in 15th International Society for Music Information Retrieval Conference, Taipei, Taiwan, Oct. 2014.
* [5] Rachel M. Bittner, Julia Wilkins, Hanna Yip and Juan P. Bello, "MedleyDB 2.0: New Data and a System for Sustainable Data Collection" Late breaking/demo extended abstract, 17th International Society for Music Information Retrieval (ISMIR) conference, August 2016.
* [6] Zhiyao Duan, Bryan Pardo and Changshui Zhang, "Multiple fundamental frequency estimation by modeling spectral peaks and non-peak regions," IEEE Trans. Audio Speech  Language Process., vol. 18, no. 8, pp. 2121-2133, 2010.
* [7] Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2016). "FMA: A dataset for music analysis". arXiv preprint arXiv:1612.01840.