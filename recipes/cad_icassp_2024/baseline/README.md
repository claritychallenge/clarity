# The ICASSP 2024 Cadenza Challenge (CAD_ICASSP_2024)

Cadenza challenge code for the ICASSP 2024 Cadenza Challenge.

For more information please visit the [challenge website](https://cadenzachallenge.org/docs/icassp_2024/intro).

## 1. Data structure

The ICASSP 2024 Cadenza Challenge dataset is based on the MUSDB18-HQ dataset.
To download the data, please visit [Download data and software](https://cadenzachallenge.org/docs/icassp_2024/take_part/download)
webpage.

The data is split into four packages: `cadenza_icassp2024_core.v1_0.tgz`,
`cadenza_icassp2024_augmentation_medleydb.tar.gz`, `cadenza_icassp2024_augmentation_bach10.tar.gz`
and `cadenza_icassp2024_augmentation_fma_small.tar.gz`.

Unpack packages under the same root directory using

```bash
tar -xvzf <PACKAGE_NAME>
```

### 1.1 Necessary data

* **Core** contains the metadata and audio signal to generate the ICASSP 2024 dataset.

```text
cadenza_data
├───audio
|   ├───hrtf (336 kB)
|   |   |  BTE_fr-VP_E1-n22.5.wav
|   |   |  BTE_fr-VP_E1-n30.0.wav
|   |   |  ...
|   |
|   └───music
|       └───train (20.2 GB)
|           ├───A Classic Education - NightOwl
|           |   |  bass.wav
|           |   |  drums.wav
|           |   |  other.wav
|           |   |  vocals.wav
|           |   |  mixture.wav
|           |
|           ├───...
|
└───metadata  (328 kB)
    |  gains.json
    |  head_positions.json
    |  listeners.train.json
    |  listeners.valid.json
    |  musdb18.train.json
    |  musdb18.valid.json
    |  scene_listeners.train.json
    |  scenes.train.json
    |  ...

```

### 1.2 Additional optional data

If you need additional music data for training your model, please restrict to the use of [MedleyDB](https://medleydb.weebly.com/) [[5](#references)] [[6](#references)],
[BACH10](https://labsites.rochester.edu/air/resource.html) [7] and [FMA-small](https://github.com/mdeff/fma) [7].

**Keeping the augmentation data restricted to these datasets will ensure that the evaluation is fair for all participants**.

* **MedleyDB** contains both MedleyDB versions 1 [[5](#references)] and 2 [[6](#references)] datasets.

```text
cadenza_data
└───audio
    └───MedleyDB (164 GB)
        ├───Audio
        └───Metadata
```

* **BACH10** contains the BACH10 dataset [[7](#references)].

Tracks from the BACH10 dataset are not included in MUSDB18-HQ and can all be used as training augmentation data.

```text
cadenza_data
└───audio
    └───Bach10 (150 MB)
        ├───01-AchGottundHerr
        ├───...
```

* **FMA Small** contains the FMA small subset of the FMA dataset [[8](references)].

Tracks from the FMA small dataset are not included in the MUSDB18-HQ.
This dataset does not provide independent stems but only the full mix.
However, it can be used to train an unsupervised model to better initialise a supervised model.

```text
cadenza_data
└───audio
    └───fma_small (8 GB)
        ├───000
        ├───001
        ├───...
```

## 2. Baseline

In the `baseline/` folder, we provide code for running the baseline enhancement system and performing the objective evaluation.
Note that we use [hydra](https://hydra.cc/docs/intro/) for config handling.

### 2.1 Enhancement

The baseline enhance takes an out-of-the-box source separation model and estimates
the VDBO (vocals, drums, bass and others) stems for each song-listener pair.

For each estimated stem, the baseline applies the gains and remix the signal.
A simple NAL-R [2] fitting amplification is applied to the final remix

The basile offers 2 source separation options:

1. [Hybrid Demucs](https://github.com/facebookresearch/demucs) [[1](#references)]  distributed on [TorchAudio](https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html)
2. [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) [[2](#references)]  distributed through Pytorch hub.

To run the baseline enhancement system first, make sure that `paths.root` in `config.yaml` points to
where you have installed the Cadenza data.
You can also define your own `path.exp_folder` to store enhanced
signals and evaluated results.

Then run:

```bash
python enhance.py
```

Alternatively, you can provide the root variable on the command line, e.g.,

```bash
python enhance.py path.root=/Volumes/data/cadenza_data
```

To get a full list of the parameters, run:

```bash
python enhance.py --help
```

The folder `enhanced_signals` will appear in the `exp` folder.

### 2.2 Evaluation

The `evaluate.py` simply takes the signals stored in `enhanced_signals` and computes the HAAQI [[3](#references)] scores

To run the evaluation stage, make sure that `path.root` is set in the `config.yaml` file and then run

```bash
python evaluate.py
```

A csv file containing the left and right channels HAAQI scores and the mean of both will be generated in the `path.exp_folder`.

To check the HAAQI code, see [here](https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py).

Please note: you will not get identical HAAQI scores for the same signals if the random seed is not defined
(in the given recipe, the random seed for each signal is set as the last eight digits of the song md5).
As there are random noises generated within HAAQI, but the differences should be sufficiently small.

The average validation score for the baseline is:

* Demucs = 0.6496 HAAQI
* Open-Unmix = 0.5822 HAAQI

## References

* [1] Défossez, A. "Hybrid Spectrogram and Waveform Source Separation". Proceedings of the ISMIR 2021 Workshop on Music Source Separation. [doi:10.48550/arXiv.2111.03600](https://arxiv.org/abs/2111.03600)
* [2] Stöter, F. R., Liutkus, A., Ito, N., Nakashika, T., Ono, N., & Mitsufuji, Y. (2019). "Open-Unmix: A Reference Implementation for Music Source Separation". Journal of Open Source Software, 4(41), 1667. [doi:10.21105/joss.01667](https://doi.org/10.21105/joss.01667)
* [3] Byrne, Denis, and Harvey Dillon. "The National Acoustic Laboratories'(NAL) new procedure for selecting the gain and frequency response of a hearing aid." Ear and hearing 7.4 (1986): 257-265. [doi:10.1097/00003446-198608000-00007](https://doi.org/10.1097/00003446-198608000-00007)
* [4] Kates J M, Arehart K H. "The Hearing-Aid Audio Quality Index (HAAQI)". IEEE/ACM transactions on audio, speech, and language processing, 24(2), 354–365. [doi:10.1109/TASLP.2015.2507858](https://doi.org/10.1109%2FTASLP.2015.2507858)
* [5] R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam and J. P. Bello, "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research", in 15th International Society for Music Information Retrieval Conference, Taipei, Taiwan, Oct. 2014. [pdf](https://archives.ismir.net/ismir2014/paper/000322.pdf)
* [6] Rachel M. Bittner, Julia Wilkins, Hanna Yip and Juan P. Bello, "MedleyDB 2.0: New Data and a System for Sustainable Data Collection" Late breaking/demo extended abstract, 17th International Society for Music Information Retrieval (ISMIR) conference, August 2016. [pdf](https://wp.nyu.edu/ismir2016/wp-content/uploads/sites/2294/2016/08/bittner-medleydb.pdf)
* [7] Zhiyao Duan, Bryan Pardo and Changshui Zhang, "Multiple fundamental frequency estimation by modeling spectral peaks and non-peak regions," IEEE Trans. Audio Speech  Language Process., vol. 18, no. 8, pp. 2121-2133, 2010. [doi:10.1109/TASL.2010.2042119](https://doi.org/10.1109/TASL.2010.2042119)
* [8] Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2016). "FMA: A dataset for music analysis". arXiv preprint arXiv:1612.01840. [doi:10.48550/arXiv.1612.01840](https://doi.org/10.48550/arXiv.1612.01840)
