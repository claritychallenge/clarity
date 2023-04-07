# The 1st Clarity Enhancement Challenge

Clarity challenge code for the first enhancement challenge (CEC1).

Please visit the [Clarity Challenge website](https://claritychallenge.github.io/clarity_CC_doc/docs/cec1/cec1_intro) for CEC1 information, and the [Clarity Workshop website](https://claritychallenge.github.io/clarity2021-workshop/results.html) for CEC1 results.

## Data structure

To download data, please visit [here](https://mab.to/bavGDV87BZISg). The data is split into two packages: `clarity_CEC1_data.train.tgz` [192 GB], `clarity_CEC1_data.dev_eval_metadata.tgz` [163 GB]. Please also download and unpack `clarity_CEC1_data.anechoic.v1_3.tgz` [11.4 GB], which contains the correct version of anechoic signals for reference, and replace the old anechoic signals within the train.tgz and dev_eval_metadata.tgz.

Unpack packages under the same root directory using

```bash
tar -xvzf <PACKAGE_NAME>
```

**Train** contains the training data:

```text
clarity_data
|
└───train
    └───interferers
    |   |    nosie 3.9G
    |   |    speech 4.5G
    |
    └───rooms
    |   |    ac 48M
    |   |    brir 46G
    |   |    rpf 379M
    |
    └───scenes 166G
    |
    └───targets 2.8G
```

**Dev_Eval_Metadata** contains development set, evaluation set (eval2 is the processed evaluation data by the baseline), and metadata,

```text
clarity_data
|
└───dev
|   └───interferers
|   |   |    nosie 587M
|   |   |    speech 1.4G
|   |
|   └───rooms
|   |   |    ac 20M
|   |   |    brir 20G
|   |   |    rpf 158M
|   |
|   └───scenes 72G
|   |
|   └───targets 1.3G
|
└───eval
|   |   |    nosie 675M
|   |   |    speech 1.3G
|   |
|   └───rooms
|   |   |    ac 12M
|   |   |    brir 12G
|   |   |    rpf 95M
|   |
|   └───scenes 58G
|   |
|   └───targets 749M
|
└───eval2/scenes 21G
```

## Data preparation

In this folder, we provide the code for generating train & scenes. If you simply tends to use the CEC1 data, please download with the link above.

## Baseline

In the baseline, the baseline enhancement code using OpenMHA is provided. The evaluation code using the Cambridge Auditory Group MSBG hearing loss model and MBSTOI is also provided.

## Refernces

* [1] Kayser, Hendrik, et al. "Open community platform for hearing aid algorithm research: open Master Hearing Aid (openMHA)." SoftwareX 17 (2022): 100953.
* [2] Baer, Thomas, and Brian CJ Moore. "Effects of spectral smearing on the intelligibility of sentences in noise." The Journal of the Acoustical Society of America 94.3 (1993): 1229-1241.
* [3] Baer, Thomas, and Brian CJ Moore. "Effects of spectral smearing on the intelligibility of sentences in the presence of interfering speech." The Journal of the Acoustical Society of America 95.4 (1994): 2277-2280.
* [4] Moore, Brian CJ, and Brian R. Glasberg. "Simulation of the effects of loudness recruitment and threshold elevation on the intelligibility of speech in quiet and in a background of speech." The Journal of the Acoustical Society of America 94.4 (1993): 2050-2062.
* [5] Stone, Michael A., and Brian CJ Moore. "Tolerable hearing aid delays. I. Estimation of limits imposed by the auditory path alone using simulated hearing losses." Ear and Hearing 20.3 (1999): 182-192.
* [6] Andersen, Asger Heidemann, et al. "Refinement and validation of the binaural short time objective intelligibility measure for spatially diverse conditions." Speech Communication 102 (2018): 1-13.

## Citing CEC1

```text
@inproceedings{graetzer2021clarity,
  title={Clarity-2021 challenges: Machine learning challenges for advancing hearing aid processing},
  author={Graetzer, SN and Barker, Jon and Cox, Trevor J and Akeroyd, Michael and Culling, John F and Naylor, Graham and Porter, Eszter and Viveros Munoz, R and others},
  booktitle={INTERSPEECH},
  volume={2},
  pages={686--690},
  year={2021},
  organization={International Speech Communication Association (ISCA)}
}
```
