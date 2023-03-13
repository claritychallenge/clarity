# Introduction

pyClarity is core data used across all [Clarity Challenges](https://claritychallenge.org/) and provides baselines,
toolkits and systems for working with the challenge data. It can be used for scene generation, hearing aid modelling and
HASPI speech intelligibility model to evaluate solutions.

## Tutorials

A series of Jupyter Notebooks are available on Colab to help you learn how to use the tools made available within this
package. To access these please refer to the [Clarity Tutorials](https://claritychallenge.org/tutorials).

## Challenges

- [2nd Clarity Enhancement Challenge (CEC2)](https://claritychallenge.org/docs/cec2/cec2_intro) ([GitHub Recipes](https://github.com/claritychallenge/clarity/tree/main/recipes/cec2))
- [1st Clarity Enhancement Challenge (CEC1)](https://claritychallenge.org/docs/cec1/cec1_intro) ([GitHub Recipes](https://github.com/claritychallenge/clarity/tree/main/recipes/cec1))
- [1st Clarity Prediction Challenge (CPC1)](https://claritychallenge.org/docs/cpc1/cpc1_intro) ([GitHub Recipes](https://github.com/claritychallenge/clarity/tree/main/recipes/cpc1))

## Workshops

- [Clarity 2021](https://claritychallenge.org/clarity2021-workshop/)
- [Clarity 2022](https://claritychallenge.org/clarity2022-workshop/)

## Tools

A number of tools are included in this repository

- **Hearing loss simulation**
  - [Cambridge MSBG hearing loss simulator](https://github.com/claritychallenge/clarity/tree/main/clarity/evaluator/msbg): descriptions can be found in the [CEC1 description](https://github.com/claritychallenge/clarity/tree/main/recipes/cec1); an usage example can be found in the [CEC1 baseline](https://github.com/claritychallenge/clarity/tree/main/recipes/cec1/baseline) evaluation script `evaluate.py`.
- **Objective intelligibility measurement**
  - [Modified binaural STOI (MBSTOI)](https://github.com/claritychallenge/clarity/tree/main/clarity/evaluator/mbstoi/mbstoi.py): a python implementation of MBSTOI. It is jointly used with the MSBG hearing loss model in the [CEC1 baseline](https://github.com/claritychallenge/clarity/tree/main/recipes/cec1/baseline). The official matlab implementation can be found here: <http://ah-andersen.net/code/>
  - [Hearing-aid speech perception index (HASPI)](https://github.com/claritychallenge/clarity/tree/main/clarity/evaluator/haspi/haspi.py): a python implementation of HASPI Version 2, and the better-ear HASPI for binaural speech signals. For official matlab implementation, request here: <https://www.colorado.edu/lab/hearlab/resources>
- **Hearing aid enhancement**
  - [Cambridge hearing aid fitting (CAMFIT)](https://github.com/claritychallenge/clarity/tree/main/clarity/enhancer/gha/gainrule_camfit.py): a python implementation of CAMFIT, translated from the [HörTech Open Master Hearing Aid (OpenMHA)](http://www.openmha.org/about/); the CAMFIT is used together with OpenMHA enhancement as the [CEC1 baseline](https://github.com/claritychallenge/clarity/tree/main/recipes/cec1/baseline), see `enhance.py`.
  - [NAL-R hearing aid fitting](https://github.com/claritychallenge/clarity/tree/main/clarity/enhancer/nalr.py): a python implementation of NAL-R prescription fitting. It is used as the [CEC2 baseline](https://github.com/claritychallenge/clarity/tree/main/recipes/cec2/baseline), see `enhance.py`.

In addition, differentiable approximation to some tools are provided:

- [x] [Differentiable MSBG hearing loss model](https://github.com/claritychallenge/clarity/tree/main/clarity/predictor/torch_msbg.py). See also the BUT implementation: <https://github.com/BUTSpeechFIT/torch_msbg_mbstoi>
- [ ] Differentiable HASPI (coming)

## Open-source systems

- CPC1:
  - [Exploiting Hidden Representations from a DNN-based Speech Recogniser for Speech Intelligibility Prediction in Hearing-impaired Listeners](https://github.com/claritychallenge/clarity/tree/main/recipes/cpc1/e032_sheffield)
  - [Unsupervised Uncertainty Measures of Automatic Speech Recognition for Non-intrusive Speech Intelligibility Prediction](https://github.com/claritychallenge/clarity/tree/main/recipes/cpc1/e029_sheffield)
- CEC1:
  - [A Two-Stage End-to-End System for Speech-in-Noise Hearing Aid Processing](https://github.com/claritychallenge/clarity/tree/main/recipes/cec1/e009_sheffield)
