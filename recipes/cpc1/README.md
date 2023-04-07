# The 1st Clarity Prediction Challenge (CPC1)

Clarity challenge code for the 1st prediction challenge (CPC1).

There are two traks in CPC1:

- **closed-set**: the evaluation systems and listeners are covered in the training set, the signals and listener repsonses are provieded in `CPC1.train.json`
- **open-set**: the evaluation systems and listeners are not seen in the training set, the signals and listener responses are provieded in `CPC1_indep.train.json`

For more information about the CPC1 please visit [claritychallenge.org/docs/cpc1/cpc1_intro](https://claritychallenge.org/docs/cpc1/cpc1_intro).

## Data structure

To download the CPC1 data, please visit [here](https://mab.to/R6H84YNf74p5U).

**clarity_CPC1_data.v1_1** contains the training data:

```text
clarity_data
|
└───HA_outputs
|   |
|   └───train 3.8G
|   |
|   └───train_indep 2.8G
|
└───scenes 12.1G

metadata
    |CPC1.train.json
    |CPC1.train_indep.json
    |listener_data.CPC1_train.xlsx
    listeners.CPC1_train.json
    |scenes.CPC1_train.json
```

**clarity_CPC1_data.test.v1** follows the same structure as **clarity_CPC1_data.v1_1**, except that the listener responses (i.e. test labels) are not included. The test listener responses are in the `test_listener_responses`.

## Baseline

The baseline folder provides the code of the Cambridge Auditory Group MSBG hearing loss model and MBSTOI, see [CEC1](../cec1/baseline). Run `run.py` to generate the predicted intelligibility, and then run `compute_scores.py` to apply logistic fitting and compute the evaluation scores, including RMSE, normalised cross-correlation, Kendall's Tau coefficient.

## Citing CPC1

```text
@inproceedings{barker2022the,
  title={The 1st Clarity Prediction Challenge: A machine learning challenge for hearing aid intelligibility prediction},
  author={Jon Barker, Michael Akeroyd, Trevor J. Cox, John F. Culling, Jennifer Firth, Simone Graetzer, Holly Griffiths, Lara Harris, Graham Naylor, Zuzanna Podwinska, Eszter Porter and Rhoddy Viveros Munoz},
  year={2022}
}

```
