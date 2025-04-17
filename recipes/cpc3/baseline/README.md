# The 3rd Clarity Prediction Challenge (CPC3) baseline code

TODO: update this

Code to support the 3rd Clarity Prediction Challenge (CPC3).

For more information about the CPC3 please [visit](https://claritychallenge.org/)

## 1. Data structure

### 1.1 Obtaining the CPC3 data

To download the CPC3 data, please follow the instructions on the [download page](https://claritychallenge.org/docs/cpc3/taking_part/cpc3_download) of the challenge website.

The data will download into two package files called `clarity_CPC3_data.v1_1.tar.gz` and `clarity_CPC3_data.dev.v1_0.tar.gz.

Unpack these packages under the same root using

```bash
tar -xvzf clarity_CPC3_data.v1_1.tar.gz
tar -xvzf clarity_CPC3_data.dev.v1_0.tar.gz
```

(Ignore any warnings about overwriting files, this is expected.)

Once unpacked the directory structure will be as follows

```bash
clarity_CPC3_data/
├── clarity_data/
│   ├── metadata/  # Listener responses and characteristics
│   ├── train/
│   │   ├── references/  # Reference signals for intelligibility prediction
│   │   └── signals/     # Hearing aid output signals
│   └── dev/
│       ├── references/
│       └── signals/
└── manifest/
```

### 1.2 Demo data

Running the baseline code over the full dataset can be quite time consuming. To allow you to easily try out the code with a small amount of data, we have provided a small subset of demo data containing 30 signals.

To use the demo data download the file `clarity_CPC3_demo_data.v1_0.tar.gz` which can be found at the same download sites as the full dataset. ([See our website](https://claritychallenge.org/docs/cpc3/cpc3_download) for details of how to obtain the data.)

Unpack this package under this directory, i.e., under `recipes/cpc3/baseline` using

```bash
tar -xvzf clarity_CPC3_demo_data.v1_0.tar.gz
```

Note, the `root.path` variable in `config.yaml` is already set to point to the demo data by default.

The demo data only contains has the following structure,

```bash
clarity_CPC3_data/
├── clarity_demo_data/
│   ├── metadata/  # Listener responses and characteristics
│   └─── train/
│       ├── references/  # Reference signals for intelligibility prediction
│       └── signals/
└── manifest/
```

## 2. Baseline

The baseline prediction model is a simple logistic regression model that maps HASPI scores [[1](#references)] onto the sentence correctness values.

### 2.1 Computing the HASPI scores

Compute the HASPI scores over the training data set and store results

```bash
python compute_haspi.py dataset=clarity_demo_data split=train
```

This will generate the output file containing HASPI scores. This file will be called `exp/<DATASET>.haspi.jsonl` where `<DATASET>` is the name specified on the commandline.

Note, if the results file is already present, the script will only compute the scores for the signals that are not already present in the file. These missing results will be appended to the output file. This allows the script to be halted and restarted.

### 2.2 Making intelligibility predictions

The baseline intelligibility predictions are made by using a logistic fitting to map the HASPI scores onto the sentence correctness values. This is done using `predict.py` which will produce a `csv` file named `exp/<DATASET>.predict.csv` containing the predictions in the format that is required for submission to the challenge.

```bash
python predict.py dataset=clarity_demo_data split=train
```

### 2.3 Evaluating the predictions

Finally, the `evaluate.py` script will compare the provided predictions with the ground truth and compute the error metrics.

```bash
python evaluate.py dataset=clarity_demo_data split=train
```

Results will be displayed on the terminal and saved to the file `exp/<DATASET>.evaluate.jsonl`.

## References

[1] Kates, J.M. and Arehart, K.H., 2021. The hearing-aid speech perception index (HASPI) version 2. Speech Communication, 131, pp.35-46.
