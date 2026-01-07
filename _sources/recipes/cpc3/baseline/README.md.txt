# The 3rd Clarity Prediction Challenge (CPC3) baseline code

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

### 1.3 Precomputed HASPI scores

For convenience, precomputed HASPI scores are provided in the `precomputed_haspi` directory. These scores are stored in JSONL files and can be used directly without running the `compute_haspi.py` script. The files are named as follows:

- `precomputed_haspi/clarity_data.train.haspi.jsonl`: HASPI scores for the full training dataset.
- `precomputed_haspi/clarity_demo_data.train.haspi.jsonl`: HASPI scores for the demo training dataset.
- `precomputed_haspi/clarity_data.dev.haspi.jsonl`: HASPI scores for the development dataset.

To use these precomputed scores, copy the relevant file to the `exp/` directory and rename it to match the expected output of the `compute_haspi.py` script. For example:

```bash
cp precomputed_haspi/clarity_data.train.haspi.jsonl exp/clarity_data.train.haspi.jsonl
```

This allows you to skip the HASPI computation step and proceed directly to making intelligibility predictions.

## 2. Baseline - evaluating with training data

The baseline prediction model is a simple logistic regression model that maps HASPI scores [[1](#references)] onto the sentence correctness values. It can be run using and evaluated on the training data set (this section), or using the development data set (see the next section).

### 2.1 Computing the HASPI scores

To compute the HASPI scores for the training data set, run the following command:

```bash
python compute_haspi.py dataset=clarity_data split=train
```

If you are using the demo dataset, replace `clarity_data` with `clarity_demo_data`:

```bash
python compute_haspi.py dataset=clarity_demo_data split=train
```

This will generate an output file containing the HASPI scores. The file will be saved in the `exp/` directory and named `<DATASET>.train.haspi.jsonl`, where `<DATASET>` is the dataset name specified in the command (e.g., `clarity_data` or `clarity_demo_data`).

For example:

- For the full dataset: `exp/clarity_data.train.haspi.jsonl`
- For the demo dataset: `exp/clarity_demo_data.train.haspi.jsonl`

If the results file already exists, the script will only compute scores for signals that are not already present in the file. Missing results will be appended to the existing file. This allows the script to be halted and restarted without losing progress.

Note: The `exp/` directory will be created automatically if it does not already exist. Ensure you have write permissions in the current directory.

### 2.2 Making intelligibility predictions

The baseline intelligibility predictions are made by using a logistic fitting to map the HASPI scores onto the sentence correctness values. This is done using `predict_train.py`, which will produce a CSV file named `exp/<DATASET>.train.predict.csv` containing the predictions in the format required for submission to the challenge.

```bash
python predict_train.py dataset=clarity_data
```

Note that the algorithm avoids overfitting by ensuring that the predictions for a signal are formed from HASPI scores from a disjoint subset of the training data. Specifically, this set is formed by removing any signals that have the same target sentence, listener or HA system as the signal for which the prediction is being made. This helps to ensure that the performance will be representative of the performance on the disjoint development and test sets.

### 2.3 Evaluating the predictions

Finally, the `evaluate.py` script will compare the provided predictions with the ground truth and compute the error metrics.

```bash
python evaluate.py dataset=clarity_demo_data split=train
```

Results will be displayed on the terminal and saved to the file `exp/<DATASET>.evaluate.jsonl`.

## 3. Baseline - evaluating with development data

The baseline prediction model can also be run over the development data set. The development set ground truth is not provided, but once predictions are made they can be submitted to the leaderboard for evaluation. See the challenge website for details of how to submit your predictions.

### 3.1 Computing the HASPI scores

This is done identically to the training data set, but using the `split=dev` option.

```bash
python compute_haspi.py dataset=clarity_data split=dev
```

This will generate the output file containing HASPI scores. This file will be called `exp/<DATASET>.dev.haspi.jsonl` where `<DATASET>` is the name specified on the commandline.

### 3.2 Making intelligibility predictions

The baseline intelligibility predictions for the development set use a logistic fitting to map the HASPI scores onto the sentence correctness values. This is done using `predict_dev.py`, which will produce a CSV file named `exp/<DATASET>.dev.predict.csv`.

```bash
python predict_dev.py dataset=clarity_data
```

These predictions are intended for submission to the leaderboard for evaluation. See the challenge website for details on how to submit your predictions.

### 3.3 Evaluating the predictions

Unlike for the training data set, the development set ground truth is not provided, but once predictions are made they can be submitted to the leaderboard for evaluation. See the challenge website for details of how to submit your predictions.

## References

[1] Kates, J.M. and Arehart, K.H., 2021. The hearing-aid speech perception index (HASPI) version 2. Speech Communication, 131, pp.35-46.
