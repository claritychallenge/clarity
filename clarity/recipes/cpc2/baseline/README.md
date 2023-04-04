# The 2nd Clarity Prediction Challenge (CPC2) baseline code

Code to support the 2nd Clarity Prediction Challenge (CPC2).

For more information about the CPC2 please [visit](https://claritychallenge.org/)

## 1. Data structure

### 1.1 Obtaining the CPC2 data

To download the CPC2 data, please follow the instructions on the [download page](https://claritychallenge.org/docs/cpc2/taking_part/cpc2_download) of the challenge website.

The data will download into a package file called `clarity_CPC2_data.v1_1.tar.gz`.

Unpack this package using

```bash
tar -xvzf clarity_CPC2_data.v1_1.tar.gz
```

Once unpacked the directory structure will be as follows

**clarity_CPC2_data.v1_1** contains the training data:

```bash
clarity_CPC2_data
├── clarity_data
│   ├── HA_outputs  # The hearing aid output signals
│   │   ├── signals
│   │   │   ├── CEC1
│   │   │   └── CEC2
│   │   ├── train.1
│   │   │   ├── CEC1
│   │   │   └── CEC2
│   │   ├── train.2
│   │   │   ├── CEC1
│   │   │   └── CEC2
│   │   └── train.3
│   │       ├── CEC1
│   │       └── CEC2
│   ├── metadata    # Metadata including the intelligibility scores
│   └── scenes      # Contains reference signals
│       ├── CEC1
│       └── CEC2
└── manifest        # Lists the package contents
```

### 1.2 Demo data

Running the baseline code over the full dataset can be quite time consuming. To allow you to easily try out the code with a small amount of data, we have provided a small subset of demo data containing 30 signals.

To use the demo data download the file `clarity_CPC2_data_demo.v1_1.tar.gz` using the following [Google Drive link](https://drive.google.com/file/d/1oLlo9w9Z7A2NP35IntJsRySIPl-oUajU/view?usp=sharing).

Unpack this package under this directory, i.e., under `recipes/cpc2/baseline` using

```bash
tar -xvzf clarity_CPC2_data_demo.v1_1.tar.gz
```

Note, the `root.path` variable in `config.yaml` is already set to point to the demo data by default.

The demo data only contains one training set split, `train.1` and `CEC2` data. It has the following structure,

```bash
clarity_CPC2_data_demo
├── clarity_data
│   ├── HA_outputs
│   │   ├── signals
│   │   │   └── CEC2
│   │   └── train.1
│   │       └── CEC2
│   ├── metadata
│   └── scenes
│       └── CEC2
└── manifest
```

## 2. Baseline

The baseline prediction model is a simple logistic regression model that maps HASPI scores [[1](#references)] onto the sentence correctness values.

### 2.1 Computing the HASPI scores

Compute the HASPI scores over the training data set and store results

```bash
python compute_haspi.py dataset=CEC2.train.1
```

This will generate the output file containing HASPI scores. This file will be called `exp/<DATASET>.haspi.jsonl` where `<DATASET>` is the name specified on the commandline.

Note, if the results file is already present, the script will only compute the scores for the signals that are not already present in the file. These missing results will be appended to the output file. This allows the script to be halted and restarted.

### 2.2 Making intelligibility predictions

The baseline intelligibility predictions are made by using a logistic fitting to map the HASPI scores onto the sentence correctness values. This is done using `predict.py` which will produce a `csv` file named `exp/<DATASET>.predict.jsonl` containing the predictions in the format that is required for submission to the challenge.

```bash
python predict.py dataset=CEC2.train.1
```

### 2.3 Evaluating the predictions

Finally, the `evaluate.py` script will compare the provided predictions with the ground truth and compute the error metrics.

```bash
python evaluate.py dataset=CEC2.train.1
```

Results will be displayed on the terminal and saved to the file `exp/<DATASET>.evaluate.jsonl`.

## References

[1] Kates, J.M. and Arehart, K.H., 2021. The hearing-aid speech perception index (HASPI) version 2. Speech Communication, 131, pp.35-46.
