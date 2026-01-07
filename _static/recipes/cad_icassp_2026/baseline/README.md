# The ICASSP 2026 Cadenza Challenge: Predicting Lyric Intelligibility (CLIP1) baseline code

Code to support the ICASSP 2026 Cadenza Challenge (CLIP1)

For more information about the CLIP1 please [visit](https://cadenzachallenge.org/)

## 1. Data Structure

### 1.1 Obtaining the CLIP1 data

To download the CLIP1 data, please [register](https://cadenzachallenge.org/docs/clip1/take_part/registration).

The data will download into two package files called `cadenza_clip1_data.train.v1.0.tar.gz` and `cadenza_clip1_data.valid.v1.0.tar.gz`

Unpack these packages under the same root using

```bash
tar -xvzf cadenza_clip1_data.train.v1.0.tar.gz  # For training data
tar -xvzf cadenza_clip1_data.valid.v1.0.tar.gz # For validation data
```

Once unpacked the directory structure will be as follows

```bash
cadenza_data/
├── metadata/
├── train/
│   ├── signals/      # Audio (1) to predict intelligibility
│   └── unprocessed/  # Audio (2) without hearing loss
├── valid/
│   ├── signals/
│   └── unprocessed/
└── Manifest/
```

### 1.2 Demo data

Running the baseline code over the full dataset can be quite time consuming.
To allow you to easily try out the code with a small amount of data, we have provided a small subset of demo data containing 10 signals.

To use the demo data download the file `cadenza_clip1_data.demo.v1.0.tar.gz` which can be found at the same download sites as the full dataset.

Unpack this package under this directory, i.e., under `recipes/cad_icassp_2026/baseline` using

```bash
tar -xvzf cadenza_clip1_data.demo.v1.0.tar.gz
```

Note, you need to set the `root.path` variable to the parent directory of `cadenza_data` in `common.yaml` config file.

### 1.3 Precomputed Scores

For convenience, precomputed STOI and Whisper correctness scores are provided in the `precomputed` directory.
These scores are stored in JSONL files and can be used directly without running the `compute_stoi.py` or `compute_whisper` scripts.
The files are named as follows:

- `cadenza_data.train.stoi.jsonl`: Contains precomputed STOI scores for the training set.
- `cadenza_data.valid.stoi.jsonl`: Contains precomputed STOI scores for the validation set.
- `cadenza_data.train.whisper.mixture.jsonl`: Contains precomputed Whisper correctness scores from the mixture for the training set.
- `cadenza_data.valid.whisper.mixture.jsonl`: Contains precomputed Whisper correctness scores from the mixture for the validation set.

To use these precomputed scores, copy the relevant file to the `exp/` directory and rename it to match the expected output of the `compute_stoi.py` or `compute_whisper.py` scripts.
For example, to copy the precomputed STOI scores for the training set, run:

```bash
cp precomputed/cadenza_data.train.stoi.jsonl exp/cadenza_data.train.stoi.jsonl
```

This allows you to skip the computation step and proceed directly to making intelligibility predictions.

## 2. Baseline

The baseline prediction model is a simple logistic regression model that maps STOI or Whisper correctness scores onto the sentence correctness values.

There are two baselines:

1. STOI-based baseline
2. Whisper-based baseline using the mixture as input

### 2.1 STOI-Based baseline

The STOI-based baseline uses the Short-Time Objective Intelligibility (STOI) metric to predict the intelligibility of the audio signals.
For reference signal, it estimates the vocals from the unprocessed signal.

### 2.2 Whisper-Based baseline

The Whisper-based baseline uses OpenAI's Whisper model to transcribe the audio signals and compute a correctness score.
It uses the processed signal as input to Whisper and computes the `hits/total_words` as score.
The transcription and ground truth are normalised and contractions expanded before computing the correctness.

## 3. Running the Baseline

### 3.1 Computing scores

To compute the scores for the training data set, run the following commands:

For stoi:

```bash
python compute_stoi.py data.cadenza_data_root=/path/to/cadenza_data/parent split=train baseline.system=stoi
```

For Whisper:

```bash
python compute_whisper.py data.cadenza_data_root=/path/to/cadenza_data/parent split=train baseline.system=whisper
```

This will generate an output file containing the corresponding scores.
The file will be saved in the `exp/` directory and named `<DATASET>.train.<BASELINE SYSTEM>.jsonl`, where `<DATASET>` is the dataset name specified in the command (e.g., `cadenza_data`).

To run the same commands for the validation set, simply change the `split` argument to `valid`.

### 3.2 Making intelligibility predictions

The baseline intelligibility predictions are made by using a logistic fitting to map the corresponding scores onto the sentence correctness values.
This is done using `predict.py`, which will produce a CSV file named `exp/<DATASET>.train.<BASELINE SYSTEM>.predict.csv` containing the predictions in the format required for submission to the challenge.

For example, to make predictions for the validation set using the STOI-based baseline, run:

```bash
python predict.py data.cadenza_data_root=/path/to/cadenza_data/parent split=valid baseline.system=stoi
```

### 3.3 Evaluating the predictions

Finally, the `evaluate.py` script will compare the provided predictions with the ground truth and compute the error metrics.

```bash
python evaluate.py data.cadenza_data_root=/path/to/cadenza_data/parent split=valid baseline.system=stoi
```

Results will be displayed on the terminal and saved to the file `exp/<DATASET>.<BASELINE SYSTEM>evaluate.jsonl`.

These predictions are intended for submission to the leaderboard for evaluation. See the challenge website for details on how to submit your predictions.

### 3.4 Evaluating the predictions

Unlike for the training data set, the validation set ground truth is not provided, but once predictions are made they can be submitted to the leaderboard for evaluation.
See the challenge website for details of how to submit your predictions.
