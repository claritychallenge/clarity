# Baseline code for the Second Lyric Intelligibility Prediction (CLIP2) challenge

Code to support the CLIP2 challenge.

For more information about the CLIP2 please the [Cadenza Challenge](https://cadenzachallenge.org/) website.

## 1. Files

| File                             | Description                        |
|----------------------------------|------------------------------------|
| `conf/config.yaml`               | Configuration file                 |
| `clip_dataset.py`                | Dataloader for CLIP2 dataset       |
| `extra_requirements.txt`         | Python requirements                |
| `inference.py`                   | Command-line inference script      |
| `lyric_intelligibility_model.py` | Model definition                   |
| `README.md`                      | This file                          |
| `test.py`                        | Evaluation script                  |
| `train.py`                       | Training script                    |
| `train_job.sh`                   | Orchrestrate the training pipeline |

## 2. Data Structure

### 2.1 Obtaining the CLIP2 data

To download the CLIP2 data, please [register](https://cadenzachallenge.org/docs/clip2/take_part/registration).

The data will download into two package files called `cadenza_clip2_data.train.v1.0.tar.gz` and `cadenza_clip2_data.valid.v1.0.tar.gz`

Unpack these packages under the same root using

```bash
tar -xvzf cadenza_clip2_data.train.v1.0.tar.gz  # For training data
tar -xvzf cadenza_clip2_data.valid.v1.0.tar.gz # For validation data
```

Once unpacked, the directory structure will be as follows

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

### 2.2 Demo data

Running the baseline code over the full dataset can be time-consuming.
To allow you to easily try out the code with a small amount of data, we have provided a small subset of demo data containing 10 signals.

To use the demo data download the file [cadenza_clip2_data.demo.v1.0.tar.gz](https://drive.google.com/file/d/16WVDpJDFZNER1GoReomizWR-zg6J-QOn/view?usp=drive_link).

Note, you need to set the `root.path` variable to the parent directory `cadenza_data/clip2` in `conf/config.yaml` config file.

## 3. Baseline

The baseline prediction model is a DNN regression model that uses Whisper encoder as front-end and a series of MLP layers as back-end.

There are two baselines:

1. Mono (`cadenzachallenge/CLIP2-BaselineMono`): which downmix the music signal to mono before the prediction.
2. Better-ear (`cadenzachallenge/CLIP2-BaselineBE`): that predicts intelligibility for each channel independently using the same model and returns the maximum value.

### 3.1 Train the model

To train the baseline model, you can follow the procedure in `train_job.sh` script.
This script orchestrates the whole training by:

1. Trains a mono model (downmixing stereo to mono).
2. Trains a better-ear model (predicting intelligibility for each channel independently).
3. Evaluate/Predict from validation set.

Note that for training, the script is splitting the training set into train/valid subsets (80% / 20%)

## 4. Making intelligibility predictions

### 4.1 Loading and using the model

To make intelligibility predictions, you can use the `IntelligibilityPredictor` class.
This will download the pre-trained model from HuggingFace and make prediction from a song

```python
from lyric_intelligibility_model import IntelligibilityPredictor

predictor = IntelligibilityPredictor.from_pretrained(
    "cadenzachallenge/CLIP2-BaselineMono"
)
result = predictor.predict("path/to/song.wav")
print(result["score"])          # e.g. 0.83
print(result["channel"])
```

or from a directory

```python
results = predictor.predict("path/to/songs_dir")
for filename, r in results.items():
    print(filename, r["score"])
```

### 4.2 Command-line

We also provided the 'inference.py' script to make predictions from the command-line.

```bash
pip install torch transformers huggingface_hub safetensors soundfile soxr

python inference.py \
    --repo_id cadenzachallenge/CLIP2-BaselineMono \
    --audio path/to/song.wav          # or a directory

# options:
#   --no-better-ear   downmix stereo to mono instead of left/right max
#   --max-100         scale scores to [0, 100] instead of [0, 1]
#   --output FILE     custom CSV path (default: results.csv)
```
