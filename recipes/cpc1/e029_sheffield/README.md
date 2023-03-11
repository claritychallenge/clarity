# E029 - Unsupervised Uncertainty Measures of Automatic Speech Recognition for Non-intrusive Speech Intelligibility Prediction

The implementation of ["Unsupervised Uncertainty Measures of Automatic Speech Recognition for Non-intrusive Speech Intelligibility Prediction"](https://arxiv.org/abs/2204.04288), accepted to InterSpeech 2022. For the 1st Clarity Prediction Challenge (CPC1) programme and details, please see [here](https://claritychallenge.github.io/clarity2022-workshop/programme.html).

Please note: This code provides the implementation of the LS(LibriSpeech) + CPC1 model. The training with LS train-clean-100 added with CEC1 noise data is not provided here, as it is a bit over-complicated.

## Run the scripts

### Requirements

```text
torch==1.10.0
torchaudio==0.10.0
speechbrain==0.5.9
```

### Prepare ASR data

Same as e032_sheffield:

To train the ASR model and generate the hidden representations from it, the CPC1 data needs to be processed first. This part of code will (1) randomly split train and dev set; (2) run MSBG hearing loss simulation to all signals in `clarity_data/HA_output`; (3) resample signals to 16kHz and generate csv files for SpeechBrain ASR model training.

- Download `clarity_CPC1_data.v1_1.tgz` and `clarity_CPC1_data.test.v1.tgz`, untar them into `clarity_CPC1_data_train` and `clarity_CPC1_data_test`, respectively. See recipes/cpc1/baseline/README.
- Specify `root` in config.yaml. Both `clarity_CPC1_data_train` and `clarity_CPC1_data_test` should be in your root folder. You could also specify your own `exp_folder`.
- Specify `cpc1_track` as 'open' or 'closed'.
- Run `python prepare_data.py`
- `cpc1_asr_data` which contains the processed train & test CPC1 data and their csv files, and `data_split` which contains the train set and dev set scenes, will appear in your `exp_folder`.

### Train ASR models

- Create your `transformer_cpc1` folder to save ASR models. As we need an ensemble of ASR models, we need to train them separately. Therefore, create `N` sub-folders `asr0`, `asr1`, ..., `asrN` under `transformer_cpc1`.
- Specify output_folder & data_folder in transformer_cpc1.yaml:
`data_folder: !ref your_path/e029/cpc1_asr_data  # for closed-set`
OR
`data_folder: !ref your_path/e029/cpc1_asr_data_indep  # for open-set`
- Download the `save` folder (i.e. ASR transformer checkpoint) from: <https://drive.google.com/drive/folders/1ZudxqMWb8VNCJKvY2Ws5oNY3WI1To0I7>, and place it under your `transformer_cpc1/asr0/`, `transformer_cpc1/asr1/`, ..., `transformer_cpc1/asrN/` folder
- Train the ASR models, run:

```bash
python train_asr.py transformer_cpc1.yaml --seed 0000 --output_folder your_path/transformer_cpc1/asr0
python train_asr.py transformer_cpc1.yaml --seed 0001 --output_folder your_path/transformer_cpc1/asr1
python train_asr.py transformer_cpc1.yaml --seed 0002 --output_folder your_path/transformer_cpc1/asr2
...
python train_asr.py transformer_cpc1.yaml --seed 000N --output_folder your_path/transformer_cpc1/asrN
```

- Put all the trained ASR models in the same folder `asr_ensemble`. The eventual structure will look like:

```text
your_path
|
└───transformer_cpc1
     |  asr0/save/
     |  asr1/save/
     |  asr2/save/
     |  ...
     |  asrN/save/
     |
     └───asr_ensemble
            └───save
               |  CKPT+xxxxx/ (from asr0)
               |  CKPT+xxxxx/ (from asr1)
               |  CKPT+xxxxx/ (from asr2)
               |  ...
               |  CKPT+xxxxx/ (from asrN)
               |  5000_unigram.model
               |  lm.ckpt
               |  lm_model.ckpt
               |  tokenizer.ckpt
```

- Specify `output_folder` in transformer_cpc1.yaml as `your_path/e029/transformer_cpc1/asr_ensemble`. Specify `n_ensembles`  in transformer_cpc1.yaml as the number of ASR models.

### Infer uncertainty

This part generates the uncertainties, including the confidence and negative entropy.

- Run `python infer.py`. Four json files `dev_conf.json`, `dev_negent.json`, `test_conf.json`, `test_negent.json` will be generated in your `exp_folder`.

### Evaluation

Same as e032_sheffield:

This part optimize a logistic fitting function with the dev set similarities and dev set labels, and applies the fitting function to the test set similarities for scaled predicted test set prediction. And the evaluation results of RMSE, Std, NCC and KT will be computed and stored in the `results.json`.

- Run `python evaluate.py`

## Citation

If you use this code for your research, please cite:

```text
@inproceedings{tu2022unsupervised,
  title={Unsupervised Uncertainty Measures of Automatic Speech Recognition for Non-intrusive Speech Intelligibility Prediction},
  author={Tu, Zehai and Ma, Ning and Barker, Jon},
  booktitle={INTERSPEECH 2022},
  year={2022}
}
```
