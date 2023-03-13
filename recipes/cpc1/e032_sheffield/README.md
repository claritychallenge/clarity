# E032 - Exploiting Hidden Representations from a DNN-based Speech Recogniser for Speech Intelligibility Prediction in Hearing-impaired Listeners

The implementation of ["Exploiting Hidden Representations from a DNN-based Speech Recogniser for Speech Intelligibility Prediction in Hearing-impaired Listeners"](https://arxiv.org/abs/2204.04287), accepted to InterSpeech 2022. For the 1st Clarity Prediction Challenge (CPC1) programme and details, please see [here](https://claritychallenge.github.io/clarity2022-workshop/programme.html).

Please note: This code provides the implementation of the LS(LibriSpeech) + CPC1 model, as shown in the third row of Table 2 in the paper. Since the generation of CLS data (LS train-clean-100 set added with noises from the training set in CEC1) and the training with CLS data are over-complicated, the scripts are not provided here. Anyway, the improvement with CLS is limited...

As only the training data is provided in CPC1, we split the training data into a `train set` and `dev set`. The train set is used for ASR training, and the dev set is used for optimizing the logistic fitting function.

## Run the scripts

### Requirements

```text
torch==1.10.0
torchaudio==0.10.0
speechbrain==0.5.9
fastdtw==0.3.4
```

You probably need a GPU as well...

### Prepare ASR data

To train the ASR model and generate the hidden representations from it, the CPC1 data needs to be processed first. This part of code will (1) randomly split train and dev set; (2) run MSBG hearing loss simulation to all signals in `clarity_data/HA_output`; (3) resample signals to 16kHz and generate csv files for SpeechBrain ASR model training.

1. Download `clarity_CPC1_data.v1_1.tgz` and `clarity_CPC1_data.test.v1.tgz`, untar them into `clarity_CPC1_data_train` and `clarity_CPC1_data_test`, respectively. See recipes/cpc1/baseline/README.
2. Specify `root` in config.yaml. Both `clarity_CPC1_data_train` and `clarity_CPC1_data_test` should be in your root folder. You could also specify your own `exp_folder (e032 by default)`.
3. Specify `cpc1_track` as 'open' or 'closed'.
4. Run `python prepare_data.py` (Note, same as data preparation for the `e029_sheffield recipe`).
5. `cpc1_asr_data` which contains the processed train & test CPC1 data and their csv files, and `data_split` which contains the train set and dev set scenes, will appear in your `exp_folder (e032 by default)`.

### Train ASR

1. Create your `transformer_cpc1` folder to save ASR models & results
2. Specify output_folder & data_folder in transformer_cpc1.yaml:
`output_folder: !ref your_path/transformer_cpc1` &
`data_folder: !ref your_path/e032/cpc1_asr_data  # for closed-set`
OR
`data_folder: !ref your_path/e032/cpc1_asr_data_indep  # for open-set`
3. Download the `save` folder (i.e. ASR transformer checkpoint) from: <https://drive.google.com/drive/folders/1ZudxqMWb8VNCJKvY2Ws5oNY3WI1To0I7>, and place it under your `transformer_cpc1` folder
4. Run `python train_asr.py transformer_cpc1.yaml`
5. The trained ASR model checkpoint will appear in `your_path/transformer_cpc1/save`

### Infer hidden representation similarity

This part generates the similarities of encoder representations and decoder representations between the MSBG processed signals and reference signals.

1. Run `python infer.py`. Four json files `dev_dec_similarity.json`, `dev_enc_similarity.json`, `test_dec_similarity.json`, `test_enc_similarity.json` will be generated in you `exp_folder`.

### Evaluation

This part optimize a logistic fitting function with the dev set similarities and dev set labels, and applies the fitting function to the test set similarities for scaled predicted test set prediction. And the evaluation results of RMSE, Std, NCC and KT will be computed and stored in the `results.json`.

1. Run `python evaluate.py`

## Citation

If you use this code for your research, please cite:

```text
@inproceedings{tu2022exploiting,
  title={Exploiting Hidden Representations from a DNN-based Speech Recogniser for Speech Intelligibility Prediction in Hearing-impaired Listeners},
  author={Tu, Zehai and Ma, Ning and Barker, Jon},
  booktitle={INTERSPEECH 2022},
  year={2022}
}
```
