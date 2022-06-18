### Requirements:
```
torch==1.10.0
torchaudio==0.10.0
speechbrain==0.5.9
fastdtw==0.3.4
```

### Prepare ASR data
To train the ASR model and generate the hidden representations from it, the CPC1 data needs to be processed first. This part of code will (1) split train and dev set; (2) run MSBG hearing loss simulation to all signals in `clarity_data/HA_output`; (3) resample signals to 16kHz and generate csv files for SpeechBrain ASR model training.
1. Download `clarity_CPC1_data.v1_1.tgz` and `clarity_CPC1_data.test.v1.tgz`, untar them into `clarity_CPC1_data_train` and `clarity_CPC1_data_test`, respectively.
2. Specify `root` in config.yaml. Both `clarity_CPC1_data_train` and `clarity_CPC1_data_test` should be in your root folder. You could also specify your own `exp_folder`.
3. Run `python prepare_data.py`
4. `cpc1_asr_data` which contains the processed train & test CPC1 data and their csv files, and `data_split` which contains the train set and dev set scenes, will appear in your `exp_folder`.

### Train ASR
1. Create your `transformer_cpc1` folder to save ASR models & results
2. Specify output_folder & data_folder in transformer_cpc1.yaml:
`output_folder: !ref your_path/transformer_cpc1` &
`data_folder: !ref your_path/exp_folder/e032/cpc1_asr_data  # for closed-set`
OR
`data_folder: !ref your_path/exp_folder/e032/cpc1_asr_data_indep  # for open-set`
3. Download the `save` folder (i.e. ASR transformer checkpoint) from: https://drive.google.com/drive/folders/1ZudxqMWb8VNCJKvY2Ws5oNY3WI1To0I7, and place it under your `transformer_cpc1` folder
4. Run `python train_asr.py transformer_cpc1.yaml`
5. The trained ASR model checkpoint will appear in `your_path/transformer_cpc1/save`