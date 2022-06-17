## Requirements:
```
torch==1.10.0
torchaudio==0.10.0
speechbrain==0.5.9
```

## Train ASR
1. Create your `transformer_cpc1` folder to save ASR models & results
2. Specify output_folder & data_folder in transformer_cpc1.yaml:
```
output_folder: !ref your/transformer_cpc1
data_folder: !ref your/exp_folder/e032/cpc1_asr_data  # for closed-set
OR
data_folder: !ref your/exp_folder/e032/cpc1_asr_data_indep  # for open-set
```

3. Download the save folder (i.e. ASR transformer checkpoint) from: https://drive.google.com/drive/folders/1ZudxqMWb8VNCJKvY2Ws5oNY3WI1To0I7, and place it under your `transformer_cpc1` folder