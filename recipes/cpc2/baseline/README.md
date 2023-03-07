# Baseline CPC2 code

## Computing the HASPI scores

```bash
python compute_haspi.py clarity_data_dir=<CPC2_data_path> exp_dir=.exp
```

```bash
python compute_haspi.py path.clarity_data_dir=/Users/jon/shared/data/clarity_CPC2_data path.exp_dir=./exp dataset=CEC2.train.1 compute_haspi.n_batches=1 compute_haspi.batch=1 results_file=results_1
```

```bash
python predict.py path.clarity_data_dir=/Users/jon/shared/data/clarity_CPC2_data dataset=CEC2.train.1 +haspi_score_file=../results/results_1
```

```bash
python evaluate.py path.clarity_data_dir=/Users/jon/shared/data/clarity_CPC2_data dataset=CEC2.train.1
```
