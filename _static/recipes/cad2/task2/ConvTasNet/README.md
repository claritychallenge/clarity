# CAD2-TASK2 target instrument/accompaniment separation model

This recipe contains the necessary content to replicate the separation models used in CAD2-Task1.

- The system is based on Asteroid Source Separation system.
- ConvTasNet implementation is based on stereo adaptation by Alexandre Defossez <https://github.com/facebookresearch/demucs/blob/v1/demucs/tasnet.py>

You can replicate the Causal and Non-Causal model by running:

- **To replicate the Non-Causal model**

```bash
python train.py \
    --exp_dir /path/to/save/exps \
    --batch_size 4 \
    --aggregate 2 \
    --lr 0.0005 \
    --root /path/to/MUSDB18 \
    --sample_rate 44100 \
    --segment 5.0 \
    --samples_per_track 64
```

- **To replicate the Causal model**

```bash
python train.py \
    --exp_dir /path/to/save/exps \
    --batch_size 4 \
    --aggregate 1 \
    --lr 0.0005 \
    --root /path/to/MUSDB18 \
    --sample_rate 44100 \
    --segment 4.0 \
    --samples_per_track 64 \
    --causal True \
    --n_src 2 \
    --norm_type cLN
```
