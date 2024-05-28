
## Non-Causal
python train.py \
    --exp_dir /path/to/save/exps \
    --batch_size 4 \
    --aggregate 2 \
    --lr 0.0005 \
    --root /path/to/MUSDB18 \
    --sample_rate 44100 \
    --segment 5.0 \
    --samples_per_track 64

## Causal
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
