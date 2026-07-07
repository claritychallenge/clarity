#!/bin/bash
#SBATCH --export=PATH,LD_LIBRARY_PATH
#SBATCH --time=96:00:00
#SBATCH --comment=B-Clip2
#SBATCH --job-name=B-Clip2
#SBATCH --mem=50GB
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

set -e   # abort on any error

module load libsndfile/1.2.2-GCCcore-12.3.0
source .venv/bin/activate

# == Parameters ==============================================================
backbone=whisper-large-v3
layers="enc[32]"
batch=32
lr=1e-4
layers_str=$(echo "$layers" | tr -d '[],' | tr ' ' '-')
dataset=clip2

# Extract encoder layer list from "enc[X,Y,...]" format → "[X,Y,...]"
enc_layers="${layers#enc}"


epochs_stage1=50                      # Max epoch stage 1 training - mono
epochs_stage2=50                      # Max epoch stage 2 training - better-ear
es_patience=10                        # Early stopping patience (in epochs)
lr_patience=5                         # Learning rate scheduler patience (in epochs)
finetune_epoch=5                      # Epochs at when finetuning whisper encoder layers (stage 1)
finetune_epoch_stage2=8               # Epochs at when finetuning whisper encoder layers (stage 2)
finetune_layers=2                     # Number of whisper encoder layers to finetune (from the top)
data_root=../cadenza_data/${dataset}  # Path to dataset root dir
workers=${SLURM_CPUS_PER_TASK:-4}     # Number of workers for data loading (default: 4, or SLURM_CPUS_PER_TASK if set)

# -- Learning rates ------------------------------------------------------------
lr_stage2=$(python3 -c "print(float('$lr') / 10)")
lr_finetune=$(python3 -c "print(float('$lr') / 1000)")
lr_finetune_stage2=$(python3 -c "print(float('$lr') / 10000)")

# -- Output dirs ---------------------------------------------------------------
output=output/${dataset}/${backbone}-${layers_str}-batch${batch}-lr${lr}
stage1=$output/stage1
stage2=$output/stage2

echo "============================================================"
echo "  dataset               : $dataset"
echo "  backbone              : $backbone"
echo "  enc_layers            : $enc_layers"
echo "  batch                 : $batch"
echo "  epochs stage1/2       : $epochs_stage1 / $epochs_stage2"
echo "  lr head  stage1/2     : $lr / $lr_stage2"
echo "  lr finetune stage1/2  : $lr_finetune / $lr_finetune_stage2"
echo "  finetune_epoch s1/s2  : $finetune_epoch / $finetune_epoch_stage2"
echo "  finetune_layers       : $finetune_layers"
echo "  es_patience           : $es_patience"
echo "  lr_patience           : $lr_patience"
echo "  stage1                : $stage1"
echo "  stage2                : $stage2"
echo "============================================================"

# -- Stage 1 — mono pretraining ------------------------------------------------
echo "Starting Stage 1 (mono)..."
python train.py \
    model.backbone=$backbone \
    "model.use_encoder_layers=$enc_layers" \
    "hydra.run.dir=$stage1" \
    data.batch_size=$batch \
    train.epochs=$epochs_stage1 \
    train.lr=$lr \
    train.finetune.epoch=$finetune_epoch \
    train.finetune.layers=$finetune_layers \
    train.finetune.lr=$lr_finetune \
    data.root_path=$data_root \
    train.output_dir=$stage1 \
    data.num_workers=$workers \
    train.better_ear=false \
    train.early_stopping.patience=$es_patience \
    train.lr_patience=$lr_patience \
    || { echo "Stage 1 training failed."; exit 1; }

# -- Stage 2 — better-ear fine-tuning -----------------------------------------
echo "Starting Stage 2 (better-ear)..."
python train.py \
    model.backbone=$backbone \
    "model.use_encoder_layers=$enc_layers" \
    "hydra.run.dir=$stage2" \
    data.batch_size=$batch \
    train.epochs=$epochs_stage2 \
    train.lr=$lr_stage2 \
    train.finetune.epoch=$finetune_epoch_stage2 \
    train.finetune.layers=$finetune_layers \
    train.finetune.lr=$lr_finetune_stage2 \
    data.root_path=$data_root \
    train.output_dir=$stage2 \
    train.resume_from=$stage1/best_model.pt \
    data.num_workers=$workers \
    train.better_ear=true \
    train.early_stopping.patience=$es_patience \
    train.lr_patience=$lr_patience \
    || { echo "Stage 2 training failed."; exit 1; }

# -- Inference ----------------------------------------------------------------
echo "Inferring Stage 1 (mono)..."
python test.py \
    "hydra.run.dir=$stage1" \
    "hydra.job.name=valid_mono.log" \
    test.checkpoint=$stage1/best_model.pt \
    test.output_dir=$stage1/valid \
    data.root_path=$data_root \
    test.split=valid \
    test.better_ear=false \
    data.num_workers=$workers \
    test.inference_only=true \
    || { echo "Stage 1 eval failed."; exit 1; }

echo "Inferring Stage 2 (mono)..."
python test.py \
    "hydra.run.dir=$stage2" \
    "hydra.job.name=valid_mono.log" \
    test.checkpoint=$stage2/best_model.pt \
    test.output_dir=$stage2/valid_mono \
    data.root_path=$data_root \
    test.split=valid \
    test.better_ear=false \
    data.num_workers=$workers \
    test.inference_only=true \
    || { echo "Stage 2 mono eval failed."; exit 1; }

echo "Inferring Stage 2 (better-ear)..."
python test.py \
    "hydra.run.dir=$stage2" \
    "hydra.job.name=valid_be.log" \
    test.checkpoint=$stage2/best_model.pt \
    test.output_dir=$stage2/valid_better_ear \
    data.root_path=$data_root \
    test.split=valid \
    test.better_ear=true \
    data.num_workers=$workers \
    test.inference_only=true \
    || { echo "Stage 2 better-ear eval failed."; exit 1; }
