#!/bin/bash
#SBATCH --account=clarity
#SBATCH --partition=clarity

module load Anaconda3/5.3.0
module load CUDA/10.2.89-GCC-8.3.0
source activate clarity

srun --export=ALL python render_additional_scenes.py 'render_starting_chunk=range(0, 1000, 200)' --multirun
