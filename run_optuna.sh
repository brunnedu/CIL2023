#!/bin/bash

#SBATCH -n 1
#SBATCH -G 1
#SBATCH -t 00-24
#SBATCH --mem-per-cpu=8192
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cil

python optuna_optimization.py
