#!/bin/bash

#SBATCH -n 1
#SBATCH -G 1
#SBATCH --gres=gpumem:16384m
#SBATCH -t 00-24
#SBATCH --mem-per-cpu=8192
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cil

python main.py train
