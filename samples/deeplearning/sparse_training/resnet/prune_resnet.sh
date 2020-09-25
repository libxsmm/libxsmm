#!/bin/bash

#SBATCH -J prune_resnet
#SBATCH --partition=nv-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=23:59:00

TARGET_SPARSITY=${1}
PRUNE_TYPE=${2}

source ~/.bashrc
conda activate fastai2

python train.py ${TARGET_SPARSITY}
