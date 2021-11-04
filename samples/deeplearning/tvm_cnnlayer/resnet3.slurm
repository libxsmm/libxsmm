#!/usr/bin/env bash
#SBATCH --job-name resnet3
#SBATCH --time 1-00:00:00
#SBATCH -N 1
#SBATCH -c 112
#SBATCH --output resnet3.out
#SBATCH --partition clx
#SBATCH --mail-type=END,FAIL      # notifications for job done & fail
#SBATCH --mail-user=anand.venkat@intel.com # send-to address

export KMP_AFFINITY=granularity=fine,compact,1,28
export OMP_NUM_THREADS=28
export TVM_NUM_THREADS=28
LD_PRELOAD=./libxsmm_wrapper/libxsmm_wrapper.so srun python -u mb1_tuned_latest.py -d resnet3
