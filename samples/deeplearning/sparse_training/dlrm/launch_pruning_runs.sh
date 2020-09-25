#!/usr/bin/bash

#target_sparsity=(0.5 0.6 0.8 0.95)
#target_sparsity=(0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89)
target_sparsity=(0.95)
#prune_type=('magnitude' 'random')
prune_type=('magnitude')

for PRUNE_TYPE in ${prune_type[*]}; do
for TARGET_SPARSITY in ${target_sparsity[*]}; do
    sbatch ./run_terabyte.sh ${TARGET_SPARSITY} ${PRUNE_TYPE}
done;
done;
