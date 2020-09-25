#!/usr/bin/bash

target_sparsity=(0.5 0.8)

for TARGET_SPARSITY in ${target_sparsity[*]}; do
    sbatch ./prune_resnet.sh ${TARGET_SPARSITY}
done;
