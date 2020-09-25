#!/usr/bin/bash

target_sparsity=(0.5 0.8)
prune_type=('magnitude' 'random')
embedding=(0 1)

for PRUNE_TYPE in ${prune_type[*]}; do
for TARGET_SPARSITY in ${target_sparsity[*]}; do
for EMB in ${embedding[*]}; do
    sbatch ./prune_en_de.sh ${TARGET_SPARSITY} ${PRUNE_TYPE} ${EMB}
    #CUDA_VISIBLE_DEVICES=0,1 ./prune_en_de.sh ${TARGET_SPARSITY} ${PRUNE_TYPE} ${EMB}
done;
done;
done;
