#!/bin/bash

#SBATCH -J wmt16_en_de
#SBATCH --partition=nv-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=23:59:00

TARGET_SPARSITY=${1}
PRUNE_TYPE=${2}
PRUNE_EMBEDDING=${3}

NUM_TRAIN_STEPS=500000
#BATCH_SIZE=8192
BATCH_SIZE=4096
PRUNE_INTERVAL=1000
PRUNE_START_STEP=50000
PRUNE_END_STEP=250000

let NUM_PRUNING_STEPS=$(((PRUNE_END_STEP - PRUNE_START_STEP) / PRUNE_INTERVAL))

DATASET=wmt16_en_de_bpe32k
DATA_DIR=data-bin/$DATASET
EXP_NAME=prune_${NUM_TRAIN_STEPS}_${TARGET_SPARSITY}_${PRUNE_TYPE}_P_EMB_${PRUNE_EMBEDDING}
LOG_DIR=logs/$EXP_NAME

source ~/.bashrc
conda activate fairseq
mkdir -p checkpoints/$EXP_NAME
mkdir -p $LOG_DIR


CUDA_VISIBLE_DEVICES=0,1 python ./fairseq_cli/train.py \
  $DATA_DIR --arch transformer --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --dropout 0.3 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens $BATCH_SIZE --max-update $NUM_TRAIN_STEPS \
  --eval-bleu --no-epoch-checkpoints --ddp-backend=no_c10d \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --save-dir checkpoints/$EXP_NAME \
  --target_sparsity $TARGET_SPARSITY --pruning_interval $PRUNE_INTERVAL \
  --prune_start_step $PRUNE_START_STEP --num_pruning_steps $NUM_PRUNING_STEPS \
  --prune_type $PRUNE_TYPE --log-format json --tensorboard-logdir $LOG_DIR \
  --prune_embedding $PRUNE_EMBEDDING > $EXP_NAME.out
