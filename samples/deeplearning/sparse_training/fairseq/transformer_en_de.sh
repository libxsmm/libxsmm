#!/bin/bash


#SBATCH -J nmt_en_de
#SBATCH --partition=nv-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=23:59:00
#SBATCH --output=%j.out

source ~/.bashrc
conda activate fairseq

CMD="python fairseq_cli/train.py data-bin/wmt16_en_de_bpe32k --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 12000 --arch transformer --save-dir checkpoints/transformer --distributed-world-size=1 --fp16 --max-update 100000 --save-interval-updates 10000 --tensorboard-logdir tb_output"

echo "#### Running python command #### "
echo $CMD
eval $CMD
