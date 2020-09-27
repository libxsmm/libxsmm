#!/bin/bash

source ~/.bashrc
conda activate fairseq

fairseq-generate data-bin/wmt17_en_de \
      --path checkpoints/checkpoint_best.pt \
      --batch-size 128 --beam 5 --remove-bpe #| tee /tmp/gen.out

