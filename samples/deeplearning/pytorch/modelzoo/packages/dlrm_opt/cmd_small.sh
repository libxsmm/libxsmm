#!/bin/bash
if [ "x$MPI_LOCALRANKID" == "x" ] ; then
  NUMANODE=0
else
  NUMANODE=$MPI_LOCALRANKID
fi

numactl -m $NUMANODE python -u dlrm_s_pytorch.py --mini-batch-size=2048 --test-mini-batch-size=16384 --test-num-workers=0 --num-batches=100 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=10 --print-time $@
# --enable-profiling


