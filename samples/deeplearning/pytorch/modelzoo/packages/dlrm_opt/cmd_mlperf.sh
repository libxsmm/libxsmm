#!/bin/bash
data_path=/cold_storage/ml_datasets/dlrm/Criteo_Terabyte/preprocessed
#data_path=/cold_storage/ml_datasets/dlrm/terabyte_smallbindata

if [ "x$MPI_LOCALRANKID" == "x" ] ; then
  NUMANODE=0
else
  NUMANODE=$MPI_LOCALRANKID
fi

CVG_ARGS_32K=" --mini-batch-size=8192  --learning-rate=18.0 --print-freq=128 --test-freq=0 --test-mini-batch-size=65536 --num-warmup-iters=8000 --num-decay-iters=30000 --decay-start-iter=70000 "
CVG_ARGS=$CVG_ARGS_32K

# --arch-embedding-dtype=bf16 --arch-mlp-dtype=bf16 --arch-mlp-impl=xsmm

numactl -m $NUMANODE python -u dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=10000000  --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/terabyte_processed.npz --loss-function=bce --round-targets=True --print-time --test-num-workers=0 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --dist-backend=ccl $CVG_ARGS  --arch-embedding-dtype=fp32 --arch-mlp-dtype=fp32 --num-batches=2048 $@
#numactl -m $NUMANODE python -u dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=10000000  --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/terabyte_processed.npz --loss-function=bce --round-targets=True --print-time --test-num-workers=0 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --dist-backend=ccl $CVG_ARGS  --arch-embedding-dtype=fp32 --arch-mlp-dtype=fp32 --num-batches=2048 $@


