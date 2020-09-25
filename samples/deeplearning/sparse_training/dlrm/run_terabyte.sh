#!/bin/bash

#SBATCH -J dlrm_terabyte
#SBATCH --partition=clxtrb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=23:59:00

TARGET_SPARSITY=${1}
PRUNE_TYPE=${2}

source ~/.bashrc

conda activate dlrm
export LD_LIBRARY_PATH=~/anaconda3/envs/dlrm/lib:$LD_LIBRARY_PATH

#if [ "x$MPI_LOCALRANKID" == "x" ] || [ "$MPI_LOCALRANKID" == "0" ] ; then
#nfs_data_path=/scratch/ddkalamk/terabyte_bindata
#cp -r $nfs_data_path /local_scratch/
#fi
#echo "Copy done, run mem clean  `date`"
#numactl -m $MPI_LOCALRANKID ~/a.out 90
#echo "mem clean done  `date`"
data_path=/scratch/ddkalamk/terabyte_bindata
#data_path=/local_scratch/terabyte_bindata
#data_path=/scratch/ddkalamk/terabyte_smallbindata
#cp -r $data_path /local_scratch/
#data_path=/local_scratch/terabyte_smallbindata

#data_path=/scratch/ddkalamk/terabyte_smallbindata
#python -u dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=$data_path/day --processed-data-file=$data_path/processed.npz --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2048 --print-time --test-freq=102400 --test-mini-batch-size=16384 --test-num-workers=16 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle

#unbuffer python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/processed.npz --loss-function=bce --round-targets=True --learning-rate=32.0 --mini-batch-size=65536 --print-freq=64 --print-time --test-freq=3200 --test-mini-batch-size=16384 --test-num-workers=0 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --dist-backend=ccl --num-warmup-iters=3400 --num-decay-iters=50000 --decay-start-iter=32000 # --arch-embedding-dtype=bf16 --arch-mlp-dtype=bf16 --arch-mlp-impl=mkldnn

#unbuffer python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/processed.npz --loss-function=bce --round-targets=True --learning-rate=24.0 --mini-batch-size=53248 --print-freq=80 --print-time --test-freq=4000 --test-mini-batch-size=16384 --test-num-workers=0 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --dist-backend=ccl --num-warmup-iters=9850 --num-decay-iters=30000 --decay-start-iter=59000
#unbuffer python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/processed.npz --loss-function=bce --round-targets=True --learning-rate=24.0 --mini-batch-size=65536 --print-freq=64 --print-time --test-freq=100000 --test-mini-batch-size=65536 --test-num-workers=0 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --dist-backend=ccl --num-warmup-iters=8000 --num-decay-iters=24000 --decay-start-iter=48000
#unbuffer python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/processed.npz --loss-function=bce --round-targets=True --learning-rate=24.0 --mini-batch-size=65536 --print-freq=64 --print-time --test-freq=100000 --test-mini-batch-size=65536 --test-num-workers=0 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --dist-backend=ccl --num-warmup-iters=8000 --num-decay-iters=24000 --decay-start-iter=48000
##unbuffer python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/processed.npz --loss-function=bce --round-targets=True --learning-rate=24.0 --mini-batch-size=65536 --print-freq=64 --print-time --test-freq=3200 --test-mini-batch-size=65536 --test-num-workers=0 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --dist-backend=ccl --num-warmup-iters=8000 --num-decay-iters=24000 --decay-start-iter=48000
#unbuffer python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/processed.npz --loss-function=bce --round-targets=True --learning-rate=32.0 --mini-batch-size=65536 --print-freq=64 --print-time --test-freq=3200 --test-mini-batch-size=16384 --test-num-workers=0 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --dist-backend=ccl --num-warmup-iters=3200 --num-decay-iters=40000 --decay-start-iter=32000 --arch-embedding-dtype=bf16 --arch-mlp-dtype=bf16 --arch-mlp-impl=xsmm
#unbuffer python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/processed.npz --loss-function=bce --round-targets=True --learning-rate=18.0 --mini-batch-size=32768 --print-freq=128 --print-time --test-freq=6400 --test-mini-batch-size=65536 --test-num-workers=0 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --dist-backend=ccl --num-warmup-iters=8000 --num-decay-iters=30000 --decay-start-iter=70000

#numactl --interleave=all python -u dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=18.0 --mini-batch-size=32768 --print-freq=128 --print-time --test-freq=6400 --test-mini-batch-size=65536 --test-num-workers=0 --memory-map --mlperf-logging --mlperf-bin-loader --dist-backend=ccl --num-warmup-iters=8000 --num-decay-iters=48000 --decay-start-iter=96000 --arch-mlp-impl=xsmm --target-sparsity=$TARGET_SPARSITY --prune-type=$PRUNE_TYPE

numactl --interleave=all python -u dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=18.0 --mini-batch-size=4096 --print-freq=128 --print-time --test-freq=6400 --test-mini-batch-size=65536 --test-num-workers=0 --memory-map --mlperf-logging --mlperf-bin-loader --dist-backend=ccl --num-warmup-iters=8000 --num-decay-iters=48000 --decay-start-iter=96000 --arch-mlp-impl=xsmm --target-sparsity=$TARGET_SPARSITY --prune-type=$PRUNE_TYPE
