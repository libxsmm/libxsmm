#!/usr/bin/env bash

# args: iters N E M S P alpha

ITERS=100
S=1
alpha=1.16

if [ "x$1" != "x" ] ; then ITERS=$1; fi
if [ "x$2" != "x" ] ; then S=$2; fi
if [ "x$3" != "x" ] ; then alpha=$3; fi

BIN_FN=./main
NUMACTL_ARGS="numactl -m 0 "

for N in 2048 8192 32768 65536 ; do
  for E in 64 128 256 ; do
    for P in 1 25 50 ; do
      for M in 10000000 ; do
        echo "Running: $NUMACTL_ARGS $BIN_FN $ITERS $N $E $M $S $P $alpha"
        $NUMACTL_ARGS $BIN_FN $ITERS $N $E $M $S $P $alpha
      done
    done
  done
done

