#!/bin/bash

ITERS=10

for m in `seq 1 1 100`
do
  for n in `seq 1 1 100`
  do
    for k in `seq 1 1 100`
    do
      taskset -c 4 ./kernel $m $n $k 100 100 100 1 1 0 0 0 0 nopf DP $ITERS
      taskset -c 4 ./kernel $m $n $k 100 100 100 1 1 0 0 0 1 nopf DP $ITERS
      taskset -c 4 ./kernel $m $n $k $m  $k  $m  1 1 0 0 0 0 nopf DP $ITERS
      taskset -c 4 ./kernel $m $n $k $m  $n  $m  1 1 0 0 0 1 nopf DP $ITERS
    done
  done
done
