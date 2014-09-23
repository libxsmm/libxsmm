#!/bin/bash

DIRINC=include
DIRSRC=src/knc
DIRTST=tests
SIZE=8

if [ "$1" != "" ] ; then
  SIZE=$1
fi


python scripts/generate_inc.py $SIZE $SIZE $SIZE > $DIRINC/xsmmknc.h

rm $DIRSRC/*.cpp
rm $DIRSRC/*.c
rm $DIRSRC/*.o
rm $DIRSRC/*.a
for m in $(seq 1 $SIZE); do
  for n in $(seq 1 $SIZE); do
    for k in $(seq 1 $SIZE); do
      echo $m $n $k
      python scripts/generate_src.py $m $n $k >> $DIRSRC/xsmm_dnn_"$m"_"$n"_"$k".c
    done
  done
done

python scripts/generate_main.py $SIZE $SIZE $SIZE >> $DIRSRC/xsmm.c
python scripts/generate_ben.py 4 32 > $DIRTST/benchmark.c
