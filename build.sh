#!/bin/bash

DIRINC=include
DIRSRC=src/knc
SIZE=8

if [ "$1" != "" ] ; then
  SIZE=$1
fi


python scripts/generate_inc.py $SIZE $SIZE $SIZE > $DIRINC/xsmmknc.h
#python scripts/generate_lib.py 1 $SIZE > src/xsmmknc.c
#python scripts/generate_ben.py 4 32 > src/xsmmbench.c
rm $DIRSRC/*.cpp
rm $DIRSRC/*.c
rm $DIRSRC/*.o
rm $DIRSRC/*.a
#echo "#include <immintrin.h>" >> $DIRSRC/igemm.c
#echo "#include <xsmmkncmisc.h>" >> $DIRSRC/igemm.c
#echo "#include <mkl.h>" >> $DIRSRC/igemm.c

for m in $(seq 1 $SIZE); do
  for n in $(seq 1 $SIZE); do
    for k in $(seq 1 $SIZE); do
      echo $m $n $k
      python scripts/generate_singleigemm.py $m $n $k >> $DIRSRC/xsmm_dnn_"$m"_"$n"_"$k".c
    done
  done
done

python scripts/generate_switchingfunction.py $SIZE $SIZE $SIZE >> $DIRSRC/xsmm.c

cd $DIRSRC
make -j 16
cd ../..
cd tests
icc -offload-attribute-target=mic -mkl=sequential -I ../$DIRINC -c benchmark.cpp
icc -offload-attribute-target=mic -mkl=sequential benchmark.o ../$DIRSRC/libxsmmknc.a
cd ..
