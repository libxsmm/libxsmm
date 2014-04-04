#!/bin/sh

SIZE=24

if [ "$1" != "" ] ; then
  SIZE=$1
fi


python scripts/generate_inc.py $SIZE $SIZE $SIZE  > include/micsmm.h
#python scripts/generate_lib.py 1 $SIZE > src/micsmm.c
#python scripts/generate_ben.py 4 32 > src/micsmmbench.c
rm src/gemms/*.cpp
rm src/gemms/*.c
rm src/gemms/*.o
rm src/gemms/*.a
#echo "#include <immintrin.h>" >> src/gemms/igemm.c
#echo "#include <micsmmmisc.h>" >> src/gemms/igemm.c
#echo "#include <mkl.h>" >> src/gemms/igemm.c

for m in `seq 1 $SIZE`; do
  for n in `seq 1 $SIZE`; do
    for k in `seq 1 $SIZE`; do
      echo $m $n $k
      #python scripts/generate_singleigemm.py $m $k $n > src/gemms/igemm_"$m"_"$k"_"$n".c
      python scripts/generate_singleigemm.py $m $n $k>> src/gemms/smm_dnn_"$m"_"$n"_"$k".c
    done
  done
done

python scripts/generate_switchingfunction.py $SIZE $SIZE $SIZE  >> src/gemms/smm.c

cd src/gemms
#icc  -c -I ../include/ micsmm.c
make -j 16
cd ../..
cd tests
icc -offload-attribute-target=mic -mkl=sequential  -I ../include/  -c benchmark.cpp
icc -offload-attribute-target=mic -mkl=sequential benchmark.o ../src/gemms/micsmm.a
cd ..
