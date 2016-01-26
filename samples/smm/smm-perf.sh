#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)

if [ "-2" = "$1" ]; then
  shift
  RUNS=" \
    2_2_2 \
    4_4_4 \
    6_6_6 \
    8_8_8 \
    10_10_10 \
    12_12_12 \
    14_14_14 \
    16_16_16 \
    18_18_18 \
    20_20_20"
else
  RUNS=" \
    5_5_5 \
    5_5_13 \
    5_13_5 \
    13_5_5 \
    13_13_5 \
    13_5_13 \
    5_13_13 \
    13_13_13 \
    13_13_26 \
    13_26_13 \
    26_13_13 \
    26_26_13 \
    26_13_26 \
    13_26_26 \
    23_23_23 \
    26_26_26 \
    32_32_32"
fi

cat /dev/null > smm-blas.txt
cat /dev/null > smm-dispatched.txt
cat /dev/null > smm-inlined.txt
cat /dev/null > smm-specialized.txt

NRUN=1
NMAX=$(echo ${RUNS} | wc -w)
for RUN in ${RUNS} ; do
  MVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f1)
  NVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f2)
  KVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f3)

  >&2 echo "Test ${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})"

  env LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ${HERE}/blas.sh        ${MVALUE} ${NVALUE} ${KVALUE} >> smm-blas.txt
  echo                                                                                        >> smm-blas.txt

  env LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ${HERE}/dispatched.sh  ${MVALUE} ${NVALUE} ${KVALUE} >> smm-dispatched.txt
  echo                                                                                        >> smm-dispatched.txt

  env LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ${HERE}/inlined.sh     ${MVALUE} ${NVALUE} ${KVALUE} >> smm-inlined.txt
  echo                                                                                        >> smm-inlined.txt

  env LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ${HERE}/specialized.sh ${MVALUE} ${NVALUE} ${KVALUE} >> smm-specialized.txt
  echo                                                                                        >> smm-specialized.txt

  NRUN=$((NRUN + 1))
done

