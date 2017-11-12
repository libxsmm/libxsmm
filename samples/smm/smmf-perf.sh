#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
ECHO=$(which echo)
FILE=${HERE}/smmf-perf.txt

RUNS="2_2_2 4_4_4 4_6_9 5_5_5 5_5_13 5_13_5 5_13_13 6_6_6 8_8_8 10_10_10 12_12_12 13_5_5 13_5_7 13_5_13 13_13_5 13_13_13 13_13_26 \
  13_26_13 13_26_26 14_14_14 16_16_16 18_18_18 20_20_20 23_23_23 24_3_36 24_24_24 26_13_13 26_13_26 26_26_13 26_26_26 32_32_32 \
  40_40_40 48_48_48 56_56_56 64_64_64 72_72_72 80_80_80 88_88_88 96_96_96 104_104_104 112_112_112 120_120_120 128_128_128"

if [ "" != "$1" ]; then
  FILE=$1
  shift
fi
cat /dev/null > ${FILE}

NRUN=1
NMAX=$(${ECHO} ${RUNS} | wc -w)
for RUN in ${RUNS} ; do
  MVALUE=$(${ECHO} ${RUN} | cut -d_ -f1)
  NVALUE=$(${ECHO} ${RUN} | cut -d_ -f2)
  KVALUE=$(${ECHO} ${RUN} | cut -d_ -f3)
  >&2 ${ECHO} -n "${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})... "
  ERROR=$({ CHECK=1 ${HERE}/smm.sh ${MVALUE} ${NVALUE} ${KVALUE} $* >> ${FILE}; } 2>&1)
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    ${ECHO} "FAILED(${RESULT}) ${ERROR}"
    exit 1
  else
    ${ECHO} "OK ${ERROR}"
  fi
  ${ECHO} >> ${FILE}
  NRUN=$((NRUN+1))
done

