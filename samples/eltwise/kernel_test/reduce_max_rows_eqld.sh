#!/usr/bin/env bash

PREC_LIST="F32";

if [[ ${LIBXSMM_TARGET} == "clx" ]]; then
    PREC_LIST="F32 BF16 BF8";
fi
if [[ ${LIBXSMM_TARGET} == "avx512_vl256_clx" ]]; then
    PREC_LIST="F32 BF16 BF8";
fi
if [[ ${LIBXSMM_TARGET} == "snb" ]]; then
    exit 0;
fi
if [[ ${LIBXSMM_TARGET} == "wsm" ]]; then
    exit 0;
fi

TESTFILE1=$(mktemp)

if [ -x "$(command -v python3)" ]; then
  PYTHON=$(command -v python3)
else
  PYTHON=$(command -v python)
fi

${PYTHON} << END
import random as rnd
import time as time
rnd.seed(time.time())
randnum = rnd.sample(range(1,101), 18)
f1 = open("${TESTFILE1}", "w+")
for m in randnum:
    for n in randnum:
        line = str(m) + '_' + str(n) + '_' \
             + str(m) + '_' + str(n) + '\n'
        f1.write(line)
f1.close()
END

REDUCE_X=1
REDUCE_X2=0
REDUCE_ROWS=1
REDUCE_OP=1
N_IDX=0

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LDI=`echo ${i} | awk -F"_" '{print $3}'`
  LDO=`echo ${i} | awk -F"_" '{print $4}'`
  echo ${M} ${N} ${LDI} ${LDI}
  N_ADJ=$((${N} + ${N_IDX}))
  for PREC in ${PREC_LIST}
  do
    ./eltwise_unary_reduce ${M} ${N_ADJ} ${LDI} ${REDUCE_X} ${REDUCE_X2} ${REDUCE_ROWS} ${REDUCE_OP} ${PREC} ${N_IDX} 0
  done
done

rm ${TESTFILE1}
