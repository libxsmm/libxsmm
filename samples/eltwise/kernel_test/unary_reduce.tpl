#!/usr/bin/env bash

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
randnum = rnd.sample(range(1,101), SAMPLESIZE)
f1 = open("${TESTFILE1}", "w+")
for m in randnum:
    for n in randnum:
        line = str(m) + '_' + str(n) + '_' \
             + str(m) + '_' + str(m) + '\n'
        f1.write(line)
f1.close()
END

PREC=0
REDUCE_X=0
REDUCE_X2=0
REDUCE_ROWS=0
REDUCE_OP=0
N_IDX=0
USE_ACC=0
IDX_TYPE=0
RECORD_IDX=0

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LDI=`echo ${i} | awk -F"_" '{print $3}'`
  LDO=`echo ${i} | awk -F"_" '{print $4}'`
  echo ${M} ${N} ${LDI} ${LDI}
  N_ADJ=$((${N} + ${N_IDX}))
  ./eltwise_unary_reduce ${M} ${N_ADJ} ${LDI} ${REDUCE_X} ${REDUCE_X2} ${REDUCE_ROWS} ${REDUCE_OP} ${PREC} ${N_IDX} ${IDX_TYPE} ${RECORD_IDX} ${USE_ACC}
done

rm ${TESTFILE1}
