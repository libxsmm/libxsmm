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
CASES=0

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LDI=`echo ${i} | awk -F"_" '{print $3}'`
  LDO=`echo ${i} | awk -F"_" '{print $4}'`
  echo ${M} ${N} ${LDI} ${LDI}
  PREC_IN=`echo ${PREC} | awk -F"_" '{print $1}'`
  PREC_OUT=`echo ${PREC} | awk -F"_" '{print $2}'`
  PREC_COMP=`echo ${PREC} | awk -F"_" '{print $3}'`
  for RELU_OP in ${CASES}
  do
    ./eltwise_unary_relu ${RELU_OP} F 0 ${PREC_IN} ${PREC_COMP} ${PREC_OUT} ${M} ${N} ${LDI} ${LDO}
    ./eltwise_unary_relu ${RELU_OP} F 1 ${PREC_IN} ${PREC_COMP} ${PREC_OUT} ${M} ${N} ${LDI} ${LDO}
    ./eltwise_unary_relu ${RELU_OP} B 1 ${PREC_IN} ${PREC_COMP} ${PREC_OUT} ${M} ${N} ${LDI} ${LDO}
  done
done

rm ${TESTFILE1}
