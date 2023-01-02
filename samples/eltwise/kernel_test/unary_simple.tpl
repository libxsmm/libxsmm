#!/usr/bin/env bash

if [ -x "$(command -v python3)" ]; then
  PYTHON=$(command -v python3)
else
  PYTHON=$(command -v python)
fi

TESTFILE1=$(mktemp)

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

UNARY_OP=0
PREC=0

for i in $(cat ${TESTFILE1}); do
  M=$(echo ${i} | awk -F"_" '{print $1}')
  N=$(echo ${i} | awk -F"_" '{print $2}')
  LDI=$(echo ${i} | awk -F"_" '{print $3}')
  LDO=$(echo ${i} | awk -F"_" '{print $4}')
  echo ${M} ${N} ${LDI} ${LDI}
  PREC_IN=$(echo ${PREC} | awk -F"_" '{print $1}')
  PREC_OUT=$(echo ${PREC} | awk -F"_" '{print $2}')
  PREC_COMP=$(echo ${PREC} | awk -F"_" '{print $3}')
  for BCAST_IN in 0 1 2 3; do
    ./eltwise_unary_simple ${UNARY_OP} ${BCAST_IN} ${PREC_IN} ${PREC_COMP} ${PREC_OUT} ${M} ${N} ${LDI} ${LDO}
  done
done

rm ${TESTFILE1}
