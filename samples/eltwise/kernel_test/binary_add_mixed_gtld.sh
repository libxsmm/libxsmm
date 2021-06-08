#!/usr/bin/env bash

TESTFILE1=$(mktemp -p .)

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

BINARY_OP=1

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LDI=`echo ${i} | awk -F"_" '{print $3}'`
  LDO=`echo ${i} | awk -F"_" '{print $4}'`
  echo ${M} ${N} ${LDI} ${LDI}
  for PREC_IN in 2 4
  do
    for PREC_COMP in 4
    do
      for PREC_OUT in 2 4
      do
        for BCAST in 0 1 2 3 4 5 6
        do
          ./eltwise_binary_simple ${BINARY_OP} ${BCAST} ${PREC_IN} ${PREC_COMP} ${PREC_OUT} ${M} ${N} 100 100
        done
      done
    done
  done
done

rm ${TESTFILE1}
