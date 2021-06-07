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
randnum = rnd.sample(range(16,97,16), 6)
f1 = open("${TESTFILE1}", "w+")
for m in randnum:
    for n in randnum:
        line = str(m) + '_' + str(n) + '_' \
             + str(m) + '_' + str(n) + '\n'
        f1.write(line)
f1.close()
END

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LDI=`echo ${i} | awk -F"_" '{print $3}'`
  LDO=`echo ${i} | awk -F"_" '{print $4}'`
  echo ${M} ${N} ${LDI} ${LDI}
  for PREC_IN in 2 4
  do
    for PREC_OUT in 2 4
    do
      ./eltwise_unary_dropout F 1 ${PREC_IN} ${PREC_OUT} ${M} ${N} 112 112
      ./eltwise_unary_dropout B 1 ${PREC_IN} ${PREC_OUT} ${M} ${N} 112 112
    done
  done
done

rm ${TESTFILE1}
