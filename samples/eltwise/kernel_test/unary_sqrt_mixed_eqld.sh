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

UNARY_OP=4

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LDI=`echo ${i} | awk -F"_" '{print $3}'`
  LDO=`echo ${i} | awk -F"_" '{print $4}'`
  echo ${M} ${N} ${LDI} ${LDI}
  for PREC_IN in 1 2 4
  do
    for PREC_COMP in 4
    do
      if [ ${PREC_IN} -ne 1 ];
      then
        for PREC_OUT in 2 4
        do
          for BCAST_IN in 0 1 2 3
          do
            ./eltwise_unary_simple ${UNARY_OP} ${BCAST_IN} ${PREC_IN} ${PREC_COMP} ${PREC_OUT} ${M} ${N} ${LDI} ${LDI}
          done
        done
      else
        for PREC_OUT in 1 4
        do
          for BCAST_IN in 0 1 2 3
          do
            ./eltwise_unary_simple ${UNARY_OP} ${BCAST_IN} ${PREC_IN} ${PREC_COMP} ${PREC_OUT} ${M} ${N} ${LDI} ${LDI}
          done
        done
      fi
    done
  done
done

rm ${TESTFILE1}
