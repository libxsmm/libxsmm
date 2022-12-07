#!/usr/bin/env bash

source setup_tpp_prec_list.sh unary_no_simple



TESTFILE1=$(mktemp)

if [ -x "$(command -v python3)" ]; then
  PYTHON=$(command -v python3)
else
  PYTHON=$(command -v python)
fi

if [ $# -ne 1 ]
then
  MAXM=64
else
  MAXM=$1
fi

${PYTHON} << END
import random as rnd
import time as time
rnd.seed(time.time())
randnum = rnd.sample(range(1,${MAXM}), 18)
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
  for PREC in ${PREC_LIST}
  do
    PREC_IN=`echo ${PREC} | awk -F"_" '{print $1}'`
    PREC_OUT=`echo ${PREC} | awk -F"_" '{print $2}'`
    PREC_COMP=`echo ${PREC} | awk -F"_" '{print $3}'`
    ./eltwise_unary_dropout F 0 ${PREC_IN} ${PREC_OUT} ${M} ${N} ${LDI} ${LDI}
    ./eltwise_unary_dropout F 1 ${PREC_IN} ${PREC_OUT} ${M} ${N} ${LDI} ${LDI}
    ./eltwise_unary_dropout B 1 ${PREC_IN} ${PREC_OUT} ${M} ${N} ${LDI} ${LDI}
  done
done

rm ${TESTFILE1}
