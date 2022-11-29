#!/usr/bin/env bash

source setup_eqn_tpp_prec_list.sh equation_softmax

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
        for k in randnum:
          line = str(m) + '_' + str(n) + '_' \
               + str(k) + '\n'
          f1.write(line)
f1.close()
END

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  K=`echo ${i} | awk -F"_" '{print $3}'`
  echo ${M} ${N} ${K}
  for PREC in ${EQN_PREC_LIST}
  do
    ./equation_softmax ${M} ${N} ${K} ${PREC} 3 0
  done
done
rm ${TESTFILE1}