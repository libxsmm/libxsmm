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

OP=0
OPRED=1
OPORDER=1
REGVECIN=1
IMPLICITIDX=1
OPARG=0
USE_BF16=0

N_IDX=42

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LDI=`echo ${i} | awk -F"_" '{print $3}'`
  LDO=`echo ${i} | awk -F"_" '{print $4}'`
  echo ${M} ${N} ${LDI} ${LDI}
  for IDXTYPE in 0 1
  do
    for SCALE in 0 1
    do
      for EQLD in 0 1
      do
        N_ADJ=$((${N} + ${N_IDX}))
        let LDI_ADJ=(1-${EQLD})*100+${LDI}*${EQLD}
        echo ${M} ${N_ADJ} ${N} ${LDI_ADJ}
        ./eltwise_opreduce_idxvecs ${M} ${N_ADJ} ${N} ${LDI_ADJ} ${OP} ${OPORDER} ${SCALE} ${OPRED} ${REGVECIN} ${IMPLICITIDX} ${OPARG} ${IDXTYPE} 0 ${USE_BF16}
      done
    done
  done

done

rm ${TESTFILE1}
