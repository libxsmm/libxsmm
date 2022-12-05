#!/usr/bin/env bash

TESTFILE1=$(mktemp)

PYTHON=$(command -v python3)

${PYTHON} << END
import random as rnd
import time as time
rnd.seed(time.time())
randnumM = rnd.sample(range(1,101,1), 18)
randnum = rnd.sample(range(4,101,4), 18)
f1 = open("${TESTFILE1}", "w+")
for m in randnumM:
    for n in randnum:
        line = str(m) + '_' + str(n) + '_' \
             + str(m) + '_' + str(m) + '\n'
        f1.write(line)
f1.close()
END

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LDI=`echo ${i} | awk -F"_" '{print $3}'`
  LDO=`echo ${i} | awk -F"_" '{print $4}'`
  echo ${M} ${N} 100 100
  ./eltwise_unary_transform W BF8 ${M} ${N} 100 100
  ./eltwise_unary_transform W I8 ${M} ${N} 100 100
done

rm ${TESTFILE1}
