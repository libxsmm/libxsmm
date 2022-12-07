#!/usr/bin/env bash

source setup_eqn_tpp_prec_list.sh equation_split_sgd

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
ldoffset = [0] + rnd.sample(range(1,99), 3)
f1 = open("${TESTFILE1}", "w+")
for m in randnum:
    for n in randnum:
        for l in ldoffset:
          line = str(m) + '_' + str(n) + '_' \
               + str(l+m) + '\n'
          f1.write(line)
f1.close()
END

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LD=`echo ${i} | awk -F"_" '{print $3}'`
  echo ${M} ${N} ${LD}
  if [ ! "${PEXEC_NI}" ]; then
    ./equation_splitSGD ${M} ${N} ${LD} 0
  else
    ./equation_splitSGD ${M} ${N} ${LD} 0 &
    if [ "${NI}" ]; then NI=$((NI+1)); else NI=1; fi
    if [ "0" != "$((PEXEC_NI<=NI))" ]; then wait; unset NI; fi
  fi
done
rm ${TESTFILE1}
