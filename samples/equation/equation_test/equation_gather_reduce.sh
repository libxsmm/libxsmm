#!/usr/bin/env bash

source setup_eqn_tpp_prec_list.sh equation_gather_reduce

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

for i in $(cat ${TESTFILE1}); do
  M=$(echo ${i} | awk -F"_" '{print $1}')
  N=$(echo ${i} | awk -F"_" '{print $2}')
  LD=$(echo ${i} | awk -F"_" '{print $3}')
  echo ${M} ${N} ${LD}
  for PREC in ${EQN_PREC_LIST}; do
    for IDXTYPE in 0 1; do
      if [ ! "${PEXEC_NI}" ]; then
        ./equation_gather_reduce ${M} ${N} ${LD} ${PREC} ${IDXTYPE} 0
      else
        ./equation_gather_reduce ${M} ${N} ${LD} ${PREC} ${IDXTYPE} 0 &
        PEXEC_PID+=("$!")
        if [ "0" != "$((PEXEC_NI<=${PEXEC_PID[@]}))" ]; then
          for PID in "${PEXEC_PID[@]}"; do wait "${PID}"; done; unset PEXEC_PID
        fi
      fi
    done
  done
done

rm ${TESTFILE1}
