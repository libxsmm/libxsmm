#!/usr/bin/env bash

source setup_tpp_prec_list.sh binary_simple

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
f1 = open("${TESTFILE1}", "w+")
for m in randnum:
    for n in randnum:
        line = str(m) + '_' + str(n) + '_' \
             + str(m) + '_' + str(n) + '\n'
        f1.write(line)
f1.close()
END

BINARY_OP=5

for i in $(cat ${TESTFILE1}); do
  M=$(echo ${i} | awk -F"_" '{print $1}')
  N=$(echo ${i} | awk -F"_" '{print $2}')
  LDI=$(echo ${i} | awk -F"_" '{print $3}')
  LDO=$(echo ${i} | awk -F"_" '{print $4}')
  echo ${M} ${N} ${LDI} ${LDI}
  for PREC in ${PREC_LIST}; do
    PREC_IN0=$(echo ${PREC} | awk -F"_" '{print $1}')
    PREC_IN1=$(echo ${PREC} | awk -F"_" '{print $2}')
    PREC_OUT=$(echo ${PREC} | awk -F"_" '{print $3}')
    PREC_COMP=$(echo ${PREC} | awk -F"_" '{print $4}')
    for BCAST in 0 1 2 3 4 5 6; do
      if [ ! "${PEXEC_NI}" ]; then
        ./eltwise_binary_simple ${BINARY_OP} ${BCAST} ${PREC_IN0} ${PREC_IN1} ${PREC_COMP} ${PREC_OUT} ${M} ${N} ${LDI} ${LDI}
      else
        ./eltwise_binary_simple ${BINARY_OP} ${BCAST} ${PREC_IN0} ${PREC_IN1} ${PREC_COMP} ${PREC_OUT} ${M} ${N} ${LDI} ${LDI} &
        PEXEC_PID+=("$!")
        if [ "0" != "$((PEXEC_NI<=${PEXEC_PID[@]}))" ]; then
          for PID in "${PEXEC_PID[@]}"; do wait "${PID}"; done; unset PEXEC_PID
        fi
      fi
    done
  done
done

rm ${TESTFILE1}
