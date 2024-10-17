#!/usr/bin/env bash

source setup_eqn_tpp_prec_list.sh equation_gather_bcstmul_add

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
mrange = range(16,513,32)
colsrange = range(1024,16537,512)
f1 = open("${TESTFILE1}", "w+")
for m in mrange:
    for n in colsrange:
        line = str(m) + '_' + str(n) + '\n'
        f1.write(line)
f1.close()
END

for i in $(cat ${TESTFILE1}); do
  M=$(echo ${i} | awk -F"_" '{print $1}')
  COLS=$(echo ${i} | awk -F"_" '{print $2}')
  echo ${M} ${COLS}
  for PREC in ${EQN_PREC_LIST}; do
    if [ ! "${PEXEC_NI}" ]; then
      ${BIN_INSTR_TOOL} ./equation_gather_bcstmul_add ${COLS} ${M} 256 16 0
    else
      ${BIN_INSTR_TOOL} ./equation_gather_bcstmul_add ${COLS} ${M} 256 16 0 &
      PEXEC_PID+=("$!")
      if [ "0" != "$((PEXEC_NI<=${PEXEC_PID[@]}))" ]; then
        for PID in "${PEXEC_PID[@]}"; do wait "${PID}"; done; unset PEXEC_PID
      fi
    fi
  done
done

rm ${TESTFILE1}
