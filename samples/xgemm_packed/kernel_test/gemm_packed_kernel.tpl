#!/usr/bin/env bash
export LIBXSMM_ENV_VAR=1

TESTFILE1=$(mktemp)
TESTFILE2=$(mktemp)

if [ -x "$(command -v python3)" ]; then
  PYTHON=$(command -v python3)
else
  PYTHON=$(command -v python)
fi

${PYTHON} << END
import random as rnd
import time as time
rnd.seed(time.time())
randnumm = rnd.sample(range(MSTART,101,MSTEP), SAMPLESIZE)
randnumn = rnd.sample(range(NSTART,101,NSTEP), SAMPLESIZE)
randnumk = rnd.sample(range(KSTART,101,KSTEP), SAMPLESIZE)
randnumr = rnd.sample(range(RSTART,101,RSTEP), RSAMPLESIZE)
f1 = open("${TESTFILE1}", "w+")
f2 = open("${TESTFILE2}", "w+")
i = 0
for m in randnumm:
    for n in randnumn:
        for k in randnumk:
            for r in randnumr:
                line = str(m) + ' ' + str(n) + ' ' + str(k) + ' '\
                    + str(m) + ' ' + str(k) + ' ' + str(m) + ' ' + str(r) + '\n'
                if 0 == (i % 2):
                    f1.write(line)
                else:
                    f2.write(line)
                i = i + 1
f1.close()
f2.close()
END

PREC=0
TRA=0
TRB=0
PREC_A=$(echo ${PREC} | awk -F"_" '{print $1}')
PREC_B=$(echo ${PREC} | awk -F"_" '{print $2}')
PREC_COMP=$(echo ${PREC} | awk -F"_" '{print $3}')
PREC_C=$(echo ${PREC} | awk -F"_" '{print $4}')

${BIN_INSTR_TOOL} ./gemm_packed_kernel ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE1} 1 0 ${TRA} ${TRB} 1

${BIN_INSTR_TOOL} ./gemm_packed_kernel ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE2} 1 1 ${TRA} ${TRB} 1

rm ${TESTFILE1}
rm ${TESTFILE2}
