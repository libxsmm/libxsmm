#!/usr/bin/env bash
export LIBXSMM_ENV_VAR=1

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
randnumm = (16,32) 
randnumn = rnd.sample(range(NSTART,32,NSTEP), SAMPLESIZE)
randnumk = rnd.sample(range(KSTART,2048,KSTEP), SAMPLESIZE)
f1 = open("${TESTFILE1}", "w+")
for m in randnumm:
    for n in randnumn:
        for k in randnumk:
            line = str(m) + ' ' + str(n) + ' ' + str(k) + ' ' \
                 + str(m) + ' ' + str(k) + ' ' + str(m) + '\n'
            f1.write(line)
f1.close()
END

PREC=0
TRA=0
TRB=0
BINARY_POSTOP=0
UNARY_POSTOP=0
AVNNI=0
BVNNI=0
CVNNI=0
PREC_A=$(echo ${PREC} | awk -F"_" '{print $1}')
PREC_B=$(echo ${PREC} | awk -F"_" '{print $2}')
PREC_COMP=$(echo ${PREC} | awk -F"_" '{print $3}')
PREC_C=$(echo ${PREC} | awk -F"_" '{print $4}')

${BIN_INSTR_TOOL} ./gemm_kernel ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} spmm 0.4 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}
${BIN_INSTR_TOOL} ./gemm_kernel ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} spmm 0.8 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}

rm ${TESTFILE1}
