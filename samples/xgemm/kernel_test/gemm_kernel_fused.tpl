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
f1 = open("${TESTFILE1}", "w+")
f2 = open("${TESTFILE2}", "w+")
i = 0
for m in randnumm:
    for n in randnumn:
        for k in randnumk:
            line = str(m) + ' ' + str(n) + ' ' + str(k) + ' ' \
                 + str(m) + ' ' + str(k) + ' ' + str(m) + '\n'
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
BINARY_POSTOP=0
UNARY_POSTOP=0
AVNNI=0
BVNNI=0
CVNNI=0
PREC_A=$(echo ${PREC} | awk -F"_" '{print $1}')
PREC_B=$(echo ${PREC} | awk -F"_" '{print $2}')
PREC_COMP=$(echo ${PREC} | awk -F"_" '{print $3}')
PREC_C=$(echo ${PREC} | awk -F"_" '{print $4}')

${BIN_INSTR_TOOL} ./gemm_kernel_fused ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} nobr   1 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}

${BIN_INSTR_TOOL} ./gemm_kernel_fused ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} addrbr 5 0 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}

${BIN_INSTR_TOOL} ./gemm_kernel_fused ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} strdbr 5 0 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}

${BIN_INSTR_TOOL} ./gemm_kernel_fused ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} offsbr 5 0 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}

${BIN_INSTR_TOOL} ./gemm_kernel_fused ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} nobr   1 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}

${BIN_INSTR_TOOL} ./gemm_kernel_fused ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} addrbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}

${BIN_INSTR_TOOL} ./gemm_kernel_fused ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} offsbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}

${BIN_INSTR_TOOL} ./gemm_kernel_fused ${PREC_A} ${PREC_B} ${PREC_COMP} ${PREC_C} ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${AVNNI} ${BVNNI} ${CVNNI} strdbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP}

rm ${TESTFILE1}
rm ${TESTFILE2}
