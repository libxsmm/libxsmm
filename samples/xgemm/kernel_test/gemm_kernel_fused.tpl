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
CVNNI=0

./gemm_kernel_fused ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${PREC} nobr   1 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./gemm_kernel_fused ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${PREC} addrbr 5 0 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./gemm_kernel_fused ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${PREC} strdbr 5 0 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./gemm_kernel_fused ${TESTFILE1} 1 0 0 0 ${TRA} ${TRB} ${PREC} offsbr 5 0 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./gemm_kernel_fused ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${PREC} nobr   1 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./gemm_kernel_fused ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${PREC} addrbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./gemm_kernel_fused ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${PREC} offsbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./gemm_kernel_fused ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${PREC} strdbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

rm ${TESTFILE1}
rm ${TESTFILE2}
