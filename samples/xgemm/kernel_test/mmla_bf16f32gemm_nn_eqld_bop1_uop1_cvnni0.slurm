#!/usr/bin/env bash





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
randnum = rnd.sample(range(2,101,2), 18)
randnumk = rnd.sample(range(4,101,4), 18)
f1 = open("${TESTFILE1}", "w+")
f2 = open("${TESTFILE2}", "w+")
i = 0
for m in randnum:
    for n in randnum:
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

PREC="BF16F32"
TRA=0
TRB=0
BINARY_POSTOP=1
UNARY_POSTOP=1
CVNNI=0

./kernel_fused ${TESTFILE1} 1 1 0 0 ${TRA} ${TRB} ${PREC} nobr   1 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./kernel_fused ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${PREC} nobr   1 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./kernel_fused ${TESTFILE1} 1 1 0 0 ${TRA} ${TRB} ${PREC} addrbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./kernel_fused ${TESTFILE1} 1 1 0 0 ${TRA} ${TRB} ${PREC} offsbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./kernel_fused ${TESTFILE1} 1 1 0 0 ${TRA} ${TRB} ${PREC} strdbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./kernel_fused ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${PREC} addrbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./kernel_fused ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${PREC} offsbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./kernel_fused ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${PREC} strdbr 5 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}



rm ${TESTFILE1}
rm ${TESTFILE2}
