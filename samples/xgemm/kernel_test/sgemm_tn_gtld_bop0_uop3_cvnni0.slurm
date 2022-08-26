#!/usr/bin/env bash

if [[ ${LIBXSMM_TARGET} == "snb" ]]; then
  exit 0;
fi
if [[ ${LIBXSMM_TARGET} == "wsm" ]]; then
  exit 0;
fi

TESTFILE1=$(mktemp -p .)
TESTFILE2=$(mktemp -p .)

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
f1 = open("${TESTFILE1}", "w+")
f2 = open("${TESTFILE2}", "w+")
i = 0
for m in randnum:
    for n in randnum:
        for k in randnum:
            line = str(m) + ' ' + str(n) + ' ' + str(k) + ' 100 100 100\n'
            if 0 == (i % 2):
                f1.write(line)
            else:
                f2.write(line)
            i = i + 1
f1.close()
f2.close()
END

PREC="F32"
TRA=1
TRB=0
BINARY_POSTOP=0
UNARY_POSTOP=3
CVNNI=0

./kernel_fused ${TESTFILE1} 1 1 0 0 ${TRA} ${TRB} ${PREC} nobr   1 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}

./kernel_fused ${TESTFILE2} 1 1 0 0 ${TRA} ${TRB} ${PREC} nobr   1 1 1 1 0 ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}



#./kernel ${TESTFILE1} 1 1 0 0 1 0 F32 addrbr 5 0 1 0

#./kernel ${TESTFILE1} 1 1 0 0 1 0 F32 offsbr 5 0 1 0

#./kernel ${TESTFILE1} 1 1 0 0 1 0 F32 strdbr 5 0 1 0

#./kernel ${TESTFILE2} 1 1 0 0 1 0 F32 addrbr 5 1 1 0

#./kernel ${TESTFILE2} 1 1 0 0 1 0 F32 offsbr 5 1 1 0

#./kernel ${TESTFILE2} 1 1 0 0 1 0 F32 strdbr 5 1 1 0

rm ${TESTFILE1}
rm ${TESTFILE2}
