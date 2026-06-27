#!/usr/bin/env bash

TESTFILE1=$(mktemp)

if [ -x "$(command -v python3)" ]; then
  PYTHON=$(command -v python3)
else
  PYTHON=$(command -v python)
fi

# Settings substituted by the generator:
#   FORMATVAL - quantization output format (mxfp4/mxbf8/nvfp4)
#   BLOCKVAL  - MX/NV block size (32 for mxfp4/mxbf8, 16 for nvfp4);
#               both M and the leading dimensions are kept multiples of it
#   LDMODEVAL - eqld (ldi == M) or gtld (ldi > M)
FORMAT="FORMATVAL"
BLOCK=BLOCKVAL
LDMODE="LDMODEVAL"

${PYTHON} << END
import random as rnd
import time as time
rnd.seed(time.time())
mults = rnd.sample(range(1,13), SAMPLESIZE)
nums  = rnd.sample(range(1,64), SAMPLESIZE)
f1 = open("${TESTFILE1}", "w+")
for k in mults:
    m = k * ${BLOCK}
    for n in nums:
        line = str(m) + '_' + str(n) + '\n'
        f1.write(line)
f1.close()
END

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  if [ "${LDMODE}" == "gtld" ]; then
    LDI=$((M + BLOCK))
  else
    LDI=${M}
  fi
  LDO=${M}
  echo ${M} ${N} ${LDI} ${LDO}
  ${BIN_INSTR_TOOL} ./eltwise_unary_quantization_to_${FORMAT} ${M} ${N} ${LDI} ${LDO}
done

rm ${TESTFILE1}
