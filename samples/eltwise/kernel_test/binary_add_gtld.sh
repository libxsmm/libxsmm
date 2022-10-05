#!/usr/bin/env bash

PREC_LIST="F32_F32_F32_F32";

if [[ ${LIBXSMM_TARGET} == "clx" ]]; then
    PREC_LIST="F32_F32_F32_F32 F32_F32_BF16_F32 F32_BF16_F32_F32 F32_BF16_BF16_F32 BF16_F32_F32_F32 BF16_F32_BF16_F32 BF16_BF16_F32_F32 BF16_BF16_BF16_F32 F32_F32_F16_F32 F32_F16_F32_F32 F32_F16_F16_F32 F16_F32_F32_F32 F16_F32_F16_F32 F16_F16_F32_F32 F16_F16_F16_F32 F32_F32_BF8_F32 F32_BF8_F32_F32 F32_BF8_BF8_F32 BF8_F32_F32_F32 BF8_F32_BF8_F32 BF8_BF8_F32_F32 BF8_BF8_BF8_F32 F32_F32_HF8_F32 F32_HF8_F32_F32 F32_HF8_HF8_F32 HF8_F32_F32_F32 HF8_F32_HF8_F32 HF8_HF8_F32_F32 HF8_HF8_HF8_F32";
fi
if [[ ${LIBXSMM_TARGET} == "avx512_vl256_clx" ]]; then
    PREC_LIST="F32_F32_F32_F32 F32_F32_BF16_F32 F32_BF16_F32_F32 F32_BF16_BF16_F32 BF16_F32_F32_F32 BF16_F32_BF16_F32 BF16_BF16_F32_F32 BF16_BF16_BF16_F32 F32_F32_F16_F32 F32_F16_F32_F32 F32_F16_F16_F32 F16_F32_F32_F32 F16_F32_F16_F32 F16_F16_F32_F32 F16_F16_F16_F32 F32_F32_BF8_F32 F32_BF8_F32_F32 F32_BF8_BF8_F32 BF8_F32_F32_F32 BF8_F32_BF8_F32 BF8_BF8_F32_F32 BF8_BF8_BF8_F32 F32_F32_HF8_F32 F32_HF8_F32_F32 F32_HF8_HF8_F32 HF8_F32_F32_F32 HF8_F32_HF8_F32 HF8_HF8_F32_F32 HF8_HF8_HF8_F32";
fi
if [[ ${LIBXSMM_TARGET} == "snb" ]]; then
    exit 0;
fi
if [[ ${LIBXSMM_TARGET} == "wsm" ]]; then
    exit 0;
fi

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
randnum = rnd.sample(range(1,101), 18)
f1 = open("${TESTFILE1}", "w+")
for m in randnum:
    for n in randnum:
        line = str(m) + '_' + str(n) + '_' \
             + str(m) + '_' + str(n) + '\n'
        f1.write(line)
f1.close()
END

BINARY_OP=1

for i in `cat ${TESTFILE1}`
do
  M=`echo ${i} | awk -F"_" '{print $1}'`
  N=`echo ${i} | awk -F"_" '{print $2}'`
  LDI=`echo ${i} | awk -F"_" '{print $3}'`
  LDO=`echo ${i} | awk -F"_" '{print $4}'`
  echo ${M} ${N} 100 100
  for PREC in ${PREC_LIST}
  do
    PREC_IN0=`echo ${PREC} | awk -F"_" '{print $1}'`
    PREC_IN1=`echo ${PREC} | awk -F"_" '{print $2}'`
    PREC_OUT=`echo ${PREC} | awk -F"_" '{print $3}'`
    PREC_COMP=`echo ${PREC} | awk -F"_" '{print $4}'`
    for BCAST in 0 1 2 3 4 5 6
    do
      ./eltwise_binary_simple ${BINARY_OP} ${BCAST} ${PREC_IN0} ${PREC_IN1} ${PREC_COMP} ${PREC_OUT} ${M} ${N} 100 100
    done
  done
done

rm ${TESTFILE1}
