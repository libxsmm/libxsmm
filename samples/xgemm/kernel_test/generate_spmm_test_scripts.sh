#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=10
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

echo "#!/bin/bash" > ${HERE}/generate_spmm_test_scripts_gen_parallel.sh
echo "" >> ${HERE}/generate_spmm_test_scripts_gen_parallel.sh

for PREC in 'F32_F32_F32_F32' 'BF16_BF16_F32_F32' 'BF16_BF16_F32_BF16' 'BF8_BF16_F32_F32' 'BF8_BF16_F32_BF16' 'BF8_F16_F32_F32' 'BF8_F16_F32_F16' 'HF8_BF16_F32_F32' 'HF8_BF16_F32_BF16'; do
  sed "s/PRECDESC/${PREC}/g" ${HERE}/generate_spmm_test_scripts.tpl > ${HERE}/generate_spmm_test_scripts_gen_$PREC.sh
  chmod 755 ${HERE}/generate_spmm_test_scripts_gen_$PREC.sh
  echo "./generate_spmm_test_scripts_gen_$PREC.sh &" >> ${HERE}/generate_spmm_test_scripts_gen_parallel.sh
done

echo "wait" >> ${HERE}/generate_spmm_test_scripts_gen_parallel.sh
echo "sync" >> ${HERE}/generate_spmm_test_scripts_gen_parallel.sh
chmod 755 ${HERE}/generate_spmm_test_scripts_gen_parallel.sh

${HERE}/generate_spmm_test_scripts_gen_parallel.sh

