#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

echo "#!/bin/bash" > ${HERE}/generate_gemm_test_scripts_gen_parallel.sh
echo "" >> ${HERE}/generate_gemm_test_scripts_gen_parallel.sh

for PREC in 'F64_F64_F64_F64' 'F32_F32_F32_F32' 'BF16_BF16_F32_F32' 'BF16_BF16_F32_BF16' 'BF8_BF8_F32_F32' 'BF8_BF8_F32_BF8' 'HF8_HF8_F32_F32' 'HF8_HF8_F32_HF8' 'I16_I16_I32_I32' 'U8_I8_I32_I32' 'I8_U8_I32_I32' 'U8_U8_I32_I32' 'I8_I8_I32_I32' 'U8_I8_I32_F32' 'I8_U8_I32_F32' 'U8_U8_I32_F32' 'I8_I8_I32_F32' 'F16_F16_F16_F16' 'I8_F16_F16_F16' 'BF8_F16_F16_F16' 'F16_F16_F32_F16' 'I8_F16_F32_F16' 'BF8_F16_F32_F16' 'F16_F16_IMPLICIT_F16' 'I8_F16_IMPLICIT_F16' 'BF8_F16_IMPLICIT_F16' 'F16_F16_F16_F32' 'I8_F16_F16_F32' 'BF8_F16_F16_F32' 'F16_F16_F32_F32' 'I8_F16_F32_F32' 'BF8_F16_F32_F32' 'F16_F16_IMPLICIT_F32' 'I8_F16_IMPLICIT_F32' 'BF8_F16_IMPLICIT_F32' 'I8_BF16_F32_F32' 'I8_BF16_F32_BF16' 'I4_F16_IMPLICIT_F16' 'I4_F16_F32_F16' 'I4_F16_F16_F16' 'I4_F16_IMPLICIT_F32' 'I4_F16_F16_F32' 'I4_F16_F32_F32' 'U4_U8_I32_I32' 'U4_F16_IMPLICIT_F16' 'U4_F16_F32_F16' 'U4_F16_F16_F16' 'U4_F16_IMPLICIT_F32' 'U4_F16_F16_F32' 'U4_F16_F32_F32' 'U8_F16_F16_F16' 'U8_F16_F32_F16' 'U8_F16_IMPLICIT_F16' 'U8_F16_F16_F32' 'U8_F16_F32_F32' 'U8_F16_IMPLICIT_F32' 'U8_BF16_F32_F32' 'U8_BF16_F32_BF16' 'BF8_BF16_F32_F32' 'BF8_BF16_F32_BF16' 'HF8_BF16_F32_F32' 'HF8_BF16_F32_BF16' 'MXFP4_BF16_F32_F32' 'MXFP4_BF16_F32_BF16' 'MXFP4_I8_I32_F32' 'MXFP4_I8_I32_BF16' 'MXFP4_F32_F32_F32'; do
  sed "s/PRECDESC/${PREC}/g" ${HERE}/generate_gemm_test_scripts.tpl > ${HERE}/generate_gemm_test_scripts_gen_$PREC.sh
  chmod 755 ${HERE}/generate_gemm_test_scripts_gen_$PREC.sh
  echo "./generate_gemm_test_scripts_gen_$PREC.sh &" >> ${HERE}/generate_gemm_test_scripts_gen_parallel.sh
done

echo "wait" >> ${HERE}/generate_gemm_test_scripts_gen_parallel.sh
echo "sync" >> ${HERE}/generate_gemm_test_scripts_gen_parallel.sh
chmod 755 ${HERE}/generate_gemm_test_scripts_gen_parallel.sh

${HERE}/generate_gemm_test_scripts_gen_parallel.sh
