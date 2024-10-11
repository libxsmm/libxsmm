#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

echo "#!/bin/bash" > generate_binary_test_scripts_gen_parallel.sh
echo "" >> generate_binary_test_scripts_gen_parallel.sh

for PREC in 'F32_F32_F32_F32' 'BF16_BF16_BF16_BF16' 'F32_F32_BF16_F32' 'F32_BF16_F32_F32' 'F32_BF16_BF16_F32' 'BF16_F32_F32_F32' 'BF16_F32_BF16_F32' 'BF16_BF16_F32_F32' 'BF16_BF16_BF16_F32' 'F16_F16_F16_F16' 'F32_F32_F16_F32' 'F32_F16_F32_F32' 'F32_F16_F16_F32' 'F16_F32_F32_F32' 'F16_F32_F16_F32' 'F16_F16_F32_F32' 'F16_F16_F16_F32' 'BF8_BF8_BF8_BF8' 'F32_F32_BF8_F32' 'F32_BF8_F32_F32' 'F32_BF8_BF8_F32' 'BF8_F32_F32_F32' 'BF8_F32_BF8_F32' 'BF8_BF8_F32_F32' 'BF8_BF8_BF8_F32' 'HF8_HF8_HF8_HF8' 'F32_F32_HF8_F32' 'F32_HF8_F32_F32' 'F32_HF8_HF8_F32' 'HF8_F32_F32_F32' 'HF8_F32_HF8_F32' 'HF8_HF8_F32_F32' 'HF8_HF8_HF8_F32' 'F64_F64_F64_F64' 'U16_U16_U32_IMPLICIT'; do
  sed "s/PRECDESC/${PREC}/g" generate_binary_test_scripts.tpl > generate_binary_test_scripts_gen_$PREC.sh
  chmod 755 generate_binary_test_scripts_gen_$PREC.sh
  echo "./generate_binary_test_scripts_gen_$PREC.sh &" >> generate_binary_test_scripts_gen_parallel.sh
done

echo "wait" >> generate_binary_test_scripts_gen_parallel.sh
echo "sync" >> generate_binary_test_scripts_gen_parallel.sh
chmod 755 generate_binary_test_scripts_gen_parallel.sh

./generate_binary_test_scripts_gen_parallel.sh
