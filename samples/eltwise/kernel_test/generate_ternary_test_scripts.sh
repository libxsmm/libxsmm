#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

echo "#!/bin/bash" > generate_ternary_test_scripts_gen_parallel.sh
echo "" >> generate_ternary_test_scripts_gen_parallel.sh

for PREC in 'F32_F32_IMPLICIT_F32_F32' 'BF16_BF16_IMPLICIT_BF16_F32' 'F16_F16_IMPLICIT_F16_F32' 'BF8_BF8_IMPLICIT_BF8_F32' 'HF8_HF8_IMPLICIT_HF8_F32'; do
  sed "s/PRECDESC/${PREC}/g" generate_ternary_test_scripts.tpl > generate_ternary_test_scripts_gen_$PREC.sh
  chmod 755 generate_ternary_test_scripts_gen_$PREC.sh
  echo "./generate_ternary_test_scripts_gen_$PREC.sh &" >> generate_ternary_test_scripts_gen_parallel.sh
done

echo "wait" >> generate_ternary_test_scripts_gen_parallel.sh
echo "sync" >> generate_ternary_test_scripts_gen_parallel.sh
chmod 755 generate_ternary_test_scripts_gen_parallel.sh

./generate_ternary_test_scripts_gen_parallel.sh
