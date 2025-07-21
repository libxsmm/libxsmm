#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

echo "#!/bin/bash" > generate_unary_quant_test_scripts_gen_parallel.sh
echo "" >> generate_unary_quant_test_scripts_gen_parallel.sh

for PREC in 'F32_I8' 'F32_I16' 'F32_I32'; do
  sed "s/PRECDESC/${PREC}/g" generate_unary_quant_test_scripts.tpl > generate_unary_quant_test_scripts_gen_$PREC.sh
  chmod 755 generate_unary_quant_test_scripts_gen_$PREC.sh
  echo "./generate_unary_quant_test_scripts_gen_$PREC.sh &" >> generate_unary_quant_test_scripts_gen_parallel.sh
done

echo "wait" >> generate_unary_quant_test_scripts_gen_parallel.sh
echo "sync" >> generate_unary_quant_test_scripts_gen_parallel.sh
chmod 755 generate_unary_quant_test_scripts_gen_parallel.sh

./generate_unary_quant_test_scripts_gen_parallel.sh
