#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

echo "#!/bin/bash" > generate_unary_transform_test_scripts_gen_parallel.sh
echo "" >> generate_unary_transform_test_scripts_gen_parallel.sh

for PREC in 'I8' 'I16' 'I32' 'I64' 'BF8' 'HF8' 'BF16' 'F16' 'F32' 'F64'; do
  sed "s/PRECDESC/${PREC}/g" generate_unary_transform_test_scripts.tpl > generate_unary_transform_test_scripts_gen_$PREC.sh
  chmod 755 generate_unary_transform_test_scripts_gen_$PREC.sh
  echo "./generate_unary_transform_test_scripts_gen_$PREC.sh &" >> generate_unary_transform_test_scripts_gen_parallel.sh
done

echo "wait" >> generate_unary_transform_test_scripts_gen_parallel.sh
echo "sync" >> generate_unary_transform_test_scripts_gen_parallel.sh
chmod 755 generate_unary_transform_test_scripts_gen_parallel.sh

./generate_unary_transform_test_scripts_gen_parallel.sh
