#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

echo "#!/bin/bash" > generate_unary_quant_fp_test_scripts_gen_parallel.sh
echo "" >> generate_unary_quant_fp_test_scripts_gen_parallel.sh

for FORMAT in 'mxfp4' 'mxbf8' 'nvfp4'; do
  sed "s/FORMATDESC/${FORMAT}/g" generate_unary_quant_fp_test_scripts.tpl > generate_unary_quant_fp_test_scripts_gen_$FORMAT.sh
  chmod 755 generate_unary_quant_fp_test_scripts_gen_$FORMAT.sh
  echo "./generate_unary_quant_fp_test_scripts_gen_$FORMAT.sh &" >> generate_unary_quant_fp_test_scripts_gen_parallel.sh
done

echo "wait" >> generate_unary_quant_fp_test_scripts_gen_parallel.sh
echo "sync" >> generate_unary_quant_fp_test_scripts_gen_parallel.sh
chmod 755 generate_unary_quant_fp_test_scripts_gen_parallel.sh

./generate_unary_quant_fp_test_scripts_gen_parallel.sh
