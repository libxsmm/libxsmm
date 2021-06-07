#!/bin/bash

if [ $# = 2 ]; then
  NAME=$1
  NUMBER=$2
else
  NAME=unkn
  NUMBER=9999
fi

cp unary_copy_32b_eqld.sh unary_${NAME}_32b_eqld.sh
cp unary_copy_mixed_eqld.sh unary_${NAME}_mixed_eqld.sh
cp unary_copy_32b_gtld.sh unary_${NAME}_32b_gtld.sh
cp unary_copy_mixed_gtld.sh unary_${NAME}_mixed_gtld.sh

sed "s/UNARY_OP=1/UNARY_OP=${NUMBER}/g" -i unary_${NAME}_32b_eqld.sh
sed "s/UNARY_OP=1/UNARY_OP=${NUMBER}/g" -i unary_${NAME}_mixed_eqld.sh
sed "s/UNARY_OP=1/UNARY_OP=${NUMBER}/g" -i unary_${NAME}_32b_gtld.sh
sed "s/UNARY_OP=1/UNARY_OP=${NUMBER}/g" -i unary_${NAME}_mixed_gtld.sh
