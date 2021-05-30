#!/bin/bash

if [ $# = 2 ]; then
  NAME=$1
  NUMBER=$2
else
  NAME=unkn
  NUMBER=9999
fi

cp binary_add_32b_eqld.sh binary_${NAME}_32b_eqld.sh
cp binary_add_mixed_eqld.sh binary_${NAME}_mixed_eqld.sh
cp binary_add_32b_gtld.sh binary_${NAME}_32b_gtld.sh
cp binary_add_mixed_gtld.sh binary_${NAME}_mixed_gtld.sh

sed "s/BINARY_OP=1/BINARY_OP=${NUMBER}/g" -i binary_${NAME}_32b_eqld.sh
sed "s/BINARY_OP=1/BINARY_OP=${NUMBER}/g" -i binary_${NAME}_mixed_eqld.sh
sed "s/BINARY_OP=1/BINARY_OP=${NUMBER}/g" -i binary_${NAME}_32b_gtld.sh
sed "s/BINARY_OP=1/BINARY_OP=${NUMBER}/g" -i binary_${NAME}_mixed_gtld.sh
