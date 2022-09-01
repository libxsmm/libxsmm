#!/usr/bin/env bash

if [ $# = 2 ]; then
  NAME=$1
  NUMBER=$2
else
  NAME=unkn
  NUMBER=9999
fi

sed "s/UNARY_OP=3/UNARY_OP=${NUMBER}/g" unary_x2_eqld.sh >unary_${NAME}_eqld.sh
sed "s/UNARY_OP=3/UNARY_OP=${NUMBER}/g" unary_x2_gtld.sh >unary_${NAME}_gtld.sh
