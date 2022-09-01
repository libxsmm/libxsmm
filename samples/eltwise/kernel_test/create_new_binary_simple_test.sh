#!/usr/bin/env bash

if [ $# = 2 ]; then
  NAME=$1
  NUMBER=$2
else
  NAME=unkn
  NUMBER=9999
fi

sed "s/BINARY_OP=1/BINARY_OP=${NUMBER}/g" binary_add_eqld.sh >binary_${NAME}_eqld.sh
sed "s/BINARY_OP=1/BINARY_OP=${NUMBER}/g" binary_add_gtld.sh >binary_${NAME}_gtld.sh
