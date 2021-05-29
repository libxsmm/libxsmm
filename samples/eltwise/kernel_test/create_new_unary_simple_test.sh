#!/bin/bash

if [ $# = 2 ]; then
  NAME=$1
  NUMBER=$2
else
  NAME=unkn
  NUMBER=9999
fi

cp unary_copy_32b_eqld.slurm unary_${NAME}_32b_eqld.slurm
cp unary_copy_mixed_eqld.slurm unary_${NAME}_mixed_eqld.slurm
cp unary_copy_32b_gtld.slurm unary_${NAME}_32b_gtld.slurm
cp unary_copy_mixed_gtld.slurm unary_${NAME}_mixed_gtld.slurm

sed "s/UNARY_OP=1/UNARY_OP=${NUMBER}/g" -i unary_${NAME}_32b_eqld.slurm
sed "s/UNARY_OP=1/UNARY_OP=${NUMBER}/g" -i unary_${NAME}_mixed_eqld.slurm
sed "s/UNARY_OP=1/UNARY_OP=${NUMBER}/g" -i unary_${NAME}_32b_gtld.slurm
sed "s/UNARY_OP=1/UNARY_OP=${NUMBER}/g" -i unary_${NAME}_mixed_gtld.slurm
