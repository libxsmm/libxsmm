#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Evangelos Georganas, Alexander Heinecke (Intel Corp.)
###############################################################################
#Usage: ./parse.sh output_log_file batch_id

NEXT_BID=$(($2+1))
N_LINES=$(cat $1 | grep -A 10000 'Executing batch number '$2'' |  grep -B 10000 'Executing batch number '$NEXT_BID''| wc -l | tr -d ' ')
N_LINES=$(($N_LINES-1))

echo "================================================"
echo "************ Convolutions timings **************"
echo "LIBXSMM Conv FWD time in ms"
FP_TIME=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'XSMM-CONV-FP mb' | cut -d "=" -f2 | cut -d 'm' -f1 | paste -sd+ | bc)
FP_TIME=$(echo "${FP_TIME}" | bc)
echo $FP_TIME

echo "LIBXSMM Conv BWD  time in ms"
BP_TIME=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'XSMM-CONV-BP mb' | cut -d "=" -f2 | cut -d 'm' -f1 | paste -sd+ | bc)
BP_TIME=$(echo "${BP_TIME}" | bc)
echo $BP_TIME

echo "LIBXSMM Conv UPD  time in ms"
WU_TIME=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'XSMM-CONV-WU mb' | cut -d "=" -f2 | cut -d 'm' -f1 | paste -sd+ | bc)
WU_TIME=$(echo "${WU_TIME}" | bc)
echo $WU_TIME

echo "LIBXSMM Conv total time in ms for minibatch $2"
TOTAL_TIME=$(echo "${FP_TIME}+${BP_TIME}+${WU_TIME}" | bc)
echo $TOTAL_TIME
echo "------------------------------------------------"

echo "Conv FWD time in ms"
FP_TIME=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'conv' | grep -v 'pool' | grep "task 0" | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $FP_TIME

echo "Conv BWD time in ms"
BP_TIME=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'conv' | grep -v 'pool' | grep "task 1" | grep  -v "split" | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $BP_TIME

echo "Conv UPD time in ms"
WU_TIME=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'conv' | grep -v 'pool' | grep "task 2" | grep  -v "split" | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $WU_TIME

echo "================================================"
echo "************ Batch norm timings ****************"
echo "LIBXSMM BN FWD time in ms"
BN_F=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'XSMM-BN-FP mb' | cut -d "=" -f2 | cut -d 'm' -f1 | paste -sd+ | bc)
BN_F=$(echo "${BN_F}" | bc)
echo $BN_F

echo "LIBXSMM BN BWD  time in ms"
BN_B=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'XSMM-BN-BP mb' | cut -d "=" -f2 | cut -d 'm' -f1 | paste -sd+ | bc)
BN_B=$(echo "${BN_B}" | bc)
echo $BN_B

echo "LIBXSMM BN total time in ms for minibatch $2"
TOTAL_TIME=$(echo "${BN_F}+${BN_B}" | bc)
echo $TOTAL_TIME
echo "------------------------------------------------"

echo "BN FWD time in ms"
BN_F=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'bn' | grep "task 0" | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $BN_F

echo "BN BWD time in ms"
BN_B=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'bn' | grep "task 1" | grep  -v "split" | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $BN_B

echo "================================================"
echo "************ Split timings *********************"
echo "Split BWD time in ms"
SPLIT_B=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'bn' | grep "task 1" | grep "split" | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $SPLIT_B

echo "================================================"
echo "************ Pool timings **********************"
echo "Pool FWD time in ms"
POOL_F=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'pool' | grep "task 0" | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $POOL_F

echo "POOL BWD time in ms"
POOL_B=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'pool' | grep "task 1" | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $POOL_B

echo "========================================"
echo "************ FC timings ****************"
echo "FC time in ms"
FC=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep 'fc' | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $FC

echo "========================================"
echo "************ SGD timings ****************"
echo "SGD time in ms"
SGD=$(cat $1 | grep -A ${N_LINES} 'Executing batch number '$2'' | grep "task 3" | awk -F" " '{print $7}' | paste -sd+ | bc)
echo $SGD


echo "========================="
echo "Total time in ms for minibatch $2"
TOTAL_TIME=$(echo "${FP_TIME}+${BP_TIME}+${WU_TIME}+${BN_F}+${BN_B}+${SPLIT_B}+${FC}+${SGD}+${POOL_F}+${POOL_B}" | bc)
echo $TOTAL_TIME

