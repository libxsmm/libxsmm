#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.), Kunal Banerjee (Intel Corp.)
###############################################################################
set -eo pipefail

UNAME=$(command -v uname)
SORT=$(command -v sort)
GREP=$(command -v grep)
CUT=$(command -v cut)
WC=$(command -v wc)
TR=$(command -v tr)
NUMA=-1

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1000; fi
else # check
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1; fi
fi

if [ $# -ne 8 ]
then
  echo "Usage: $(basename $0) format=(nc_ck, nc_kcck) bin=(f32, bf16) iters type=(0-fwd, 1-bwd, 2-upd, 3-bwdupd)"
  FORMAT=nc_ck
  BIN=f32
  ITERS=${CHECK_DNN_ITERS}
  TYPE=0
else
  FORMAT=$1
  BIN=$2
  ITERS=$3
  TYPE=$4
fi

if [ "${GREP}" ] && [ "${SORT}" ] && [ "${CUT}" ] && [ "${TR}" ] && [ "${WC}" ]; then
  if [ "$(command -v lscpu)" ]; then
    NS=$(lscpu | ${GREP} -m1 "Socket(s)" | ${TR} -d " " | ${CUT} -d: -f2)
    if [ "" = "${NS}" ]; then NS=1; fi
    NC=$((NS*$(lscpu | ${GREP} -m1 "Core(s) per socket" | ${TR} -d " " | ${CUT} -d: -f2)))
    NT=$((NC*$(lscpu | ${GREP} -m1 "Thread(s) per core" | ${TR} -d " " | ${CUT} -d: -f2)))
  elif [ -e /proc/cpuinfo ]; then
    NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l | ${TR} -d " ")
    if [ "" = "${NS}" ] || [ "" = "${NS}" ]; then NS=1; fi
    NC=$((NS*$(${GREP} -m1 "cpu cores" /proc/cpuinfo | ${TR} -d " " | ${CUT} -d: -f2)))
    NT=$(${GREP} "core id" /proc/cpuinfo  | ${WC} -l | ${TR} -d " ")
  elif [ "Darwin" = "$(uname)" ]; then
    NS=$(sysctl hw.packages    | ${CUT} -d: -f2 | ${TR} -d " ")
    NC=$(sysctl hw.physicalcpu | ${CUT} -d: -f2 | ${TR} -d " ")
    NT=$(sysctl hw.logicalcpu  | ${CUT} -d: -f2 | ${TR} -d " ")
  fi
  if [ "${NC}" ] && [ "${NT}" ]; then
    HT=$((NT/NC))
  else
    NS=1 NC=1 NT=1 HT=1
  fi
  if [ "$(command -v numactl)" ]; then
    NN=$(numactl -H | ${GREP} "available:" | ${CUT} -d' ' -f2)
  else
    NN=${NS}
  fi
fi

CPUFLAGS=$(if [ "${GREP}" ] && [ "${CUT}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | ${CUT} -d: -f2- || true; fi)
if [ "${GREP}" ] && [ "$(echo "${CPUFLAGS}" | ${GREP} -o avx512er)" ]; then
  if [ "0" != "$((0>NUMA))" ] && [ "0" != "$((NS<NN))" ]; then
    NUMACTL="numactl --preferred=${NS} ${TOOL_COMMAND}"
  elif [ "0" != "$((0<=NUMA && NUMA<NN))" ]; then
    NUMACTL="numactl --preferred=${NUMA} ${TOOL_COMMAND}"
  elif [ "1" != "${NS}" ]; then
    #NUMACTL="numactl -i all ${TOOL_COMMAND}"
    NUMACTL="${TOOL_COMMAND}"
  fi
else
  NUMACTL="${TOOL_COMMAND}"
fi

if [ "" = "${OMP_NUM_THREADS}" ] || [ "0" = "${OMP_NUM_THREADS}" ]; then
  if [ "" = "${KMP_AFFINITY}" ]; then
    export KMP_AFFINITY=compact,granularity=fine KMP_HW_SUBSET=1T
  fi
  export OMP_NUM_THREADS=$((NC))
fi

if [ "" = "${MB}" ] || [ "0" = "${MB}" ]; then
  MB=${OMP_NUM_THREADS}
fi

if [ "" = "${LIBXSMM_TARGET_HIDDEN}" ] || [ "0" = "${LIBXSMM_TARGET_HIDDEN}" ]; then
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} NUMACTL=\"${NUMACTL}\""
  echo
fi

##### using the optimal block size as mentioned in emails
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 10 1024 512 1 10 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 10 1024 512 1 10 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 256 256 101  1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 256 256 10 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 256 256 20 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 256 256 30 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 256 256 40 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 256 256 50 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 256 256 60 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 256 256 70 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 512 512 101 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 512 512 10 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 512 512 20 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 512 512 30 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 512 512 40 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 512 512 50 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 512 512 60 1 32 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 1 512 512 70 1 32 64

${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 640 1024 512 1 64 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 640 1024 512 1 64 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 256 256 101 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 256 256 10 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 256 256 20 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 256 256 30 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 256 256 40 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 256 256 50 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 256 256 60 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 256 256 70 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 512 512 101 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 512 512 10 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 512 512 20 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 512 512 30 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 512 512 40 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 512 512 50 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 512 512 60 4 64 64
${NUMACTL} ./lstmdriver_${FORMAT}_${BIN} ${ITERS} ${TYPE} 64 512 512 70 4 64 64

