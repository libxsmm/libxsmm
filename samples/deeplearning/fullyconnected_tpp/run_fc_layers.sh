#!/usr/bin/env bash
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

if [ $# -ne 5 ]
then
  echo "Usage: $(basename $0) bin=(f32, bf16) iters MB type=(A, F, B) fuse=(0 (None), 1 (Bias), 2 (ReLU), 4 (Bias+ReLU))"
  BIN=bf16
  ITERS=100
  MB=1024
  TYPE=A
  FUSE=4
else
  BIN=$1
  ITERS=$2
  MB=$3
  TYPE=$4
  FUSE=$5
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

${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 14 512 ${FUSE} ${TYPE} 64 32 14
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 512 256 ${FUSE} ${TYPE} 64 32 32
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 256 128 ${FUSE} ${TYPE} 64 32 32
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 480 1024 ${FUSE} ${TYPE} 64 32 32
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 1024 1024 ${FUSE} ${TYPE} 64 32 32
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 1024 512 ${FUSE} ${TYPE} 64 32 32
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 512 256 ${FUSE} ${TYPE} 64 32 32
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 256 1 1 ${TYPE} 64 1 32

${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 512 512 ${FUSE} ${TYPE} 64 32 32
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 512 64 ${FUSE} ${TYPE} 64 32 32
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 100 1024 ${FUSE} ${TYPE} 64 32 50
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 1024 1024 ${FUSE} ${TYPE} 64 32 32
${NUMACTL} ./layer_example_${BIN} ${ITERS} ${MB} 1024 1 0 A 64 1 32
