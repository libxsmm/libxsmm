#!/usr/bin/env bash
set -eo pipefail

UNAME=$(command -v uname)
SORT=$(command -v sort)
GREP=$(command -v grep)
CUT=$(command -v cut)
WC=$(command -v wc)
TR=$(command -v tr)

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1000; fi
else # check
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1; fi
fi

if [ $# -ne 9 ]
then
  echo "Usage: $(basename $0) format=(L, B) bin=(f32, bf16) iters MB type=(A, F, B, U, M) fuse=(0 (None), 1 (Bias), 2 (ReLU), 3 (Sigmoid), 4 (Bias+ReLU), 5 (Bias+Sigmoid)) bn bc bk"
  FORMAT=B
  BIN=f32
  ITERS=${CHECK_DNN_ITERS}
  MB=${CHECK_DNN_MB}
  TYPE=A
  FUSE=0
  BN=32
  BC=32
  BK=32
else
  FORMAT=$1
  BIN=$2
  ITERS=$3
  MB=$4
  TYPE=$5
  FUSE=$6
  BN=$7
  BC=$8
  BK=$9
fi

if [ "" != "${GREP}" ] && [ "" != "${CUT}" ] && [ "" != "${SORT}" ] && [ "" != "${WC}" ] && [ -e /proc/cpuinfo ]; then
  export NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l | ${TR} -d " ")
  export NC=$((NS*$(${GREP} -m1 "cpu cores" /proc/cpuinfo | ${TR} -d " " | ${CUT} -d: -f2)))
  export NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l | ${TR} -d " ")
elif [ "" != "${UNAME}" ] && [ "" != "${CUT}" ] && [ "Darwin" = "$(${UNAME})" ]; then
  export NS=$(sysctl hw.packages | ${CUT} -d: -f2 | tr -d " ")
  export NC=$(sysctl hw.physicalcpu | ${CUT} -d: -f2 | tr -d " ")
  export NT=$(sysctl hw.logicalcpu | ${CUT} -d: -f2 | tr -d " ")
fi
if [ "" != "${NC}" ] && [ "" != "${NT}" ]; then
  export HT=$((NT/(NC)))
else
  export NS=1 NC=1 NT=1 HT=1
fi
if [ "" != "${GREP}" ] && [ "" != "${CUT}" ] && [ "" != "$(command -v numactl)" ]; then
  export NN=$(numactl -H | ${GREP} available: | ${CUT} -d' ' -f2)
else
  export NN=${NS}
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

./layer_example_${BIN} ${ITERS} ${MB} 128 256 ${FUSE} ${TYPE} ${FORMAT} ${BN} ${BK} ${BC}
./layer_example_${BIN} ${ITERS} ${MB} 512 1024 ${FUSE} ${TYPE} ${FORMAT} ${BN} ${BK} ${BC}
./layer_example_${BIN} ${ITERS} ${MB} 1024 1024 ${FUSE} ${TYPE} ${FORMAT} ${BN} ${BK} ${BC}
./layer_example_${BIN} ${ITERS} ${MB} 2048 512 ${FUSE} ${TYPE} ${FORMAT} ${BN} ${BK} ${BC}

