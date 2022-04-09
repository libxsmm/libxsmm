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

if [ $# -ne 7 ]
then
  echo "Usage: $(basename $0) mb iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values; using default values: 64 1000 1 f32 A L 1"
  MB=${CHECK_DNN_MB}
  ITERS=${CHECK_DNN_ITERS}
  NUMA=-1
  BIN=f32
  TYPE="A"
  FORMAT="L"
  PAD=1
else
  MB=$1
  ITERS=$2
  NUMA=$3
  BIN=$4
  TYPE=$5
  FORMAT=$6
  PAD=$7
fi

if [ "${UNAME}" ] && [ "${CUT}" ] && [ "x86_64" = "$(${UNAME} -m)" ]; then
  if [ "${GREP}" ] && [ "${CUT}" ] && [ "${SORT}" ] && [ "${WC}" ] && [ -e /proc/cpuinfo ]; then
    export NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l | ${TR} -d " ")
    export NC=$((NS*$(${GREP} -m1 "cpu cores" /proc/cpuinfo | ${TR} -d " " | ${CUT} -d: -f2)))
    export NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l | ${TR} -d " ")
  elif [ "${UNAME}" ] && [ "${CUT}" ] && [ "Darwin" = "$(${UNAME})" ]; then
    export NS=$(sysctl hw.packages | ${CUT} -d: -f2 | tr -d " ")
    export NC=$(sysctl hw.physicalcpu | ${CUT} -d: -f2 | tr -d " ")
    export NT=$(sysctl hw.logicalcpu | ${CUT} -d: -f2 | tr -d " ")
  fi
elif [ "${UNAME}" ] && [ "${CUT}" ] && [ "aarch64" = "$(${UNAME} -m)" ]; then
  export NS=1
  export NC=$(${GREP} "Features" /proc/cpuinfo | ${WC} -l | ${TR} -d " ")
  export NT=$NC
fi
if [ "${NC}" ] && [ "${NT}" ]; then
  export HT=$((NT/(NC)))
else
  export NS=1 NC=1 NT=1 HT=1
fi
if [ "${GREP}" ] && [ "${CUT}" ] && [ "$(command -v numactl)" ]; then
  export NN=$(numactl -H | ${GREP} "available:" | ${CUT} -d' ' -f2)
else
  export NN=${NS}
fi

if [ "${UNAME}" ] && [ "${CUT}" ] && [ "x86_64" = "$(${UNAME} -m)" ]; then
  CPUFLAGS=$(if [ "${GREP}" ] && [ "${CUT}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | ${CUT} -d: -f2- || true; fi)
else
  CPUFLAGS=
fi
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

if [ "" = "${LIBXSMM_TARGET_HIDDEN}" ] || [ "0" = "${LIBXSMM_TARGET_HIDDEN}" ]; then
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} NUMACTL=\"${NUMACTL}\""
  echo
fi

# ./layer_example_${BIN} iters inpWidth inpHeight nImg nIfm nOfm kw kh padw padh stride type
#
if [ "${BIN}" != "f32" ]; then
  true
else
${NUMACTL} ./layer_example_${BIN} ${ITERS}  224 224 ${MB}     3   64 7 7 3 3 2 ${TYPE} ${FORMAT} ${PAD}
fi
${NUMACTL} ./layer_example_${BIN} ${ITERS}   56  56 ${MB}   128  128 3 3 1 1 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   56  56 ${MB}    64   64 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   56  56 ${MB}   256  512 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   56  56 ${MB}    64   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   56  56 ${MB}    64  256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   56  56 ${MB}   256   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   56  56 ${MB}   256  128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   28  28 ${MB}   256  256 3 3 1 1 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   28  28 ${MB}   128  128 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   28  28 ${MB}   512 1024 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   28  28 ${MB}   512  256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   28  28 ${MB}   512  128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   28  28 ${MB}   128  512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   14  14 ${MB}   512  512 3 3 1 1 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   14  14 ${MB}   256  256 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   14  14 ${MB}  1024 2048 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   14  14 ${MB}   256 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   14  14 ${MB}  1024  512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}   14  14 ${MB}  1024  256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    7   7 ${MB}   512  512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    7   7 ${MB}   512 2048 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    7   7 ${MB}  2048  512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}

