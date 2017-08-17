#!/bin/bash

SORT=$(which sort 2> /dev/null)
GREP=$(which grep 2> /dev/null)
WC=$(which wc 2> /dev/null)

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1000; fi
else # check
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=3; fi
fi

if [ $# -ne 7 ]
then
  echo "Usage: $(basename $0) mb iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values; using default values: 64 1000 1 f32 A L 0"
  MB=${CHECK_DNN_MB}
  ITERS=${CHECK_DNN_ITERS}
  NUMA=1
  BIN=f32
  TYPE="A"
  FORMAT="L"
  PAD=0
else
  MB=$1
  ITERS=$2
  NUMA=$3
  BIN=$4
  TYPE=$5
  FORMAT=$6
  PAD=$7
fi

NUMACTL="${TOOL_COMMAND}"
CPUFLAGS=$(if [ "" != "${GREP}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | ${GREP} -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | ${GREP} "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA} ${TOOL_COMMAND}"
  fi
fi

if [ "" != "${GREP}" ] && [ "" != "${SORT}" ] && [ -e /proc/cpuinfo ]; then
  export NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)
  export NC=$((NS*$(${GREP} "core id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)))
  export NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l)
fi
if [ "" != "${NC}" ] && [ "" != "${NT}" ]; then
  export HT=$((NT/(NC)))
else
  export NS=1 NC=1 NT=1 HT=1
fi

if [[ -z "${OMP_NUM_THREADS}" ]]; then
  echo "using defaults for OMP settings!"
  export KMP_HW_SUBSET=1T
  export KMP_AFFINITY=compact,granularity=fine
  export OMP_NUM_THREADS=${NC}
else
  echo "using environment OMP settings!"
fi

if [[ -z "${OMP_NUM_THREADS}" ]]; then
  echo "using defaults for OMP settings!"
  export KMP_HW_SUBSET=1T
  export KMP_AFFINITY=compact,granularity=fine
  export OMP_NUM_THREADS=64
else
  echo "using environment OMP settings!"
fi

# ./layer_example_${BIN} iters inpWidth inpHeight nImg nIfm nOfm kw kh padw padh stride type

#${NUMACTL} ./layer_example_${BIN} ${ITERS}  224 224 ${MB}  3     64 7 7 3 3 2 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  64   256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  64    64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  64    64 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  256   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  256  512 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  256  128 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  128  128 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  128  512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  512  128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  512 1024 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  512  256 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  256  256 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  256 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB} 1024  256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB} 1024 2048 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB} 1024  512 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}   7   7  ${MB}  512  512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}   7   7  ${MB}  512 2048 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}   7   7  ${MB} 2048  512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}

