#!/bin/bash

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1000; fi
else # check
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=3; fi
fi

if [ $# -ne 6 ]
then
  echo "Usage: $(basename $0) iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values: 1000 1 f32 A L 0"
  ITERS=${CHECK_DNN_ITERS}
  NUMA=1
  BIN=f32
  TYPE="A"
  FORMAT="L"
  PAD=0
else
  ITERS=$1
  NUMA=$2
  BIN=$3
  TYPE=$4
  FORMAT=$5
  PAD=$6
fi

NUMACTL="${TOOL_COMMAND}"
CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA} ${TOOL_COMMAND}"
  fi
fi

if [[ -z "${OMP_NUM_THREADS}" ]]; then
  echo "using defaults for OMP settings!"
  export KMP_PLACE_THREADS=1T
  export KMP_AFFINITY=compact,granularity=fine
  export OMP_NUM_THREADS=64
else
  echo "using environment OMP settings!"
fi

# ./layer_example_${BIN} iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type

${NUMACTL} ./layer_example_${BIN} ${ITERS}  224 224 64  3 64  7 7 3 2 ${TYPE} ${FORMAT} ${PAD}        && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  64  256 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  64  64  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  64  64  3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  64  256 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  256 64  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  64  64  3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  64  256 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  256 64  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  64  64  3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  64  256 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  256 512 1 1 0 2 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  64  256 128 1 1 0 2 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  128 128 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  128 512 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  512 128 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  128 128 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  128 512 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  512 128 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  128 128 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  128 512 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  512 128 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  128 128 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  128 512 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  512 1024  1 1 0 2 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  64  512 256 1 1 0 2 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 256 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 1024  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  1024  256 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 256 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 1024  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  1024  256 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 256 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 1024  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  1024  256 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 256 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 1024  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  1024  256 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 256 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 1024  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  1024  256 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 256 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}      && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  256 1024  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  1024  2048  1 1 0 2 ${TYPE} ${FORMAT} ${PAD}  && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  64  1024  512 1 1 0 2 ${TYPE} ${FORMAT} ${PAD}    && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  7 7 64  512 512 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}          && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  7 7 64  512 2048  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}        && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  7 7 64  2048  512 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}        && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  7 7 64  512 512 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}          && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  7 7 64  512 2048  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}        && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  7 7 64  2048  512 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}        && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  7 7 64  512 512 3 3 1 1 ${TYPE} ${FORMAT} ${PAD}          && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}  7 7 64  512 2048  1 1 0 1 ${TYPE} ${FORMAT} ${PAD}

