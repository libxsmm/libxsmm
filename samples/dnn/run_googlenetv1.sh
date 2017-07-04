#!/bin/bash

SORT=$(which sort 2> /dev/null)
GREP=$(which grep 2> /dev/null)
WC=$(which wc 2> /dev/null)

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=128; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1000; fi
else # check
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=32; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=3; fi
fi

if [ $# -ne 7 ]
then
  echo "Usage: $(basename $0) mb iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values; using default values: 128 1000 1 f32 A L 0"
  MB=${CHECK_DNN_MB}
  ITERS=${CHECK_DNN_ITERS}
  NUMA=1
  BIN=f32
  TYPE=A
  FORMAT=L
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
  export HT=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)
  export NT=$(${GREP} "physical id" /proc/cpuinfo | ${WC} -l)
fi
if [ "" != "${NT}" ] && [ "" != "${HT}" ]; then
  export NC=$((NT/HT))
else
  export NT=1 HT=1 NC=1
fi

if [[ -z "${OMP_NUM_THREADS}" ]]; then
  echo "using defaults for OMP settings!"
  export KMP_HW_SUBSET=1T
  export KMP_AFFINITY=compact,granularity=fine
  export OMP_NUM_THREADS=${NC}
else
  echo "using environment OMP settings!"
fi

# ./layer_example_${BIN} iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type

${NUMACTL} ./layer_example_${BIN} ${ITERS}   224  224  ${MB}     3    64  7  7  3  2 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    56   56  ${MB}    64    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    56   56  ${MB}    64   192  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    96  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    96   128  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    16  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    16    32  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256   128  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   128   192  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    32    96  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480   192  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    96  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    96   208  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    16  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    16    48  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   160  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   112  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   112   224  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32    64  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   128  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   128   256  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   144  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   144   288  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32    64  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   256  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   160  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   160   320  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32   128  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   128  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   256  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   160  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   160   320  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}    32   128  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   128  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   384  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   192  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   192   384  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832    48  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}    48   128  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}
#${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    24  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
#${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    24    64  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}

