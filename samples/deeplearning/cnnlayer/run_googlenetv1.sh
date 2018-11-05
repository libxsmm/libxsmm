#!/bin/bash

UNAME=$(which uname 2>/dev/null)
SORT=$(which sort 2>/dev/null)
GREP=$(which grep 2>/dev/null)
CUT=$(which cut 2>/dev/null)
WC=$(which wc 2>/dev/null)

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=128; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1000; fi
else # check
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=32; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1; fi
fi

if [ $# -ne 7 ]
then
  echo "Usage: $(basename $0) mb iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values; using default values: 128 1000 1 f32 A L 0"
  MB=${CHECK_DNN_MB}
  ITERS=${CHECK_DNN_ITERS}
  NUMA=-1
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

if [ "" != "${GREP}" ] && [ "" != "${SORT}" ] && [ "" != "${WC}" ] && [ -e /proc/cpuinfo ]; then
  export NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)
  export NC=$((NS*$(${GREP} "core id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)))
  export NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l)
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
if [ "" != "${CUT}" ] && [ "" != "$(which numactl 2>/dev/null)" ]; then
  export NN=$(numactl -H | ${GREP} available: | ${CUT} -d' ' -f2)
else
  export NN=${NS}
fi

CPUFLAGS=$(if [ "" != "${GREP}" ] && [ "" != "${CUT}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | ${CUT} -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | ${GREP} -o avx512er)" ]; then
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
${NUMACTL} ./layer_example_${BIN} ${ITERS}   224  224  ${MB}     3    64  7  7  3  3  2 ${TYPE} ${FORMAT} ${PAD}
fi && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    56   56  ${MB}    64    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    56   56  ${MB}    64   192  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    96  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    96   128  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    16  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    16    32  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256   128  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   128   192  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    32    96  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480   192  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    96  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    96   208  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    16  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    16    48  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   160  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   112  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   112   224  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32    64  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   128  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   128   256  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   144  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   144   288  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32    64  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   256  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   160  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   160   320  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32   128  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   128  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   256  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   160  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   160   320  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}    32   128  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   128  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   384  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   192  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   192   384  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832    48  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}   && \
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}    48   128  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}
#${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    24  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
#${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    24    64  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}

