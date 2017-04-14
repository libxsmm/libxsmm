#!/bin/bash

if [ $# -ne 7 ]
then
  echo "Usage: $(basename $0) mb iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values; using default values: 128 1000 1 f32 A L 0"
  MB=128
  ITERS=1000
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
CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA} ${TOOL_COMMAND}"
  fi
fi

# ./layer_example_${BIN} iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type

${NUMACTL} ./layer_example_${BIN} ${ITERS}   224  224  ${MB}     3    64  7  7  3  2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    56   56  ${MB}    64    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    56   56  ${MB}    64   192  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    96  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    96   128  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    16  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    16    32  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256   128  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   128   192  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    32    96  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480   192  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    96  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    96   208  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    16  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    16    48  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   160  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   112  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   112   224  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
#${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    24  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
#${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    24    64  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}
echo "running with 32 output channels instead of 24"
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
echo "running with 32 input channels instead of 24"
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32    64  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    64  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   128  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   128   256  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   144  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   144   288  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32    64  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   256  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   160  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   160   320  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32   128  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   128  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   256  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   160  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   160   320  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832    32  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}    32   128  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   128  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   384  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   192  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   192   384  3  3  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832    48  1  1  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}    48   128  5  5  2  1 ${TYPE} ${FORMAT} ${PAD}

