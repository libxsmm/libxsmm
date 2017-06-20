#!/bin/bash

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
CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA} ${TOOL_COMMAND}"
  fi
fi

# ./layer_example_${BIN} iters inpWidth inpHeight nImg nIfm nOfm kw kh padh padw stride type

    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  299 299  ${MB}    3   32 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD} && \ 
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  149 149  ${MB}   32   32 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \ 
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  147 147  ${MB}   32   64 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD} && \ 
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   73  73  ${MB}   64   80 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \ 
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   73  73  ${MB}   80  192 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \ 
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  192   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   64   96 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   96   96 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  192   48 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   48   64 5 5 2 2 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  192   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  192   32 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  256   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   64   96 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   96   96 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  256   48 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   48   64 5 5 2 2 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  256   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  256   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  288   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   64   96 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   96   96 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  288   48 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   48   64 5 5 2 2 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  288   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  288   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  288   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   64   96 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}   96   96 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   35  35  ${MB}  288  384 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  128  128 1 7 3 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  128  128 7 1 0 3 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  128  128 1 7 3 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  128  192 7 1 0 3 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  128  128 7 1 0 3 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  128  192 1 7 3 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  192 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  192 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  160 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  160 1 7 3 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  160 7 1 0 3 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  160 1 7 3 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  192 7 1 0 3 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  160 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  160 7 1 0 3 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  192 1 7 3 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  192 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  192 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  160 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  160 1 7 3 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  160 7 1 0 3 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  160 1 7 3 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  160  192 7 1 0 3 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}   17  17  ${MB}  768  160 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 160 160 7 1 0 3 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 160 192 1 7 3 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 768 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 768 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 768 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 192 1 7 3 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 192 7 1 0 3 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 192 1 7 3 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 192 7 1 0 3 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 768 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 192 7 1 0 3 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 192 1 7 3 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 768 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 768 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 768 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 192 7 1 0 3 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 192 1 7 3 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 192 3 3 0 0 2 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 768 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 1280 320 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 1280 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 1280 448 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 448 384 3 3 1 1 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 384 384 1 3 1 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 384 384 3 1 0 1 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 1280 384 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 384 384 1 3 1 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 384 384 3 1 0 1 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 2048 320 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 2048 192 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 2048 448 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 448 384 3 3 1 1 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 384 384 1 3 1 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 384 384 3 1 0 1 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 2048 384 1 1 0 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 384 384 1 3 1 0 1 ${TYPE} ${FORMAT}  ${PAD} && \
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  8 8 ${MB} 384 384 3 1 0 1 1 ${TYPE} ${FORMAT} ${PAD}
    ${NUMACTL}  ./layer_example_${BIN} ${ITERS}  17 17 ${MB} 192 320 3 3 0 0 2 ${TYPE} ${FORMAT}  ${PAD} && \
