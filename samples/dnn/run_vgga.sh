#!/bin/bash

if [ $# -ne 4 ]
then
  echo "Usage: $(basename $0) mb iters numa; using default values: 256 1000 1 f32"
  MB=256
  ITERS=1000
  NUMA=1
  BIN=f32
else
  MB=$1
  ITERS=$2
  NUMA=$3
  BIN=$4
fi

NUMACTL="${TOOL_COMMAND}"
CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA} ${TOOL_COMMAND}"
  fi
fi

${NUMACTL} ./layer_example_${BIN} ${ITERS} 224 224  ${MB}    3   64  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS} 112 112  ${MB}   64  128  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  128  256  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  256  256  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  256  512  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  512  512  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  512  512  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  512  512  3  3 1 1
