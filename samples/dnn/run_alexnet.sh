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

${NUMACTL} ./layer_example_${BIN} ${ITERS} 227 227  ${MB}    3   96 11 11 0 4
${NUMACTL} ./layer_example_${BIN} ${ITERS}  31  31  ${MB}   96  256  5  5 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  15  15  ${MB}  256  384  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  15  15  ${MB}  384  384  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  15  15  ${MB}  384  256  3  3 0 1
