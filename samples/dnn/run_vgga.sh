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

${NUMACTL} ./layer_example_${BIN} ${ITERS} 226 226  ${MB}    3   64  3  3 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS} 114 114  ${MB}   64  128  3  3 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  58  58  ${MB}  128  256  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  58  58  ${MB}  256  256  3  3 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  30  30  ${MB}  256  512  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  30  30  ${MB}  512  512  3  3 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  ${MB}  512  512  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  ${MB}  512  512  3  3 0 1
