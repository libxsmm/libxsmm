#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Usage: `basename $0` mb iters numa; using default values: 256 100 1"
  MB=256
  ITERS=100
  NUMA=1
else
  MB=$1
  ITERS=$2
  NUMA=$3
fi

CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA}"
  fi
fi

${NUMACTL} ./layer_example_f32 ${ITERS} 227 227  ${MB}    3   96 11 11 0 4 1
${NUMACTL} ./layer_example_f32 ${ITERS}  31  31  ${MB}   96  256  5  5 0 1 2
${NUMACTL} ./layer_example_f32 ${ITERS}  15  15  ${MB}  256  384  3  3 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  15  15  ${MB}  384  384  3  3 1 1 2
${NUMACTL} ./layer_example_f32 ${ITERS}  15  15  ${MB}  384  256  3  3 0 1 2
