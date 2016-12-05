#!/bin/bash

if [ $# -ne 4 ]
then
  echo "Usage: $(basename $0) arch mb iters numa; using default values: 128 1000 1 f32"
  MB=128
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

${NUMACTL} ./layer_example_${BIN} ${ITERS}  11  11  ${MB}   32  128  5  5 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  11  11  ${MB}   48  128  5  5 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  480  304  1  1 2 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  480   64  1  1 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  512  128  1  1 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  512  288  1  1 2 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  512  304  1  1 2 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  512  448  1  1 2 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  512   64  1  1 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  ${MB}  112  224  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  ${MB}  128  256  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  ${MB}  144  288  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  ${MB}  160  320  3  3 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  ${MB}   96  208  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  18  18  ${MB}   16   48  5  5 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  18  18  ${MB}   32  128  5  5 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  18  18  ${MB}   32   64  5  5 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS} 229 229  ${MB}    3   64  7  7 0 2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  192  176  1  1 2 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  192   32  1  1 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  256  288  1  1 2 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  256   64  1  1 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  30  30  ${MB}   96  128  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  30  30  ${MB}   96  192  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  32  32  ${MB}   16   32  5  5 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  32  32  ${MB}   32   96  5  5 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  58  58  ${MB}   64  192  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}   7   7  ${MB}  832  128  1  1 0 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}   7   7  ${MB}  832  128  1  1 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}   7   7  ${MB}  832  448  1  1 2 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}   7   7  ${MB}  832  624  1  1 2 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}   9   9  ${MB}  160  320  3  3 1 1
${NUMACTL} ./layer_example_${BIN} ${ITERS}   9   9  ${MB}  192  384  3  3 0 1
