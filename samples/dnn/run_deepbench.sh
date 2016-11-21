#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Usage: $(basename $0) iters numa (1-mcdram/0-DDR); using default values: 1000 1 f32"
  ITERS=1000
  NUMA=1
  BIN=f32
else
  ITERS=$1
  NUMA=$2
  BIN=$3
fi

NUMACTL="${TOOL_COMMAND}"
CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA} ${TOOL_COMMAND}"
  fi
fi

#DeepBench specifies input pad (as in cuDNN, MKL specs) - manually added to input dimension here

${NUMACTL} ./layer_example_${BIN} ${ITERS}  700 161 4  1   32  5  20  0  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  700 161 8  1   32  5  20  0  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  700 161 16 1   32  5  20  0  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  700 161 32 1   32  5  20  0  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  341 79  4  32  32  5  10  0  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  341 79  8  32  32  5  10  0  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  341 79  16 32  32  5  10  0  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  341 79  32 32  32  5  10  0  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  482 50  16 1   16  3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  242 26  16 16  32  3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  122 14  16 32  64  3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  62  8   16 64  128 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  110 110 8  3   64  3  3   1  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  8  64  64  3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  29  29  8  128 128 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  8  128 256 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  9   9   8  256 512 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  226 226 8  3   64  3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  114 114 8  64  128 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  58  58  8  128 256 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  30  30  8  256 512 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  8  512 512 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  9   9   8  512 512 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  226 226 16 3   64  3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  114 114 16 64  128 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  58  58  16 128 256 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  30  30  16 256 512 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  16  16  16 512 512 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  9   9   16 512 512 3  3   1  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  230 230 16 3   64  7  7   3  2
${NUMACTL} ./layer_example_${BIN} ${ITERS}  32  32  16 192 32  5  5   2  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  16 192 64  1  1   0  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  18  18  16 512 48  5  5   2  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  16 512 192 1  1   0  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  7   7   16 832 256 1  1   0  1
${NUMACTL} ./layer_example_${BIN} ${ITERS}  11  11  16 832 128 5  5   2  1
