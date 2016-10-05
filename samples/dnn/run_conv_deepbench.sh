#!/bin/bash

if [ $# -ne 2 ]
then
  echo "Usage: `basename $0` iters numa (1-mcdram/0-DDR); using default values: 100 1"
  ITERS=1000
  NUMA=1
else
  ITERS=$1
  NUMA=$2
fi

CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA}"
  fi
fi

${NUMACTL} ./layer_example_f32 ${ITERS}  700 161 4  1   32  5  20  0  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  700 161 8  1   32  5  20  0  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  700 161 16 1   32  5  20  0  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  700 161 32 1   32  5  20  0  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  341 79  4  32  32  5  10  0  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  341 79  8  32  32  5  10  0  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  341 79  16 32  32  5  10  0  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  341 79  32 32  32  5  10  0  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  480 48  16 1   16  3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  240 24  16 16  32  3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  120 12  16 32  64  3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  60  6   16 64  128 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  108 108 8  3   64  3  3   1  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  54  54  8  64  64  3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  27  27  8  128 128 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  8  128 256 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  7   7   8  256 512 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  224 224 8  3   64  3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  112 112 8  64  128 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  56  56  8  128 256 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  28  28  8  256 512 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  8  512 512 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  7   7   8  512 512 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  224 224 16 3   64  3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  112 112 16 64  128 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  56  56  16 128 256 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  28  28  16 256 512 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  16 512 512 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  7   7   16 512 512 3  3   1  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  224 224 16 3   64  7  7   3  2  1
${NUMACTL} ./layer_example_f32 ${ITERS}  28  28  16 192 32  5  5   2  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  28  28  16 192 64  1  1   0  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  16 512 48  5  5   2  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  16 512 192 1  1   0  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  7   7   16 832 256 1  1   0  1  1
${NUMACTL} ./layer_example_f32 ${ITERS}  7   7   16 832 128 5  5   2  1  1
