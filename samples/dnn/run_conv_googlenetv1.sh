#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Usage: `basename $0` arch mb iters numa; using default values: 128 100 1"
  MB=128
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

${NUMACTL} ./layer_example_f32 ${ITERS}  11  11  ${MB}   32  128  5  5 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  11  11  ${MB}   48  128  5  5 0 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  ${MB}  480  304  1  1 2 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  ${MB}  480   64  1  1 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  ${MB}  512  128  1  1 0 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  ${MB}  512  288  1  1 2 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  ${MB}  512  304  1  1 2 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  ${MB}  512  448  1  1 2 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  14  14  ${MB}  512   64  1  1 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  16  16  ${MB}  112  224  3  3 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  16  16  ${MB}  128  256  3  3 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  16  16  ${MB}  144  288  3  3 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  16  16  ${MB}  160  320  3  3 0 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  16  16  ${MB}   96  208  3  3 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  18  18  ${MB}   16   48  5  5 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  18  18  ${MB}   32  128  5  5 0 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  18  18  ${MB}   32   64  5  5 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS} 229 229  ${MB}    3   64  7  7 0 2 1
${NUMACTL} ./layer_example_f32 ${ITERS}  28  28  ${MB}  192  176  1  1 2 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  28  28  ${MB}  192   32  1  1 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  28  28  ${MB}  256  288  1  1 2 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  28  28  ${MB}  256   64  1  1 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  30  30  ${MB}   96  128  3  3 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  30  30  ${MB}   96  192  3  3 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  32  32  ${MB}   16   32  5  5 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  32  32  ${MB}   32   96  5  5 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}  58  58  ${MB}   64  192  3  3 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}   7   7  ${MB}  832  128  1  1 0 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}   7   7  ${MB}  832  128  1  1 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}   7   7  ${MB}  832  448  1  1 2 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}   7   7  ${MB}  832  624  1  1 2 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}   9   9  ${MB}  160  320  3  3 1 1 1
${NUMACTL} ./layer_example_f32 ${ITERS}   9   9  ${MB}  192  384  3  3 0 1 1
