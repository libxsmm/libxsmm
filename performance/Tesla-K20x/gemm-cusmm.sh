#!/bin/bash

sed \
  -e "s/^.*m=\([0-9]\).*n=\([0-9]\).*k=\([0-9]\).* \(.*\) GFlop.*$/\1 \2 \3 \4/" \
  gemm-cusmm.txt | tail -n+6 | head -n-2 \
> gemm-cusmm-raw.txt

gnuplot gemm-cusmm.plt
