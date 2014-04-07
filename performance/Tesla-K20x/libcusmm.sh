#!/bin/bash

sed \
  -e "s/^.*m=\([0-9]\).*n=\([0-9]\).*k=\([0-9]\).* \(.*\) GFlop.*$/\1 \2 \3 \4/" \
  libcusmm.txt | tail -n+6 | head -n-2 \
> libcusmm-raw.txt

gnuplot libcusmm.plt
