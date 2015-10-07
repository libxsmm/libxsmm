#!/bin/bash
if [ `grep "diff" cp2k-perf.txt | grep -v -c "diff=0.000"` -eq 0 ]; then
  exit 0
else
  exit 1
fi
