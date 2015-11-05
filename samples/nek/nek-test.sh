#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)

if [ `grep "diff" ${HERE}/*-perf.txt | grep -v -c "diff:              0.0"` -eq 0 ]; then
  exit 0
else
  exit 1
fi
