#!/bin/bash

if [[ "-mic" != "$1" ]] ; then
  env \
    KMP_AFFINITY=scatter,granularity=fine \
    OFFLOAD_INIT=on_start \
    MIC_ENV_PREFIX=MIC \
    MIC_KMP_AFFINITY=scatter,granularity=fine \
  ./smm $*
else
  shift
  env \
    SINK_LD_LIBRARY_PATH=$MIC_LD_LIBRARY_PATH \
  micnativeloadex \
    ./smm -a "$*" \
    -e "KMP_AFFINITY=scatter,granularity=fine"
fi

