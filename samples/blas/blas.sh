#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)
NAME=$(basename ${HERE})

MICINFO=$(which micinfo)
if [[ "" != "${MICINFO}" ]] ; then
  MICCORES=$(${MICINFO} | sed -n "0,/\s\+Total No of Active Cores :\s\+\([0-9]\+\)/s//\1/p")
else
  MICCORES=61
fi

if [[ "-mic" != "$1" ]] ; then
  env \
    KMP_AFFINITY=scatter,granularity=fine \
    OFFLOAD_INIT=on_start \
    MIC_ENV_PREFIX=MIC \
    MIC_KMP_AFFINITY=scatter,granularity=fine \
    MIC_KMP_PLACE_THREADS=$((MICCORES-1))c3t \
  ${HERE}/${NAME} $*
else
  shift
  env \
    SINK_LD_LIBRARY_PATH=$MIC_LD_LIBRARY_PATH \
  micnativeloadex \
    ${HERE}/${NAME} -a "$*" \
    -e "KMP_AFFINITY=scatter,granularity=fine" \
    -e "MIC_KMP_PLACE_THREADS=$((MICCORES-1))3t"
fi

