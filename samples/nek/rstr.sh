#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)
NAME=$(basename $0 .sh)

MICINFO=$(which micinfo 2> /dev/null)
if [[ "" != "${MICINFO}" ]] ; then
  MICCORES=$("${MICINFO}" | sed -n "0,/\s\+Total No of Active Cores :\s\+\([0-9]\+\)/s//\1/p")
fi
if [[ "" == "${MICCORES}" ]] ; then
  MICCORES=61
fi
MICTPERC=3

if [[ "-mic" != "$1" ]] ; then
  if [[ "" != "$(ldd ${HERE}/${NAME} | grep libiomp5\.so)" ]] ; then
    env LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE} \
      KMP_AFFINITY=scatter,granularity=fine,1 \
      MIC_KMP_AFFINITY=scatter,granularity=fine \
      MIC_KMP_PLACE_THREADS=$((MICCORES-1))c${MICTPERC}t \
      MIC_ENV_PREFIX=MIC \
      OFFLOAD_INIT=on_start \
    ${HERE}/${NAME} $*
  else
    env \
      OMP_PROC_BIND=TRUE \
    ${HERE}/${NAME} $*
  fi
else
  shift
  env \
    SINK_LD_LIBRARY_PATH=${SINK_LD_LIBRARY_PATH}:${MIC_LD_LIBRARY_PATH}:${HERE} \
  micnativeloadex \
    ${HERE}/${NAME} -a "$*" \
    -e "KMP_AFFINITY=scatter,granularity=fine" \
    -e "MIC_KMP_PLACE_THREADS=$((MICCORES-1))${MICTPERC}t"
fi

