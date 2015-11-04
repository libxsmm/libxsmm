#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)
NAME=rstr

if [[ "-mic" != "$1" ]] ; then
  if [[ "" != "$(ldd ${HERE}/${NAME} | grep libiomp5\.so)" ]] ; then
    env OFFLOAD_INIT=on_start \
      KMP_AFFINITY=scatter,granularity=fine,1 \
      MIC_KMP_AFFINITY=scatter,granularity=fine \
      MIC_ENV_PREFIX=MIC \
    ${HERE}/${NAME} $*
  else
    env \
      OMP_PROC_BIND=TRUE \
    ${HERE}/${NAME} $*
  fi
else
  shift
  env \
    SINK_LD_LIBRARY_PATH=$MIC_LD_LIBRARY_PATH \
  micnativeloadex \
    ${HERE}/${NAME} -a "$*" \
    -e "KMP_AFFINITY=scatter,granularity=fine"
fi

