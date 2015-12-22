#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)

echo "============="
echo "Running tests"
echo "============="
for TEST in $(ls -1 ${HERE}/*.c) ; do
  NAME=$(basename -s.c ${TEST})
  echo -n "${NAME}... "

  if [[ "-mic" != "$1" ]] ; then
    if [[ "" != "$(ldd ${HERE}/${NAME} | grep libiomp5\.so)" ]] ; then
      env OFFLOAD_INIT=on_start \
        KMP_AFFINITY=scatter,granularity=fine,1 \
        MIC_KMP_AFFINITY=scatter,granularity=fine \
        MIC_ENV_PREFIX=MIC \
      ${HERE}/${NAME} $* 2> /dev/null
    else
      env \
        OMP_PROC_BIND=TRUE \
      ${HERE}/${NAME} $* 2> /dev/null
    fi
  else
    shift
    env \
      SINK_LD_LIBRARY_PATH=$MIC_LD_LIBRARY_PATH \
    micnativeloadex \
      ${HERE}/${NAME} -a "$*" \
      -e "KMP_AFFINITY=scatter,granularity=fine" \
      2> /dev/null
  fi
  if [[ 0 != $? ]] ; then
    echo "FAILED"
    exit 1
  else
    echo "OK"
  fi
done

