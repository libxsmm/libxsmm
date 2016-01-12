#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)
GREP=$(which grep)

if [[ "Windows_NT" == "${OS}" ]] ; then
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(which cygcheck)
else
  LDD=$(which ldd)
fi

echo "============="
echo "Running tests"
echo "============="
for TEST in $(ls -1 ${HERE}/*.c) ; do
  NAME=$(basename ${TEST} .c)
  echo -n "${NAME}... "

  if [[ "-mic" != "$1" ]] ; then
    if [[ "" != "$(${LDD} ${HERE}/${NAME} | ${GREP} libiomp5\.so)" ]] ; then
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
  if [[ 0 != $? ]] ; then
    echo "FAILED"
    exit 1
  else
    echo "OK"
  fi
done

