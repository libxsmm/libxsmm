#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
NAME=$(basename $0 .sh)
GREP=$(which grep)
ENV=$(which env)

if [ "Windows_NT" = "${OS}" ]; then
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(which cygcheck)
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}/../../lib
else
  if [ "" != "$(which ldd)" ]; then
    LDD=ldd
  elif [ "" != "$(which otool)" ]; then
    LDD="otool -L"
  else
    LDD=echo
  fi
fi

MICINFO=$(which micinfo 2> /dev/null)
if [ "" != "${MICINFO}" ]; then
  MICCORES=$("${MICINFO}" | sed -n "0,/\s\+Total No of Active Cores :\s\+\([0-9]\+\)/s//\1/p")
fi
if [ "" = "${MICCORES}" ]; then
  MICCORES=61
fi
MICTPERC=3

if [ "-mic" != "$1" ]; then
  if [ "" != "$(${LDD} ${HERE}/${NAME} 2> /dev/null | ${GREP} libiomp5\.)" ]; then
    ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../../lib \
      DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../../lib \
      KMP_AFFINITY=compact,granularity=fine,1 \
      MIC_KMP_AFFINITY=compact,granularity=fine \
      MIC_KMP_PLACE_THREADS=$((MICCORES-1))c${MICTPERC}t \
      MIC_ENV_PREFIX=MIC \
      OFFLOAD_INIT=on_start \
    ${TOOL_COMMAND} ${HERE}/${NAME} $*
  else
    ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../../lib \
      DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../../lib \
      OMP_PROC_BIND=TRUE \
    ${TOOL_COMMAND} ${HERE}/${NAME} $*
  fi
else
  shift
  ${ENV} \
    SINK_LD_LIBRARY_PATH=${SINK_LD_LIBRARY_PATH}:${MIC_LD_LIBRARY_PATH}:${HERE}/../../lib \
  micnativeloadex \
    ${HERE}/${NAME} -a "$*" \
    -e "KMP_AFFINITY=scatter,granularity=fine" \
    -e "MIC_KMP_PLACE_THREADS=$((MICCORES-1))${MICTPERC}t"
fi

