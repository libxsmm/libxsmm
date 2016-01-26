#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
NAME=$(basename $0 .sh)
GREP=$(which grep)
ENV=$(which env)

if [ "Windows_NT" = "${OS}" ]; then
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(which cygcheck)
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}
else
  LDD=$(which ldd)
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
  if [ "" != "$(${LDD} ${HERE}/${NAME} | ${GREP} libiomp5\.so)" ]; then
    ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE} \
      KMP_AFFINITY=scatter,granularity=fine,1 \
      MIC_KMP_AFFINITY=scatter,granularity=fine \
      MIC_KMP_PLACE_THREADS=$((MICCORES-1))c${MICTPERC}t \
      MIC_ENV_PREFIX=MIC \
      OFFLOAD_INIT=on_start \
    ${HERE}/${NAME} $*
  else
    ${ENV} \
      OMP_PROC_BIND=TRUE \
    ${HERE}/${NAME} $*
  fi
else
  shift
  ${ENV} \
    SINK_LD_LIBRARY_PATH=${SINK_LD_LIBRARY_PATH}:${MIC_LD_LIBRARY_PATH}:${HERE} \
  micnativeloadex \
    ${HERE}/${NAME} -a "$*" \
    -e "KMP_AFFINITY=scatter,granularity=fine" \
    -e "MIC_KMP_PLACE_THREADS=$((MICCORES-1))${MICTPERC}t"
fi

