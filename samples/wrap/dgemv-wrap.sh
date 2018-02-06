#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
NAME=$(basename $0 .sh)
ECHO=$(which echo)
GREP=$(which grep)
ENV=$(which env)

if [ "Windows_NT" = "${OS}" ]; then
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}/../../lib:/usr/x86_64-w64-mingw32/sys-root/mingw/bin
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(which cygcheck)
  EXE=.exe
else
  if [ "" != "$(which ldd)" ]; then
    LDD=ldd
  elif [ "" != "$(which otool)" ]; then
    LDD="otool -L"
  else
    LDD=${ECHO}
  fi
fi

MICINFO=$(which micinfo 2>/dev/null)
if [ "" != "${MICINFO}" ]; then
  MICCORES=$(${MICINFO} 2>/dev/null | sed -n "0,/\s\+Total No of Active Cores :\s\+\([0-9]\+\)/s//\1/p")
fi
if [ "" = "${MICCORES}" ]; then
  MICCORES=61
fi
MICTPERC=3

if [ "-mic" != "$1" ]; then
  if [ "" != "$(${LDD} ${HERE}/${NAME}${EXE} 2>/dev/null | ${GREP} libiomp5\.)" ]; then
    ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../../lib \
      DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../../lib \
      KMP_AFFINITY=scatter,granularity=fine,1 \
      MIC_KMP_AFFINITY=scatter,granularity=fine \
      MIC_KMP_HW_SUBSET=$((MICCORES-1))c${MICTPERC}t \
      MIC_ENV_PREFIX=MIC \
      OFFLOAD_INIT=on_start \
    ${TOOL_COMMAND} ${HERE}/${NAME}${EXE} $*
  else
    ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../../lib \
      DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../../lib \
      OMP_PROC_BIND=TRUE \
    ${TOOL_COMMAND} ${HERE}/${NAME}${EXE} $*
  fi
else
  shift
  ${ENV} \
    SINK_LD_LIBRARY_PATH=${SINK_LD_LIBRARY_PATH}:${MIC_LD_LIBRARY_PATH}:${HERE}/../../lib \
  micnativeloadex \
    ${HERE}/${NAME}${EXE} -a "$*" \
    -e "KMP_AFFINITY=scatter,granularity=fine" \
    -e "MIC_KMP_HW_SUBSET=$((MICCORES-1))${MICTPERC}t"
fi

