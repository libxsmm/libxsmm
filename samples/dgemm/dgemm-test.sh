#!/bin/sh

DEPDIR=../../lib

HERE=$(cd $(dirname $0); pwd -P)
UNAME=$(which uname)
ECHO=$(which echo)
GREP=$(which grep)

if [ "Darwin" != "$(${UNAME})" ]; then
  export LD_LIBRARY_PATH=${DEPDIR}:${LD_LIBRARY_PATH}
  LIBEXT=so
else
  set env DYLD_LIBRARY_PATH ${DEPDIR}:${DYLD_LIBRARY_PATH}
  LIBEXT=dylib
fi

if [ -e dgemm-blas ]; then
  ${ECHO} "============================="
  ${ECHO} "Running DGEMM (ORIGINAL BLAS)"
  ${ECHO} "============================="
  ( time ERROR=$({ ${HERE}/dgemm-blas.sh $*; } 2>&1); ) 2>&1 | ${GREP} real
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    ${ECHO} "FAILED(${RESULT}) ${ERROR}"
  else
    ${ECHO} "OK ${ERROR}"
  fi
  ${ECHO}

  if [ -e ${DEPDIR}/libxsmmld.${LIBEXT} ]; then
    ${ECHO} "============================="
    ${ECHO} "Running DGEMM (LD_PRELOAD)"
    ${ECHO} "============================="
    ( time ERROR=$({ \
      LD_PRELOAD=${DEPDIR}/libxsmmld.${LIBEXT} \
      DYLD_INSERT_LIBRARIES=${DEPDIR}/libxsmmld.${LIBEXT} \
        ${HERE}/dgemm-blas.sh $*; } 2>&1); ) 2>&1 | ${GREP} real
    RESULT=$?
    if [ 0 != ${RESULT} ]; then
      ${ECHO} "FAILED(${RESULT}) ${ERROR}"
    else
      ${ECHO} "OK ${ERROR}"
    fi
    ${ECHO}
  fi
fi

if [ -e dgemm-wrap ]; then
  ${ECHO} "============================="
  ${ECHO} "Running DGEMM (STATIC WRAP)"
  ${ECHO} "============================="
  ( time ERROR=$({ ${HERE}/dgemm-wrap.sh $*; } 2>&1); ) 2>&1 | ${GREP} real
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    ${ECHO} "FAILED(${RESULT}) ${ERROR}"
  else
    ${ECHO} "OK ${ERROR}"
  fi
  ${ECHO}
fi

