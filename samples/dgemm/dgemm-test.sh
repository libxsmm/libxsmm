#!/bin/sh

LIBXSMM=../../lib/libxsmmld.so

HERE=$(cd $(dirname $0); pwd -P)
ECHO=$(which echo)
GREP=$(which grep)

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

  if [ -e ${LIBXSMM} ]; then
    ${ECHO} "============================="
    ${ECHO} "Running DGEMM (LD_PRELOAD)"
    ${ECHO} "============================="
    ( time ERROR=$({ LD_PRELOAD=${LIBXSMM} ${HERE}/dgemm-blas.sh $*; } 2>&1); ) 2>&1 | ${GREP} real
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

