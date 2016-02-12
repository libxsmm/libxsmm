#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
ECHO=$(which echo)
GREP=$(which grep)

${ECHO} "============================="
${ECHO} "Running DGEMM (ORIGINAL BLAS)"
${ECHO} "============================="
( time ${HERE}/dgemm-blas.sh $*; ) 2>&1 | ${GREP} real

if [ -e dgemm-wrap ]; then
  ${ECHO} "============================="
  ${ECHO} "Running DGEMM (STATIC WRAP)"
  ${ECHO} "============================="
  ( time ${HERE}/dgemm-wrap.sh $*; ) 2>&1 | ${GREP} real
fi

if [ -e ../../lib/libxsmm.so ]; then
  ${ECHO} "============================="
  ${ECHO} "Running DGEMM (LD_PRELOAD)"
  ${ECHO} "============================="
  ( LD_PRELOAD=xx time ${HERE}/dgemm-wrap.sh $*; ) 2>&1 | ${GREP} real
fi

