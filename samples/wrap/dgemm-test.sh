#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)
DEPDIR=${HERE}/../..

TMPF=$(mktemp /tmp/.libxsmm_XXXXXX.out)
UNAME=$(which uname)
ECHO=$(which echo)
GREP=$(which grep)
SORT=$(which sort)
RM=$(which rm)

if [ "Darwin" != "$(${UNAME})" ]; then
  LIBEXT=so
else
  LIBEXT=dylib
fi

if [ -e ${HERE}/dgemm-blas ]; then
  ${ECHO} "============================="
  ${ECHO} "Running DGEMM (ORIGINAL BLAS)"
  ${ECHO} "============================="
  { time ${HERE}/dgemm-blas.sh $* 2>${TMPF}; } 2>&1 | ${GREP} real
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    ${ECHO} -n "FAILED(${RESULT}) "; ${SORT} -u ${TMPF}
    ${RM} -f ${TMPF}
    exit ${RESULT}
  else
    ${ECHO} -n "OK "; ${SORT} -u ${TMPF}
  fi
  ${ECHO}

  if [ -e ${DEPDIR}/lib/libxsmmext.${LIBEXT} ]; then
    ${ECHO}
    ${ECHO} "============================="
    ${ECHO} "Running DGEMM (LD_PRELOAD)"
    ${ECHO} "============================="
    { time \
      LD_LIBRARY_PATH=${DEPDIR}/lib:${LD_LIBRARY_PATH} LD_PRELOAD=${DEPDIR}/lib/libxsmmext.${LIBEXT} \
      DYLD_LIBRARY_PATH=${DEPDIR}/lib:${DYLD_LIBRARY_PATH} DYLD_INSERT_LIBRARIES=${DEPDIR}/lib/libxsmmext.${LIBEXT} \
      ${HERE}/dgemm-blas.sh $* 2>${TMPF}; } 2>&1 | ${GREP} real
    RESULT=$?
    if [ 0 != ${RESULT} ]; then
      ${ECHO} -n "FAILED(${RESULT}) "; ${SORT} -u ${TMPF}
      ${RM} -f ${TMPF}
      exit ${RESULT}
    else
      ${ECHO} -n "OK "; ${SORT} -u ${TMPF}
    fi
    ${ECHO}
  fi
fi

if [ -e ${HERE}/dgemm-wrap ]; then
  ${ECHO}
  ${ECHO} "============================="
  ${ECHO} "Running DGEMM (STATIC WRAP)"
  ${ECHO} "============================="
  { time ${HERE}/dgemm-wrap.sh $* 2>${TMPF}; } 2>&1 | ${GREP} real
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    ${ECHO} -n "FAILED(${RESULT}) "; ${SORT} -u ${TMPF}
    ${RM} -f ${TMPF}
    exit ${RESULT}
  else
    ${ECHO} -n "OK "; ${SORT} -u ${TMPF}
  fi
  ${ECHO}
fi

${RM} -f ${TMPF}

