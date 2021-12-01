#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
DEPDIR=${HERE}/../../..

TMPF=$(${DEPDIR}/.mktmp.sh /tmp/.libxsmm_XXXXXX.out)
UNAME=$(command -v uname)
GREP=$(command -v grep)
SORT=$(command -v sort)
RM=$(command -v rm)
TR=$(command -v tr)

if [ "Darwin" != "$(${UNAME})" ]; then
  LIBEXT=so
else
  LIBEXT=dylib
fi
if [ "$1" ]; then
  TEST=$1
  shift
else
  TEST=dgemm
fi

if [ -e ${HERE}/${TEST}-blas ]; then
  NAME=$(echo ${TEST} | ${TR} [:lower:] [:upper:])
  echo "============================="
  echo "Running ${NAME} (ORIGINAL BLAS)"
  echo "============================="
  { time ${HERE}/${TEST}-blas.sh "$@" 2>${TMPF}; } 2>&1 | ${GREP} real
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    echo -n "FAILED(${RESULT}) "; ${SORT} -u ${TMPF}
    ${RM} -f ${TMPF}
    exit ${RESULT}
  else
    echo -n "OK "; ${SORT} -u ${TMPF}
  fi
  echo

  if [ -e ${DEPDIR}/lib/libxsmmext.${LIBEXT} ]; then
    echo
    echo "============================="
    echo "Running ${NAME} (LD_PRELOAD)"
    echo "============================="
    { time \
      LD_LIBRARY_PATH=${DEPDIR}/lib:${LD_LIBRARY_PATH} LD_PRELOAD=${DEPDIR}/lib/libxsmmext.${LIBEXT} \
      DYLD_LIBRARY_PATH=${DEPDIR}/lib:${DYLD_LIBRARY_PATH} DYLD_INSERT_LIBRARIES=${DEPDIR}/lib/libxsmmext.${LIBEXT} \
      ${HERE}/${TEST}-blas.sh "$@" 2>${TMPF}; } 2>&1 | ${GREP} real
    RESULT=$?
    if [ 0 != ${RESULT} ]; then
      echo -n "FAILED(${RESULT}) "; ${SORT} -u ${TMPF}
      ${RM} -f ${TMPF}
      exit ${RESULT}
    else
      echo -n "OK "; ${SORT} -u ${TMPF}
    fi
    echo
  fi
fi

if [ -e ${HERE}/${TEST}-wrap ] && [ -e .state ] && \
   [ "" = "$(${GREP} 'BLAS=0' .state)" ];
then
  echo
  echo "============================="
  echo "Running ${NAME} (STATIC WRAP)"
  echo "============================="
  { time ${HERE}/${TEST}-wrap.sh "$@" 2>${TMPF}; } 2>&1 | ${GREP} real
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    echo -n "FAILED(${RESULT}) "; ${SORT} -u ${TMPF}
    ${RM} -f ${TMPF}
    exit ${RESULT}
  else
    echo -n "OK "; ${SORT} -u ${TMPF}
  fi
  echo
fi

${RM} -f ${TMPF}

