#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################
# shellcheck disable=SC2011
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
DEPDIR=${HERE}/../../..

UNAME=$(command -v uname)
GREP=$(command -v grep)
TR=$(command -v tr)

if [ "Darwin" != "$(${UNAME})" ]; then
  LIBEXT=so
else
  LIBEXT=dylib
fi
if [ "$1" ]; then
  TESTS=$1
  shift
else
  TESTS="$(ls -1 "${HERE}"/*.c | xargs -I{} basename {} .c)"
fi

TMPF=$(mktemp)
trap 'rm ${TMPF}' EXIT

# set verbosity to check for generated kernels
export LIBXSMM_VERBOSE=${LIBXSMM_VERBOSE:-2}

for TEST in ${TESTS}; do
  NAME=$(echo "${TEST}" | ${TR} [[:lower:]] [[:upper:]])

  if [ -e "${HERE}/${TEST}-blas" ]; then
    echo "-----------------------------------"
    echo "${NAME} (ORIGINAL BLAS)"
    if [ "$*" ]; then echo "args    $*"; fi
    { time "${HERE}/${TEST}-blas" "$*" 2>"${TMPF}"; } 2>&1 | ${GREP} real
    RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      echo "FAILED(${RESULT})"
      exit ${RESULT}
    elif ! ${GREP} -q 'TRY[[:space:]]\+JIT' "${TMPF}"; then
      echo "OK"
    else
      echo "FAILED"
      exit 1
    fi
    echo
  fi

  if [ -e "${HERE}/${TEST}-wrap" ] && [ -e .state ] && \
     [ ! "$(${GREP} 'BLAS=0' .state)" ];
  then
    echo "-----------------------------------"
    echo "${NAME} (STATIC WRAP)"
    if [ "$*" ]; then echo "args    $*"; fi
    { time "${HERE}/${TEST}-wrap" "$*" 2>"${TMPF}"; } 2>&1 | ${GREP} real
    RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      echo "FAILED(${RESULT})"
      exit ${RESULT}
    elif ${GREP} -q 'TRY[[:space:]]\+JIT' "${TMPF}"; then
      echo "OK"
    else
      echo "FAILED"
      exit 1
    fi
    echo
  fi

  if [ -e "${HERE}/${TEST}-blas" ] && \
     [ -e "${DEPDIR}/lib/libxsmmext.${LIBEXT}" ];
  then
    echo "-----------------------------------"
    echo "${NAME} (LD_PRELOAD)"
    if [ "$*" ]; then echo "args    $*"; fi
    { time \
      LD_LIBRARY_PATH=${DEPDIR}/lib:${LD_LIBRARY_PATH} LD_PRELOAD=${DEPDIR}/lib/libxsmmext.${LIBEXT} \
      DYLD_LIBRARY_PATH=${DEPDIR}/lib:${DYLD_LIBRARY_PATH} DYLD_INSERT_LIBRARIES=${DEPDIR}/lib/libxsmm.${LIBEXT} \
      "${HERE}/${TEST}-blas" "$*" 2>"${TMPF}"; } 2>&1 | ${GREP} real
    RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      echo "FAILED(${RESULT})"
      exit ${RESULT}
    elif ${GREP} -q 'TRY[[:space:]]\+JIT' "${TMPF}"; then
      echo "OK"
    else
      echo "FAILED"
      exit 1
    fi
    echo
  fi
done
