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

HERE=$(cd "$(dirname "$0")"; pwd -P)
GREP=$(command -v grep)
SED=$(command -v sed)
ENV=$(command -v env)
TR=$(command -v tr)
WC=$(command -v wc)

#Eventually disable a set of tests e.g., TESTS_DISABLED="headeronly"

# list of tests that produce "application must be linked against LAPACK/BLAS" in case of BLAS=0
TESTS_NEEDBLAS="gemm.c"
# grep pattern based on TESTS_NEEDBLAS
TESTS_NEEDBLAS_GREP=$(echo ${TESTS_NEEDBLAS} | ${SED} "s/[[:space:]][[:space:]]*/\\\\|/g" | ${SED} "s/\./\\\\./g")
# good-enough pattern to match main functions, and to include translation unit in test set
if [ "" = "$*" ]; then
  TESTS=$(${GREP} -l "main[[:space:]]*(.*)" ${HERE}/*.c 2>/dev/null)
else
  TESTS=$*
fi
if [ "" != "${TESTS}" ] && [ "" != "$(${GREP} 'BLAS=0' ${HERE}/../.state 2>/dev/null)" ]; then
  TESTS=$(echo "${TESTS}" | ${GREP} -v "${TESTS_NEEDBLAS_GREP}")
fi

if [ "Windows_NT" = "${OS}" ]; then
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}/../lib:/usr/x86_64-w64-mingw32/sys-root/mingw/bin
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(command -v cygcheck)
  EXE=.exe
else
  if [ "" != "$(command -v ldd)" ]; then
    LDD=ldd
  elif [ "" != "$(command -v otool)" ]; then
    LDD="otool -L"
  else
    LDD=echo
  fi
fi

echo "============="
echo "Running tests"
echo "============="

NTEST=1
NMAX=$(echo "${TESTS}" | ${WC} -w | ${TR} -d " ")
for TEST in ${TESTS}; do
  NAME=$(basename "${TEST}" .c)
  echo -n "${NTEST} of ${NMAX} (${NAME})... "
  if [ "0" != "$(echo ${TESTS_DISABLED} | ${GREP} -q ${NAME}; echo $?)" ]; then
    cd ${HERE}
    ERROR=$({
    if [ "" != "$(${LDD} ${HERE}/${NAME}${EXE} 2>/dev/null | ${GREP} libiomp5\.)" ]; then
      ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../lib \
        DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../lib \
        KMP_AFFINITY=scatter,granularity=fine,1 \
        MIC_KMP_AFFINITY=scatter,granularity=fine \
        MIC_ENV_PREFIX=MIC \
        OFFLOAD_INIT=on_start \
      ${TOOL_COMMAND} ${HERE}/${NAME}${EXE} ${TOOL_COMMAND_POST}
    else
      ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../lib \
        DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../lib \
        OMP_PROC_BIND=TRUE \
      ${TOOL_COMMAND} ${HERE}/${NAME}${EXE} ${TOOL_COMMAND_POST}
    fi >/dev/null; } 2>&1)
    RESULT=$?
  else
    ERROR="Test is disabled"
    RESULT=0
  fi
  if [ 0 != ${RESULT} ]; then
    echo "FAILED(${RESULT}) ${ERROR}"
    exit ${RESULT}
  else
    echo "OK ${ERROR}"
  fi
  NTEST=$((NTEST+1))
done

