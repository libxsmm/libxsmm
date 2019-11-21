#!/bin/bash
#############################################################################
# Copyright (c) 2015-2019, Intel Corporation                                #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions        #
# are met:                                                                  #
# 1. Redistributions of source code must retain the above copyright         #
#    notice, this list of conditions and the following disclaimer.          #
# 2. Redistributions in binary form must reproduce the above copyright      #
#    notice, this list of conditions and the following disclaimer in the    #
#    documentation and/or other materials provided with the distribution.   #
# 3. Neither the name of the copyright holder nor the names of its          #
#    contributors may be used to endorse or promote products derived        #
#    from this software without specific prior written permission.          #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#############################################################################
# Hans Pabst (Intel Corp.)
#############################################################################

HERE=$(cd $(dirname $0); pwd -P)
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
  NAME=$(basename ${TEST} .c)
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

