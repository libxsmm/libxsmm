#!/bin/sh
#############################################################################
# Copyright (c) 2015-2018, Intel Corporation                                #
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
ECHO=$(which echo)
GREP=$(which grep)
ENV=$(which env)

# TESTS_NOBLAS (which are header-only based) need additional adjustment on the link-line.
# Currently header-only cases are not tested for not requiring BLAS (e.g., descriptor).
# Test increases compilation time!
#
#TESTS_NOBLAS="atomics gemmflags hash malloc matcopy matdiff math mhd otrans threadsafety vla"
#TESTS_DISABLED="headeronly"

if [ "Windows_NT" = "${OS}" ]; then
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}/../lib:/usr/x86_64-w64-mingw32/sys-root/mingw/bin
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

${ECHO} "============="
${ECHO} "Running tests"
${ECHO} "============="

# good-enough pattern to match a main function, and to test this translation unit
if [ "" = "$*" ]; then
  TESTS=$(${GREP} -l "main\s*(.*)" ${HERE}/*.c 2>/dev/null)
else
  TESTS=$*
fi
NTEST=1
NMAX=$(${ECHO} "${TESTS}" | wc -w)
for TEST in ${TESTS}; do
  NAME=$(basename ${TEST} .c)
  ${ECHO} -n "${NTEST} of ${NMAX} (${NAME})... "
  if [ "0" != "$(${ECHO} ${TESTS_DISABLED} | ${GREP} -q ${NAME}; ${ECHO} $?)" ]; then
    cd ${HERE}
    ERROR=$({
    if [ "" != "$(${LDD} ${HERE}/${NAME}${EXE} 2>/dev/null | ${GREP} libiomp5\.)" ]; then
      ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../lib \
        DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../lib \
        KMP_AFFINITY=scatter,granularity=fine,1 \
        MIC_KMP_AFFINITY=scatter,granularity=fine \
        MIC_ENV_PREFIX=MIC \
        OFFLOAD_INIT=on_start \
      ${TOOL_COMMAND} ${HERE}/${NAME}${EXE}
    else
      ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../lib \
        DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../lib \
        OMP_PROC_BIND=TRUE \
      ${TOOL_COMMAND} ${HERE}/${NAME}${EXE}
    fi >/dev/null; } 2>&1)
    RESULT=$?
  else
    ERROR="Test is disabled"
    RESULT=0
  fi
  if [ 0 != ${RESULT} ]; then
    ${ECHO} "FAILED(${RESULT}) ${ERROR}"
    exit ${RESULT}
  else
    ${ECHO} "OK ${ERROR}"
  fi
  NTEST=$((NTEST+1))
done

# selected build-only tests that do not run anything
# below cases do not actually depend on LAPACK/BLAS
#
if [ "Windows_NT" != "${OS}" ]; then
  # Workaround for ICE in ipa-visibility.c (at least GCC 5.4.0/Cygwin)
  WORKAROUND="DBG=0"

  CWD=${PWD}
  cd ${HERE}/build
  for TEST in ${TESTS_NOBLAS}; do
    if [ -e ${HERE}/../lib/libxsmm.a ] && [ -e ${HERE}/../lib/libxsmmext.a ]; then
      make -f ${HERE}/Makefile ${WORKAROUND} BLAS=0 STATIC=1 ${TEST}
    fi
    if [ -e ${HERE}/../lib/libxsmm.so ] && [ -e ${HERE}/../lib/libxsmmext.so ]; then
      make -f ${HERE}/Makefile ${WORKAROUND} BLAS=0 STATIC=0 ${TEST}
    fi
    if [ -e ${HERE}/../lib/libxsmm.dylib ] && [ -e ${HERE}/../lib/libxsmmext.dylib ]; then
      make -f ${HERE}/Makefile ${WORKAROUND} BLAS=0 STATIC=0 ${TEST}
    fi
  done
  cd ${CWD}
fi

