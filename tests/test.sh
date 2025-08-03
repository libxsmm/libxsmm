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
# shellcheck disable=SC2086

HERE=$(cd "$(dirname "$0")" && pwd -P)
SORT=$(command -v sort)
GREP=$(command -v grep)
SED=$(command -v sed)
ENV=$(command -v env)
TR=$(command -v tr)
WC=$(command -v wc)

if [ ! "${GREP}" ] || [ ! "${SED}" ] || [ ! "${TR}" ] || [ ! "${WC}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

UNIX=`uname`
MACHINE=`uname -m`

# good-enough pattern to match main functions, and to include translation unit in test set
if [ ! "$*" ]; then
  if [ "Linux" = "${UNIX}" ]; then
    if [ "riscv64" = "$(MACHINE)" ]; then
      TESTS="$(cd "${HERE}" && ${GREP} -l "main[[:space:]]*(.*)" ./*.c 2>/dev/null) \
        smm.sh"
    else
      TESTS="$(cd "${HERE}" && ${GREP} -l "main[[:space:]]*(.*)" ./*.c 2>/dev/null) \
        dispatch.sh eltwise.sh equation.sh \
        fsspmdm.sh memcmp.sh \
        packed.sh smm.sh"
    fi
  else
    TESTS="$(cd "${HERE}" && ${GREP} -l "main[[:space:]]*(.*)" ./*.c 2>/dev/null) \
      dispatch.sh eltwise.sh equation.sh \
      memcmp.sh \
      packed.sh smm.sh"
  fi
  if [ "${SORT}" ]; then
    TESTS=$(echo "${TESTS}" | ${TR} -s " " "\n" | ${SORT})
  fi
else
  TESTS="$*"
fi

if [ "Windows_NT" = "${OS}" ]; then
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}/../lib:/usr/x86_64-w64-mingw32/sys-root/mingw/bin
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(command -v cygcheck)
  EXE=.exe
else
  if [ "$(command -v ldd)" ]; then
    LDD=ldd
  elif [ "$(command -v otool)" ]; then
    LDD="otool -L"
  else
    LDD="echo"
  fi
fi

echo "============="
echo "Running tests"
echo "============="

NTEST=1
NMAX=$(echo "${TESTS}" | ${WC} -w | ${TR} -d " ")
for TEST in ${TESTS}; do
  NAME=$(echo "${TEST}" | ${SED} 's/.*\///;s/\(.*\)\..*/\1/')
  printf "%02d of %02d: %-12s " "${NTEST}" "${NMAX}" "${NAME}"
  if [ "0" != "$(echo "${TESTS_DISABLED}" | ${GREP} -q "${NAME}"; echo $?)" ]; then
    cd "${HERE}" || exit 1
    if [ -e "${HERE}/${NAME}.sh" ]; then
      ERROR=$(bash ${HERE}/${NAME}.sh)
    elif [ -e "${HERE}/${NAME}${EXE}" ]; then
      ERROR=$({ \
        if [ "$(${LDD} "${HERE}/${NAME}${EXE}" 2>/dev/null | ${GREP} libiomp5\.)" ]; then \
          ${ENV} LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HERE}/../lib" \
            DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:${HERE}/../lib" \
            KMP_AFFINITY=scatter,granularity=fine,1 \
            MIC_KMP_AFFINITY=scatter,granularity=fine \
            MIC_ENV_PREFIX=MIC \
            OFFLOAD_INIT=on_start \
          ${TOOL_COMMAND} ${HERE}/${NAME}${EXE} ${TOOL_COMMAND_POST}; \
        else \
          ${ENV} LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HERE}/../lib" \
            DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:${HERE}/../lib" \
            OMP_PROC_BIND=TRUE \
          ${TOOL_COMMAND} ${HERE}/${NAME}${EXE} ${TOOL_COMMAND_POST}; \
        fi >/dev/null; } 2>&1)
    else
      ERROR="Test is missing"
      RESULT=1
    fi
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
