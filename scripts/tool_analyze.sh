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
# shellcheck disable=SC2064

HERE=$(cd "$(dirname "$0")" && pwd -P)
CPPCHECK=$(command -v cppcheck)
MKTEMP=${HERE}/../.mktmp.sh
MAKE=$(command -v make)
GREP=$(command -v grep)
SORT=$(command -v sort)
CAT=$(command -v cat)
CXX=$(command -v clang++)
CC=$(command -v clang)
CP=$(command -v cp)
MV=$(command -v mv)

if [ "${CPPCHECK}" ] && [ "${CAT}" ]; then
  ${CPPCHECK} "${HERE}/../src" -isrc/template --file-filter="*.c*" -I"${HERE}/../include" \
    --error-exitcode=111 --force \
    --suppress=preprocessorErrorDirective \
    --suppress=ConfigurationNotChecked \
    --suppress=missingIncludeSystem \
    --suppress=unmatchedSuppression \
    --suppress=unusedFunction \
    --suppress=unknownMacro \
    -DLIBXSMM_PLATFORM_X86 \
  2>.analyze.log
  RESULT=$?
  echo
  if [ "0" != "$((1<RESULT))" ]; then
    echo "================================================================================"
    echo "Issues found by cppcheck"
    echo "================================================================================"
    ${CAT} .analyze.log
    exit 1
  fi
fi

if [ "${MKTEMP}" ] && [ "${MAKE}" ] && \
   [ "${GREP}" ] && [ "${SORT}" ] && \
   [ "${CXX}" ] && [ "${CC}" ] && \
   [ "${CP}" ] && [ "${MV}" ];
then
  cd "${HERE}/.." || exit 1
  ARG=$*
  if [ ! "${ARG}" ]; then
    ARG=lib
  fi
  TMPF=$("${MKTEMP}" .tool_analyze.XXXXXX)
  ${CP} "${HERE}/../include/libxsmm_config.h" "${TMPF}"
  trap "${MV} ${TMPF} ${HERE}/../include/libxsmm_config.h" EXIT
  ${MAKE} -e CXX="${CXX}" CC="${CC}" FORTRAN=0 DBG=1 ILP64=1 WERROR=0 ANALYZE=1 ${ARG} 2>.analyze.log
  ISSUES=$(${GREP} -e "error:" -e "warning:" .analyze.log \
    | ${GREP} -v "make:" \
    | ${GREP} -v "is never read" \
    | ${SORT} -u)
  echo
  echo   "================================================================================"
  if [ ! "${ISSUES}" ]; then
    echo "SUCCESS"
    echo "================================================================================"
  else
    echo "Errors (warnings)"
    echo "================================================================================"
    echo "${ISSUES}"
    exit 1
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
