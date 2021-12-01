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
MKTEMP=${HERE}/../.mktmp.sh
MAKE=$(command -v make)
GREP=$(command -v grep)
SORT=$(command -v sort)
CXX=$(command -v clang++)
CC=$(command -v clang)
CP=$(command -v cp)
MV=$(command -v mv)

if [ "${MKTEMP}" ] && [ "${MAKE}" ] && \
   [ "${GREP}" ] && [ "${SORT}" ] && \
   [ "${CXX}" ] && [ "${CC}" ] && \
   [ "${CP}" ] && [ "${MV}" ];
then
  cd "${HERE}/.." || exit 1
  ARG=$*
  if [ "" = "${ARG}" ]; then
    ARG=lib
  fi
  TMPF=$("${MKTEMP}" .tool_analyze.XXXXXX)
  ${CP} "${HERE}/../include/libxsmm_config.h" "${TMPF}"
  ${MAKE} -e CXX="${CXX}" CC="${CC}" FC= FORCE_CXX=1 DBG=1 ILP64=1 EFLAGS="--analyze" ${ARG} 2> .analyze.log
  ${MV} "${TMPF}" "${HERE}/../include/libxsmm_config.h"
  ISSUES=$(${GREP} -e "error:" -e "warning:" .analyze.log \
    | ${GREP} -v "make:" \
    | ${GREP} -v "is never read" \
    | ${SORT} -u)
  echo
  echo   "================================================================================"
  if [ "" = "${ISSUES}" ]; then
    echo "SUCCESS"
    echo "================================================================================"
  else
    echo "Errors (warnings)"
    echo "================================================================================"
    echo "${ISSUES}"
    exit 1
  fi
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi
