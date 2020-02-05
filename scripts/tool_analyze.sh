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

MAKE=$(command -v make)
GREP=$(command -v grep)
SORT=$(command -v sort)
CXX=$(command -v clang++)
CC=$(command -v clang)

if [ "" != "${MAKE}" ] && [ "" != "${CXX}" ] && [ "" != "${CC}" ] && \
   [ "" != "${GREP}" ] && [ "" != "${SORT}" ];
then
  HERE=$(cd "$(dirname "$0")"; pwd -P)
  cd "${HERE}/.."
  ARG=$*
  if [ "" = "${ARG}" ]; then
    ARG=lib
  fi
  ${MAKE} -e CXX=${CXX} CC=${CC} FC= FORCE_CXX=1 DBG=1 ILP64=1 EFLAGS="--analyze" ${ARG} 2> .analyze.log
  ISSUES=$(${GREP} -e "error:" -e "warning:" .analyze.log | ${GREP} -v "is never read" | ${SORT} -u)
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
  echo "Error: missing prerequisites!"
  exit 1
fi

