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
#set -o pipefail

BASENAME=$(command -v basename)
XARGS=$(command -v xargs)
FILE=$(command -v file)
GREP=$(command -v grep)

if [ "${BASENAME}" ] && [ "${XARGS}" ] && [ "${FILE}" ] && [ "${GREP}" ]; then
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  INFO=${HERE}/tool_cpuinfo.sh
  NTASKS=$1
  if [ -e "${INFO}" ]; then
    NC=$(${INFO} -nc)
    NT=$(${INFO} -nt)
  fi
  if [ "${NC}" ]; then
    if [ "${NTASKS}" ]; then
      NTASKS=$((NTASKS<=NC?NTASKS:NC))
    else
      NTASKS=${NC}
    fi
  fi
  if [ "${NTASKS}" ]; then
    PNTASKS="-P ${NTASKS}"
    if [ "${NT}" ] && [ "0" != "$((NTASKS<=NT))" ]; then
      export OMP_NUM_THREADS=$((NT/NTASKS))
      export OMP_PROC_BIND=close
      export OMP_PLACES=cores
    else
      export OMP_NUM_THREADS=1
      export OMP_PROC_BIND=TRUE
    fi
  fi
  ${XARGS} </dev/stdin "${PNTASKS}" -I% bash -c \
    "_trap_err() { 1>&2 echo \" -> ERROR: \$(${BASENAME} %)\"; exit 1; }; trap '_trap_err' ERR; \
     if [ \"\$(${FILE} -bL --mime % | ${GREP} '^text/')\" ]; then source %; else %; fi"
  RESULT=$?
  if [ "0" != "${RESULT}" ]; then
    1>&2 echo "--------------------------------------------------------------------------------"
    exit ${RESULT}
  fi
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi
