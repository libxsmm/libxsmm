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

XARGS=$(command -v xargs)

if [ "${XARGS}" ] && [ "$(command -v basename)" ]; then
  NC=$1
  if [ ! "${NC}" ] || [ "0" = "${NC}" ]; then
    HERE=$(cd "$(dirname "$0")" && pwd -P)
    CPU=${HERE}/tool_cpuinfo.sh
    if [ -e "${CPU}" ]; then
      NC=$(${CPU} -nt)
    fi
  fi
  if [ "${NC}" ]; then
    PNC="-P ${NC}"
  fi
  OMP_NUM_THREADS=1 \
  ${XARGS} </dev/stdin "${PNC}" -I% bash -c \
    "_trap_err() { 1>&2 echo \" -> ERROR: \$(basename %)\"; exit 1; }; trap '_trap_err' ERR; source %"
  RESULT=$?
  if [ "0" != "${RESULT}" ]; then
    1>&2 echo "--------------------------------------------------------------------------------"
    exit ${RESULT}
  fi
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi
