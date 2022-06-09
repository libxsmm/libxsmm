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
#set -eo pipefail
#set -e

BASENAME=$(command -v basename)
XARGS=$(command -v xargs)
FILE=$(command -v file)
GREP=$(command -v grep)

# Usage: tool_pexec.sh [<num-tasks>] [<oversubscription-factor>]
# Use all cores and Hyperthreads like tool_pexec.sh 0 2.
# Environment variables PEXEC_NP=<num-tasks>, and
#                       PEXEC_SP=<oversubscription-factor>
# precede command line arguments.
# The script reads stdin and intents to spawn one task per line.
# Example: seq 100 | xargs -I{} echo "echo \"{}\"" \
#                  | tool_pexec.sh
# Note: avoid applying thread affinity (OMP_PROC_BIND or similar).
if [ "${BASENAME}" ] && [ "${XARGS}" ] && [ "${FILE}" ] && [ "${GREP}" ]; then
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  INFO=${HERE}/tool_cpuinfo.sh
  NP=${PEXEC_NP:-$1}; SP=${PEXEC_SP:-$2}; SP_DEFAULT=2
  if [ -e "${INFO}" ]; then
    NC=$(${INFO} -nc)
    NT=$(${INFO} -nt)
  fi
  if [ ! "${NP}" ] || [ "0" = "$((0<NP))" ]; then
    NP=$(((NC*SP_DEFAULT)<=NT?(NC*SP_DEFAULT):NC))
  fi
  if [ "${NP}" ]; then
    if [ "${SP}" ] && [ "0" != "$((1<SP))" ]; then
      NP=$((NP*SP))
    fi
    if [ "${NT}" ] && [ "0" != "$((NP<=NT))" ]; then
      export OMP_NUM_THREADS=$((NT/NP))
    else
      export OMP_NUM_THREADS=1
    fi
  else
    export OMP_NUM_THREADS=1
    NP=0
  fi
  unset OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY
  ${XARGS} </dev/stdin -P${NP} -I% bash -c "set -e; \
    _trap_exit() { \
      if [ \"0\" != \"\$?\" ]; then \
        1>&2 echo \" -> ERROR: \$(${BASENAME} %)\"; \
        exit 1; \
      fi; \
    }; \
    trap '_trap_exit' EXIT; \
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
