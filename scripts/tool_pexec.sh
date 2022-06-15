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
# Note: avoid applying thread affinity (OMP_PROC_BIND or similar).
if [ "${BASENAME}" ] && [ "${XARGS}" ] && [ "${FILE}" ] && [ "${GREP}" ]; then
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  INFO=${HERE}/tool_cpuinfo.sh
  SP_DEFAULT=2
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      echo "Usage: $0 [options]"
      echo "       -h|--help: this help output"
      echo "       -v|--verbose: more on stderr (PEXEC_VERBOSE)"
      echo "       -j|--nprocs N: number of processes (PEXEC_NP)"
      echo "       -s|--nscale N: oversubscription (PEXEC_SP)"
      echo
      echo "Example: seq 100 | xargs -I{} echo \"echo \\\"{}\\\"\" \\"
      echo "                 | tool_pexec.sh"
      echo
      exit 0;;
    -v|--verbose)
      VERBOSE=${PEXEC_VERBOSE:-1}
      shift 1;;
    -j|--nprocs)
      NP=${PEXEC_NP:-$2}
      shift 2;;
    -s|--nscale)
      SP=${PEXEC_SP:-$2}
      shift 2;;
    *)
      break;;
    esac
  done
  if [ -e "${INFO}" ]; then
    NC=$(${INFO} -nc); NT=$(${INFO} -nt)
  fi
  if [ ! "${NP}" ] || [ "0" != "$((1>NP))" ]; then
    NP=${NC}
  fi
  if [ "${NP}" ]; then
    if [ ! "${SP}" ]; then
      NP=$((NP*SP_DEFAULT))
      if [ "${NT}" ] && [ "0" = "$((NP<=NT))" ]; then
        NP=${NT}
      fi
    elif [ "0" != "$((1<SP))" ]; then
      NP=$((NP*SP))
    fi
    if [ "${NT}" ] && [ "0" != "$((NP<=NT))" ]; then
      if [ "${OMP_NUM_THREADS}" ] && [ "0" != "$((OMP_NUM_THREADS<=NT))" ]; then
        NP=$(((NP+OMP_NUM_THREADS-1)/OMP_NUM_THREADS))
      else
        export OMP_NUM_THREADS=$((NT/NP))
      fi
    else
      export OMP_NUM_THREADS=1
    fi
  else
    export OMP_NUM_THREADS=1
    NP=0
  fi
  if [ "0" != "$((1!=NP))" ]; then
    unset OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY
  fi
  ${XARGS} </dev/stdin -P${NP} -I% bash -c "set -e; \
    _PEXEC_NARGS=\$(IFS=\" \"; set -- %; echo \"\$#\"); \
    _PEXEC_TRAP_EXIT() { \
      if [ \"0\" != \"\$?\" ]; then \
        if [ \"1\" = \"\${_PEXEC_NARGS}\" ]; then \
          1>&2 echo \" -> ERROR: \$(${BASENAME} %)\"; \
        else \
          1>&2 echo \" -> ERROR: %\"; \
        fi; \
        exit 1; \
      elif [ \"${VERBOSE}\" ] && [ \"0\" != \"${VERBOSE}\" ]; then \
        if [ \"1\" = \"\${_PEXEC_NARGS}\" ]; then \
          1>&2 echo \" -> OK: \$(${BASENAME} %)\"; \
        else \
          1>&2 echo \" -> OK: %\"; \
        fi; \
      fi; \
    }; \
    trap '_PEXEC_TRAP_EXIT' EXIT; \
    if [ \"1\" = \"\${_PEXEC_NARGS}\" ] && \
       [ \"\$(${FILE} -bL --mime % | ${GREP} '^text/')\" ]; \
    then \
      source %; \
    else \
      %; \
    fi"
  RESULT=$?
  if [ "0" != "${RESULT}" ]; then
    1>&2 echo "--------------------------------------------------------------------------------"
    exit ${RESULT}
  fi
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi
