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
SED=$(command -v sed)

# Note: avoid applying thread affinity (OMP_PROC_BIND or similar).
if [ "${BASENAME}" ] && [ "${XARGS}" ] && [ "${FILE}" ] && [ "${SED}" ]; then
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  NAME=$(${BASENAME} "$0" .sh)
  INFO=${HERE}/tool_cpuinfo.sh
  LG_DEFAULT="./${NAME}.log"
  QT_DEFAULT=0; SP_DEFAULT=2
  CONSUMED=0
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      if [ "0" != "${QT_DEFAULT}" ]; then QT_YESNO="yes"; else QT_YESNO="no"; fi
      echo "Usage: ${NAME}.sh [options]"
      echo "       -q|--quiet    [PEXEC_QT]: no info/progress output; default=${QT_YESNO} (stderr)"
      echo "       -o|--logfile  [PEXEC_LG]: combined stdout/stderr of commands (stdout)"
      echo "       -j|--nprocs N [PEXEC_NP]: number of processes (scaled by nscale)"
      echo "       -s|--nscale N [PEXEC_SP]: oversubscription; default=${SP_DEFAULT}"
      echo "       Environment [variables] will precede command line arguments."
      echo "       ${NAME}.sh reads stdin and spawns one task per line."
      echo
      echo "Example: seq 100 | xargs -I{} echo \"echo \\\"{}\\\"\" \\"
      echo "                 | tool_pexec.sh"
      echo
      exit 0;;
    -q|--quiet)
      QUIET=1
      shift 1;;
    -o|--logfile)
      LOGFILE=$2
      shift 2;;
    -j|--nprocs)
      CONSUMED=$((CONSUMED|1))
      NP=$2
      shift 2;;
    -s|--nscale)
      CONSUMED=$((CONSUMED|2))
      SP=$2
      shift 2;;
    *)
      if [ "0" = "$((CONSUMED&1))" ]; then
        CONSUMED=$((CONSUMED|1))
        NP=$1
      elif [ "0" = "$((CONSUMED&2))" ]; then
        CONSUMED=$((CONSUMED|2))
        SP=$1
      else
        1>&2 echo "ERROR: found spurious command line argument!"
        exit 1
      fi
      shift 1;;
    esac
  done
  LOGFILE=${PEXEC_LG:-${LOGFILE}}; if [ ! "${LOGFILE}" ]; then LOGFILE=${LG_DEFAULT}; fi
  QUIET=${PEXEC_QT:-${QUIET}}; if [ ! "${QUIET}" ]; then QUIET=${QT_DEFAULT}; fi
  NP=${PEXEC_NP:-${NP}}; SP=${PEXEC_SP:-${SP}}
  if [ ! "${LOGFILE}" ]; then LOGFILE="/dev/stdout"; fi
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
  if [ "0" = "${QUIET}" ]; then
    1>&2 echo "Execute with NPROCS=${NP} and OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    1>&2 echo
  fi
  ${XARGS} </dev/stdin 3>&2 2>&1 -P${NP} -I%% bash -c "set -eo pipefail; \
    _PEXEC_NARGS=\$(IFS=\" \"; set -- %%; echo \"\$#\"); \
    _PEXEC_BASENAME() { \
      local _PEXEC_BASENAME_PRE=\"\" _PEXEC_BASENAME_CMD=\"\" _PEXEC_BASENAME_ARGS=\"\"; \
      local _PEXEC_BASENAME_INPUT=\"\$*\" _PEXEC_BASENAME_WORDS=\"\";
      for WORD in \${_PEXEC_BASENAME_INPUT}; do \
        if [ \"\$(command -v \"\${WORD}\" 2>/dev/null)\" ]; then \
          _PEXEC_BASENAME_CMD=\$(${BASENAME} \"\${WORD}\"); \
          _PEXEC_BASENAME_PRE=\${_PEXEC_BASENAME_WORDS}; \
          _PEXEC_BASENAME_ARGS=\"\"; \
          continue; \
        fi; \
        _PEXEC_BASENAME_WORDS=\"\${_PEXEC_BASENAME_WORDS} \${WORD}\"; \
        _PEXEC_BASENAME_ARGS=\"\${_PEXEC_BASENAME_WORDS} \${WORD}\"; \
      done; \
      echo \"\${_PEXEC_BASENAME_CMD}\${_PEXEC_BASENAME_PRE}\${_PEXEC_BASENAME_ARGS}\" \
      | ${SED} 's/[^[:alnum:]]/_/g;y/ABCDEFGHIJKLMNOPQRSTUVWXYZ/abcdefghijklmnopqrstuvwxyz/;s/__*/_/g'; \
    }; \
    _PEXEC_TRAP_EXIT() { \
      local _PEXEC_TRAP_RESULT=\$?; \
      if [ \"0\" != \"\${_PEXEC_TRAP_RESULT}\" ]; then \
        local ERROR=\"ERROR\"; \
        if [ \"139\" = \"\${_PEXEC_TRAP_RESULT}\" ]; then ERROR=\"CRASH\"; fi; \
        if [ \"1\" = \"\${_PEXEC_NARGS}\" ]; then \
          1>&3 printf \" -> \${ERROR}[%03d]: \$(${BASENAME} %%)\n\" \${_PEXEC_TRAP_RESULT}; \
        else \
          1>&3 printf \" -> \${ERROR}[%03d]: %%\n\" \${_PEXEC_TRAP_RESULT}; \
        fi; \
        exit 1; \
      elif [ \"0\" = \"${QUIET}\" ]; then \
        if [ \"1\" = \"\${_PEXEC_NARGS}\" ]; then \
          1>&3 echo \" -> VALID[000]: \$(${BASENAME} %%)\"; \
        else \
          1>&3 echo \" -> VALID[000]: %%\"; \
        fi; \
      fi; \
    }; \
    trap '_PEXEC_TRAP_EXIT' EXIT; trap 'exit 0' TERM INT; \
    if [ \"1\" = \"\${_PEXEC_NARGS}\" ] && \
       [ \"\$(${FILE} -bL --mime %% | ${SED} -n '/^text\//p')\" ]; \
    then \
      source %%; \
    else \
      %%; \
    fi >\$(echo \"${LOGFILE}\" | ${SED} -n \"s/\(.*[^.]\)\(\..*\)/\1-\$(_PEXEC_BASENAME %%)\2/p\")"
  RESULT=$?
  if [ "0" != "${RESULT}" ]; then
    1>&2 echo "--------------------------------------------------------------------------------"
    exit ${RESULT}
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
