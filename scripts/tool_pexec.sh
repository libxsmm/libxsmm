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

XARGS=$(command -v xargs)
FILE=$(command -v file)
SED=$(command -v sed)

# Note: avoid applying thread affinity (OMP_PROC_BIND or similar).
if [ "${XARGS}" ] && [ "${FILE}" ] && [ "${SED}" ]; then
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  NAME=$(echo "$0" | ${SED} 's/.*\///;s/\(.*\)\..*/\1/')
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
      echo "       -o|--log      [PEXEC_LG]: combined stdout/stderr of commands (stdout)"
      echo "       -c|--cut      [PEXEC_CT]: cut name of each case (-f argument of cut)"
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
    -o|--log)
      LOG=$2
      shift 2;;
    -c|--cut)
      CUT=$2
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
  LOG=${PEXEC_LG:-${LOG}}; if [ ! "${LOG}" ]; then LOG=${LG_DEFAULT}; fi
  QUIET=${PEXEC_QT:-${QUIET}}; if [ ! "${QUIET}" ]; then QUIET=${QT_DEFAULT}; fi
  CUT=${PEXEC_CT:-${CUT}}; NP=${PEXEC_NP:-${NP}}; SP=${PEXEC_SP:-${SP}}
  if [ ! "${LOG}" ]; then LOG="/dev/stdout"; fi
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
  if [[ ${LOG} != /dev/* ]]; then
    LOG_OUTER=/dev/stdout
  else
    LOG_OUTER=${LOG}
  fi
  ${XARGS} </dev/stdin >"${LOG_OUTER}" -P${NP} -I{} bash -c "set -eo pipefail; \
    _PEXEC_REPLSTRING=\$0; \
    _PEXEC_CMDPRETTY() { \
      local _PEXEC_CMDPRETTY_HERE=\$(pwd -P | ${SED} 's/\//\\\\\//g'); \
      local _PEXEC_CMDPRETTY_PRE=\"\" _PEXEC_CMDPRETTY_CMD=\"\" _PEXEC_CMDPRETTY_ARGS=\"\"; \
      local _PEXEC_CMDPRETTY_INPUT=\"\$*\" _PEXEC_CMDPRETTY_WORDS=\"\"; \
      for WORD in \${_PEXEC_CMDPRETTY_INPUT}; do \
        local _PEXEC_CMDPRETTY_WORD=\$(echo \"\${WORD}\" \
        | ${SED} \"s/\/\.\//\//;s/.*\${_PEXEC_CMDPRETTY_HERE}\///\" \
        | ${SED} 's/\(.*\)\..*/\1/'); \
        if [ \"\$(command -v \"\${WORD}\" 2>/dev/null)\" ]; then \
          _PEXEC_CMDPRETTY_PRE=\${_PEXEC_CMDPRETTY_WORDS}; \
          _PEXEC_CMDPRETTY_CMD=\${_PEXEC_CMDPRETTY_WORD}; \
          _PEXEC_CMDPRETTY_ARGS=\"\"; \
          continue; \
        fi; \
        _PEXEC_CMDPRETTY_WORDS=\"\${_PEXEC_CMDPRETTY_WORDS} \${_PEXEC_CMDPRETTY_WORD}\"; \
        _PEXEC_CMDPRETTY_ARGS=\"\${_PEXEC_CMDPRETTY_ARGS} \${_PEXEC_CMDPRETTY_WORD}\"; \
      done; \
      echo \"\${_PEXEC_CMDPRETTY_CMD}\${_PEXEC_CMDPRETTY_PRE}\${_PEXEC_CMDPRETTY_ARGS}\" \
      | ${SED} 's/[^[:alnum:]]/_/g;y/ABCDEFGHIJKLMNOPQRSTUVWXYZ/abcdefghijklmnopqrstuvwxyz/;s/__*/_/g;s/^_//;s/_$//' \
      | if [ \"${CUT}\" ]; then cut -d_ -f\"${CUT}\"; else cat; fi; \
    }; \
    _PEXEC_CMDLINE=\"\${_PEXEC_REPLSTRING}\"; _PEXEC_BASENAME=\$(_PEXEC_CMDPRETTY \${_PEXEC_REPLSTRING}); \
    _PEXEC_TRAP_EXIT() { \
      local _PEXEC_TRAP_RESULT=\$?; \
      if [ \"0\" != \"\${_PEXEC_TRAP_RESULT}\" ]; then \
        local ERROR=\"ERROR\"; \
        if [ \"139\" = \"\${_PEXEC_TRAP_RESULT}\" ]; then ERROR=\"CRASH\"; fi; \
        1>&2 printf \" -> \${ERROR}[%03d]: \${_PEXEC_BASENAME}\n\" \${_PEXEC_TRAP_RESULT}; \
        exit 1; \
      elif [ \"0\" = \"${QUIET}\" ]; then \
        1>&2 echo \" -> VALID[000]: \${_PEXEC_BASENAME}\"; \
      fi; \
    }; \
    if [[ ${LOG} != /dev/* ]]; then \
      _PEXEC_LOG=\$(echo \"${LOG}\" | ${SED} -n \"s/\(.*[^.]\)\(\..*\)/\1-\${_PEXEC_BASENAME}\2/p\"); \
    else \
      _PEXEC_LOG=/dev/stdout; \
    fi; \
    trap '_PEXEC_TRAP_EXIT' EXIT; trap 'exit 0' TERM INT; \
    if [ \"\$(${FILE} -bL --mime \"\${_PEXEC_CMDLINE%% *}\" | ${SED} -n '/^text\//p')\" ]; then \
      source \${_PEXEC_REPLSTRING}; \
    else \
      \${_PEXEC_REPLSTRING}; \
    fi >\"\${_PEXEC_LOG}\" 2>&1" "{}"
  RESULT=$?
  if [ "0" != "${RESULT}" ]; then
    1>&2 echo "--------------------------------------------------------------------------------"
    exit ${RESULT}
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
