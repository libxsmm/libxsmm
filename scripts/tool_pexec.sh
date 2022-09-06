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
      echo "       -c|--cut      [PEXEC_CT]: cut output of each case (-f argument of cut)"
      echo "       -m|--min    N [PEXEC_MT]: minimum number of tasks; see --nth argument"
      echo "       -n|--nth    N [PEXEC_NT]: only every Nth task; randomized selection"
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
    -m|--min)
      MIN=$2
      shift 2;;
    -n|--nth)
      NTH=$2
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
  NTH=${PEXEC_NT:-${NTH}}; MIN=${PEXEC_MT:-${MIN}}; MIN=$((1<MIN?MIN:1))
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
  if [[ ${LOG} != /dev/* ]]; then
    LOG_OUTER=/dev/stdout
  else
    LOG_OUTER=${LOG}
  fi
  PEXEC_SCRIPT="set -eo pipefail; \
    _PEXEC_MAKE_PRETTY() { \
      local HERE PRE CMD ARGS WORDS INPUT=\$*; \
      HERE=\$(pwd -P | ${SED} 's/\//\\\\\//g'); \
      for WORD in \${INPUT}; do \
        local PRETTY; \
        PRETTY=\$(echo \"\${WORD}\" \
        | ${SED} \"s/\/\.\//\//;s/.*\${HERE}\///\" \
        | ${SED} 's/\(.*\)\..*/\1/'); \
        if [ \"\$(command -v \"\${WORD}\" 2>/dev/null)\" ]; then \
          PRE=\${WORDS}; CMD=\${PRETTY}; ARGS=\"\"; \
          continue; \
        fi; \
        WORDS=\"\${WORDS} \${PRETTY}\"; \
        ARGS=\"\${ARGS} \${PRETTY}\"; \
      done; \
      echo \"\${CMD}\${PRE}\${ARGS}\" \
      | ${SED} 's/[^[:alnum:]]/_/g;y/ABCDEFGHIJKLMNOPQRSTUVWXYZ/abcdefghijklmnopqrstuvwxyz/;s/__*/_/g;s/^_//;s/_$//' \
      | if [ \"${CUT}\" ]; then cut -d_ -f\"${CUT}\"; else cat; fi; \
    }; \
    _PEXEC_PRETTY=\$(_PEXEC_MAKE_PRETTY \$0); \
    _PEXEC_TRAP_EXIT() { \
      local RESULT=\$?; \
      if [ \"0\" != \"\${RESULT}\" ]; then \
        local ERROR=\"ERROR\"; \
        if [ \"139\" = \"\${RESULT}\" ]; then ERROR=\"CRASH\"; fi; \
        1>&2 printf \" -> \${ERROR}[%03d]: \${_PEXEC_PRETTY}\n\" \${RESULT}; \
        exit 1; \
      elif [ \"0\" = \"${QUIET}\" ]; then \
        1>&2 echo \" -> VALID[000]: \${_PEXEC_PRETTY}\"; \
      fi; \
    }; \
    if [[ ${LOG} != /dev/* ]]; then \
      _PEXEC_LOG=\$(echo \"${LOG}\" | ${SED} -n \"s/\(.*[^.]\)\(\..*\)/\1-\${_PEXEC_PRETTY}\2/p\"); \
    else \
      _PEXEC_LOG=/dev/stdout; \
    fi; \
    trap '_PEXEC_TRAP_EXIT' EXIT; trap 'exit 0' TERM INT; \
    if [ \"\$(${FILE} -bL --mime \"\${0%% *}\" | ${SED} -n '/^text\//p')\" ]; then \
      source \$0; \
    else \
      \$0; \
    fi >\"\${_PEXEC_LOG}\" 2>&1"
  COUNTER=0
  while read -r LINE; do
    if [ ! "${NTH}" ] || [ "0" != "$((1>=NTH))" ] || [ "0" = "$(((RANDOM+1)%NTH))" ]; then
      COUNTER=$((COUNTER+1))
      COUNTED="${COUNTED}"$'\n'"${LINE}"
    elif [ "0" != "$((COUNTER<MIN))" ]; then
      ATLEAST="${ATLEAST}"$'\n'"${LINE}"
    fi
  done
  IFS=$'\n' && for LINE in ${ATLEAST}; do
    if [ "0" != "$((MIN<=COUNTER))" ]; then break; fi
    COUNTED="${COUNTED}"$'\n'"${LINE}"
    COUNTER=$((COUNTER+1))
  done
  if [ "0" = "${QUIET}" ]; then
    if [ "$(command -v tr)" ]; then
      if [ "${BUILDKITE_LABEL}" ]; then
        LABEL="$(echo "${BUILDKITE_LABEL}" | tr -s "[:punct:][:space:]" - \
        | sed 's/^-//;s/-$//' | tr "[:lower:]" "[:upper:]") "
      else
        LABEL="$(basename "$(pwd -P)" \
        | tr "[:lower:]" "[:upper:]") "
      fi
    fi
    1>&2 echo "Execute ${LABEL}with NTASKS=${COUNTER}, NPROCS=${NP}, and OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  fi
  echo -e "${COUNTED}" | ${XARGS} >"${LOG_OUTER}" -P${NP} -I{} bash -c "${PEXEC_SCRIPT}" "{}"
  RESULT=$?
  if [ "0" != "${RESULT}" ]; then
    1>&2 echo "--------------------------------------------------------------------------------"
    exit ${RESULT}
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
