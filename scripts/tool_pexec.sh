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
CAT=$(command -v cat)
CUT=$(command -v cut)

# Note: avoid applying thread affinity (OMP_PROC_BIND or similar).
if [ "${XARGS}" ] && [ "${FILE}" ] && [ "${SED}" ] && [ "${CAT}" ] && [ "${CUT}" ]; then
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  NAME=$(echo "$0" | ${SED} 's/.*\///;s/\(.*\)\..*/\1/')
  INFO=${HERE}/tool_cpuinfo.sh
  PYTHON=$(command -v python3)
  LG_DEFAULT="./${NAME}.log"
  QT_DEFAULT=0; SP_DEFAULT=2
  CONSUMED=0
  # ensure proper permissions
  if [ "${UMASK}" ]; then
    UMASK_CMD="umask ${UMASK};"
    eval "${UMASK_CMD}"
  fi
  if [ ! "${PYTHON}" ]; then PYTHON=$(command -v python); fi
  if [ "${PYTHON}" ] && [ -e "${HERE}/libxsmm_utilities.py" ]; then
    TARGET=$(${PYTHON} "${HERE}/libxsmm_utilities.py")
  fi
  if [ "${PPID}" ] && [ "$(command -v ps)" ]; then
    PARENT=$(ps -o args= ${PPID} | ${SED} -n "s/[^[:space:]]*[[:space:]]*\(..*\)\.sh.*/\1/p")
    if [ "${PARENT}" ]; then
      if [ "${TARGET}" ] && [ -e "${PARENT}_${TARGET}.txt" ]; then
        WHITE=${PARENT}_${TARGET}.txt
      elif [ -e "${PARENT}.txt" ]; then
        WHITE=${PARENT}.txt
      fi
    fi
  fi
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      if [ "0" != "${QT_DEFAULT}" ]; then QT_YESNO="yes"; else QT_YESNO="no"; fi
      echo "Usage: ${NAME}.sh [options]"
      echo "       -q|--quiet    [PEXEC_QT]: no info/progress output; default=${QT_YESNO} (stderr)"
      echo "       -w|--white    [PEXEC_WL]: whitelist (default: ${WHITE:-filename not defined})"
      echo "       -o|--log      [PEXEC_LG]: combined stdout/stderr of commands (stdout)"
      echo "       -c|--cut      [PEXEC_CT]: cut output of each case (-f argument of cut)"
      echo "       -m|--min    N [PEXEC_MT]: minimum number of tasks; see --nth argument"
      echo "       -n|--nth    N [PEXEC_NT]: only every Nth task; randomized selection"
      echo "       -j|--nprocs N [PEXEC_NP]: number of processes (scaled by nscale)"
      echo "       -k|--ninner N [PEXEC_NI]: inner processes (N=0: auto, N=-1: max)"
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
    -w|--white)
      WHITE=$2
      shift 2;;
    -o|--log)
      LOG=$2
      shift 2;;
    -c|--cut)
      CT=$2
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
    -k|--ninner)
      CONSUMED=$((CONSUMED|2))
      NI=$2
      shift 2;;
    -s|--nscale)
      CONSUMED=$((CONSUMED|4))
      SP=$2
      shift 2;;
    *)
      if [ "0" = "$((CONSUMED&1))" ]; then
        CONSUMED=$((CONSUMED|1))
        NP=$1
      elif [ "0" = "$((CONSUMED&2))" ]; then
        CONSUMED=$((CONSUMED|2))
        NI=$1
      elif [ "0" = "$((CONSUMED&4))" ]; then
        CONSUMED=$((CONSUMED|4))
        SP=$1
      else
        1>&2 echo "ERROR: found spurious command line argument!"
        exit 1
      fi
      shift 1;;
    esac
  done
  NIFIX=0
  COUNTER=0
  LOG=${PEXEC_LG:-${LOG}}; if [ ! "${LOG}" ]; then LOG=${LG_DEFAULT}; fi
  QUIET=${PEXEC_QT:-${QUIET}}; if [ ! "${QUIET}" ]; then QUIET=${QT_DEFAULT}; fi
  NI=${PEXEC_NI:-${NI}}; if [ ! "${NI}" ]; then NI=${OMP_NUM_THREADS}; else NIFIX=1; fi
  NP=${PEXEC_NP:-${NP}}; NI=${PEXEC_NI:-${NI}}; NJ=$((0<NI?NI:1)); SP=${PEXEC_SP:-${SP}}
  CT=${PEXEC_CT:-${CT}}; NTH=${PEXEC_NT:-${NTH}}
  MIN=${PEXEC_MT:-${MIN}}; MIN=$((1<MIN?MIN:1))
  WHITE=${PEXEC_WL:-${WHITE}}
  if [ "${WHITE}" ] && [ ! -e "${WHITE}" ]; then
    1>&2 echo "ERROR: \"${WHITE}\" whitelist file not found!"
    exit 1
  fi
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
    NQ=${NP}
    if [ "${NT}" ] && [ "0" != "$((NP<=NT))" ]; then
      if [ "${NJ}" ] && [ "0" != "$((NJ<=NT))" ]; then
        NP=$(((NP+NJ-1)/NJ))
      elif [ "0" != "$((COUNTER<NP))" ]; then
        NJ=$((NT/COUNTER))
      else
        NJ=$((NT/NP))
      fi
    else
      NJ=1
    fi
  else
    NP=0; NQ=0; NJ=1
  fi
  # sanitize outer parallelism
  if [ "0" != "$((COUNTER<NP))" ]; then NP=${COUNTER}; fi
  # sanitize inner parallelism
  if [ "0" != "$((0<NP))" ]; then
    NK=$(((NQ+NP-1)/NP))
  else
    NK=NJ
  fi
  # select inner parallelism
  export OMP_NUM_THREADS=1
  if [ "1" != "${NK}" ] && [ "/dev/null" = "${LOG}" ]; then
    if [ "${NIFIX}" ] && [ "0" != "${NIFIX}" ]; then
      if [ "0" != "$((0<NI))" ]; then
        export OMP_NUM_THREADS=$(((NK+NJ-1)/NJ))
      elif [ "0" = "${NI}" ]; then
        NJ=$(((NK+NJ-1)/NJ))
      else # NI<0
        NJ=${NQ}; NP=1
      fi
    else
      export OMP_NUM_THREADS=${NK}
    fi
  fi
  if [ "0" != "$((1!=NP))" ]; then
    unset OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY
  fi
  if [ "1" != "${NJ}" ]; then
    export PEXEC_NI=${NJ}
  else
    unset PEXEC_NI
  fi
  if [[ ${LOG} != /dev/* ]]; then
    LOG_OUTER=/dev/stdout
  else
    LOG_OUTER=${LOG}
  fi
  PEXEC_SCRIPT="set -eo pipefail; ${UMASK_CMD} \
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
      | if [ \"${CT}\" ]; then ${CUT} -d_ -f\"${CT}\"; else ${CAT}; fi; \
    }; \
    _PEXEC_PRETTY=\$(_PEXEC_MAKE_PRETTY \$0); \
    _PEXEC_TRAP_EXIT() { \
      local RESULT=\$?; \
      if [ \"0\" != \"\${RESULT}\" ]; then \
        if [ \"${WHITE}\" ] && [ \"\$(${SED} -n \"/\${_PEXEC_PRETTY}/p\" ${WHITE})\" ]; then \
          1>&2 printf \" -> WHITE[%03d]: \${_PEXEC_PRETTY}\n\" \${RESULT}; exit 0; \
        else \
          local ERROR=\"ERROR\"; \
          if [ \"139\" = \"\${RESULT}\" ]; then ERROR=\"CRASH\"; fi; \
          1>&2 printf \" -> \${ERROR}[%03d]: \${_PEXEC_PRETTY}\n\" \${RESULT}; exit 1; \
        fi; \
        exit 0; \
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
  if [ "0" = "${QUIET}" ]; then
    if [ "$(command -v tr)" ]; then
      if [ "${TARGET}" ]; then AT=$(echo "@${TARGET}" | tr "[:lower:]" "[:upper:]"); fi
      if [ "${BUILDKITE_LABEL}" ]; then
        LABEL="$(echo "${BUILDKITE_LABEL}" | tr -s "[:punct:][:space:]" - \
        | ${SED} 's/^-//;s/-$//' | tr "[:lower:]" "[:upper:]")${AT} "
      else
        LABEL="$(basename "$(pwd -P)" | tr "[:lower:]" "[:upper:]")${AT} "
      fi
    fi
    1>&2 echo "Execute ${LABEL}with NTASKS=${COUNTER}, NPROCS=${NP}x${NJ}, and OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    if [ "${WHITE}" ]; then
      1>&2 echo "Whitelist: ${WHITE}"
    fi
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
