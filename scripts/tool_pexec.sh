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
# shellcheck disable=SC2023
#set -eo pipefail

MKTEMP=$(command -v mktemp)
XARGS=$(command -v xargs)
FILE=$(command -v file)
DATE=$(command -v date)
SED=$(command -v sed)
CAT=$(command -v cat)
CUT=$(command -v cut)

if [ "${DATE}" ]; then
  NSECS=$(date +%s)
fi

# Note: avoid applying thread affinity (OMP_PROC_BIND or similar).
if [ "${MKTEMP}" ] && [ "${XARGS}" ] && [ "${FILE}" ] && [ "${SED}" ] && [ "${CAT}" ] && [ "${CUT}" ]; then
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  NAME=$(echo "$0" | ${SED} 's/.*\///;s/\(.*\)\..*/\1/')
  INFO=${HERE}/tool_cpuinfo.sh
  PYTHON=$(command -v python3 || true)
  FLOCK=${HERE}/../.flock.sh
  #LG_DEFAULT=${NAME}.log
  LG_DEFAULT=/dev/null
  XF_DEFAULT=1; BL_DEFAULT=1; QT_DEFAULT=0
  SP_DEFAULT=2; MT_DEFAULT=1; CONSUMED=0
  # ensure proper permissions
  if [ "${UMASK}" ]; then
    UMASK_CMD="umask ${UMASK};"
    eval "${UMASK_CMD}"
  fi
  if [ -e "${INFO}" ]; then
    NC=$(${INFO} -nc); NT=$(${INFO} -nt); HT=$(${INFO} -ht)
    if [ "0" != "$((HT<SP_DEFAULT))" ]; then SP_DEFAULT=${HT}; fi
  fi
  # ensure consistent sort
  export LC_ALL=C
  if [ ! "${PYTHON}" ]; then PYTHON=$(command -v python || true); fi
  if [ "${PYTHON}" ] && [ -e "${HERE}/libxsmm_utilities.py" ]; then
    TARGET=$(${PYTHON} "${HERE}/libxsmm_utilities.py")
  fi
  if [ "${PPID}" ] && [ "$(command -v ps)" ]; then
    PARENT=$(ps -o args= ${PPID} | ${SED} -n "s/[^[:space:]]*[[:space:]]*\(..*\)\.sh.*/\1/p")
    if [ "${PARENT}" ]; then
      if [ "${TARGET}" ]; then
        FNAME=${PARENT}_${TARGET}.txt
        if [ -e "${FNAME}" ]; then
          ALLOW=${FNAME}
        fi
      fi
      if [ ! "${ALLOW}" ] && [ -e "${PARENT}.txt" ]; then
        ALLOW=${PARENT}.txt
      fi
    fi
  fi
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      if [ "0" != "${PEXEC_QT:-${QUIET:-${QT_DEFAULT}}}" ]; then QT_YESNO="yes"; else QT_YESNO="no"; fi
      echo "Usage: ${NAME}.sh [options]"
      echo "       -x|--xfail  N* [PEXEC_XF]: results 2..254 are failures; default: ${PEXEC_XF:-${XFAIL:-${XF_DEFAULT}}}"
      echo "       -y|--shaky  N* [PEXEC_BL]: allowed failures must fail; default: ${PEXEC_BL:-${SHAKY:-${BL_DEFAULT}}}"
      echo "       -w|--allow  F  [PEXEC_WL]: allowed failures (filename); default: ${PEXEC_WL:-${ALLOW:--}}"
      echo "       -u|--build  F* [PEXEC_UP]: collect failures (filename); default: ${PEXEC_UP:-${FNAME:--}}"
      echo "       -q|--quiet  -  [PEXEC_QT]: no progress output (valid cases); default: ${QT_YESNO}"
      echo "       -o|--log    F  [PEXEC_LG]: combined stdout/stderr; default: ${PEXEC_LG:-${LOG:-${LG_DEFAULT}}}"
      echo "       -c|--cut    S  [PEXEC_CT]: cut name of case (-f argument of \"cut\")"
      echo "       -m|--min    N  [PEXEC_MT]: minimum number of tasks; see --nth argument"
      echo "       -n|--nth    N  [PEXEC_NT]: only every Nth task; randomized selection"
      echo "       -j|--nprocs N  [PEXEC_NP]: number of processes (scaled by nscale)"
      echo "       -k|--ninner N  [PEXEC_NI]: inner processes (N=0: auto, N=-1: max)"
      echo "       -s|--nscale N  [PEXEC_SP]: subscription; default: ${PEXEC_SP:-${SP:-${SP_DEFAULT}}}"
      echo "                                  under-subscription (N<0)"
      echo "       Environment [variables] will precede command line arguments."
      echo "       ${NAME}.sh reads stdin and spawns one task per line."
      echo
      echo "Example: seq 100 | xargs -I{} echo \"echo \\\"{}\\\"\" \\"
      echo "                 | $0"
      echo
      exit 0;;
    -x|--xfail)
      XFAIL=$2
      if [ ! "${XFAIL}" ] || [ "-" = "${XFAIL:0:1}" ]; then
        XFAIL=1; shift 1
      else
        shift 2
      fi;;
    -y|--shaky)
      SHAKY=$2
      if [ ! "${SHAKY}" ] || [ "-" = "${SHAKY:0:1}" ]; then
        SHAKY=1; shift 1
      else
        shift 2
      fi;;
    -w|--allow)
      ALLOW=$2
      shift 2;;
    -u|--build)
      BUILD=$2
      if [ ! "${BUILD}" ] || [ "-" = "${BUILD:0:1}" ]; then
        BUILD="-"; shift 1
      else
        shift 2
      fi;;
    -q|--quiet)
      QUIET=1
      shift 1;;
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
  NIFIX=0; TOTAL=0; COUNTER=0; PEXEC_IL=1
  LOG=${PEXEC_LG:-${LOG}}; if [ ! "${LOG}" ]; then LOG=${LG_DEFAULT}; fi
  XFAIL=${PEXEC_XF:-${XFAIL}}; if [ ! "${XFAIL}" ]; then XFAIL=${XF_DEFAULT}; fi
  SHAKY=${PEXEC_BL:-${SHAKY}}; if [ ! "${SHAKY}" ]; then SHAKY=${BL_DEFAULT}; fi
  QUIET=${PEXEC_QT:-${QUIET}}; if [ ! "${QUIET}" ]; then QUIET=${QT_DEFAULT}; fi
  NI=${PEXEC_NI:-${NI}}; if [ ! "${NI}" ]; then NI=${OMP_NUM_THREADS:-1}; else NIFIX=1; fi
  NP=${PEXEC_NP:-${NP}}; NJ=$((0<NI?NI:1)); SP=${PEXEC_SP:-${SP}}
  CT=${PEXEC_CT:-${CT}}; NTH=${PEXEC_NT:-${NTH}}; MIN=${PEXEC_MT:-${MIN}};
  if [ ! "${MIN}" ]; then MIN=${MT_DEFAULT}; else MIN=$((1<MIN?MIN:1)); MT_DEFAULT=0; fi
  BUILD=${PEXEC_UP:-${BUILD}}; if [ "-" = "${BUILD:0:1}" ]; then BUILD=${FNAME}; fi
  ALLOW=${PEXEC_WL:-${ALLOW}}; if [ "${BUILD}" ]; then unset ALLOW; fi
  MAKE_PRETTY_FUNCTION="_PEXEC_MAKE_PRETTY() { \
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
    $(if [ "${CT}" ]; then echo "| ${CUT} -d_ -f${CT}"; fi); \
  }"
  if [ "${ALLOW}" ] && [ ! -e "${ALLOW}" ]; then
    1>&2 echo "ERROR: \"${ALLOW}\" file not found!"
    exit 1
  fi
  while read -r LINE; do
    if [[ ! ${LINE} =~ ^[[:space:]]*# ]]; then # ignore comments
      if [ ! "${ALLOW}" ] || [ "0" != "$((1>=NTH))" ]; then PRETTY="";
      else PRETTY=$(eval "${MAKE_PRETTY_FUNCTION}; echo \"\$(_PEXEC_MAKE_PRETTY ${LINE})\""); fi
      if [ ! "${PRETTY}" ] || [ ! "$(${SED} -En "/^${PRETTY}([[:space:]]|$)/p" "${ALLOW}")" ]; then
        if [ ! "${NTH}" ] || [ "0" != "$((1>=NTH))" ] || [ "0" = "$(((RANDOM+1)%NTH))" ]; then
          if [ "${COUNTED}" ]; then COUNTED="${COUNTED}"$'\n'"${LINE}"; else COUNTED=${LINE}; fi
          COUNTER=$((COUNTER+1))
        elif [ "0" != "$((COUNTER<MIN))" ]; then
          ATLEAST="${ATLEAST}"$'\n'"${LINE}"
        fi
        TOTAL=$((TOTAL+1))
      fi
    fi
  done
  if [ "0" != "${MT_DEFAULT}" ]; then
    if [ "${NTH}" ] && [ "0" != "$((1<NTH))" ]; then
      TOTAL=$(((TOTAL+NTH-1)/NTH))
    else
      TOTAL=${COUNTER}
    fi
  else
    TOTAL=${MIN}
  fi
  IFS=$'\n' && for LINE in ${ATLEAST}; do
    if [ "0" != "$((TOTAL<=COUNTER))" ]; then break; fi
    COUNTED="${COUNTED}"$'\n'"${LINE}"
    COUNTER=$((COUNTER+1))
  done && unset IFS
  PEXEC_SCRARG="\$0"
  if [ "${COUNTER}" != "${TOTAL}" ] || [ "0" = "${PEXEC_IL}" ]; then
    if [ "0" = "${PEXEC_IL}" ]; then
      PEXEC_SCRIPT=$(${MKTEMP})
      PEXEC_SCRARG="\$*"
    fi
    ATLEAST=${COUNTED}; COUNTED=""; COUNTER=0
    IFS=$'\n' && for LINE in ${ATLEAST}; do
      if [ "0" != "$((TOTAL<=COUNTER))" ]; then break; fi
      if [ "${COUNTED}" ]; then COUNTED="${COUNTED}"$'\n'"${LINE}"; else COUNTED=${LINE}; fi
      COUNTER=$((COUNTER+1))
    done && unset IFS
  fi
  trap 'rm -f ${NAME}.txt ${PEXEC_SCRIPT}' EXIT
  unset ATLEAST
  if [ ! "${LOG}" ]; then LOG="/dev/stdout"; fi
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
    elif [ "0" != "$((0>SP))" ]; then
      NP=$(((NP-SP-1)/-SP))
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
  if [[ ("1" != "${NK}") && ("/dev/null" = "${LOG}" || "0" != "$((0<NI))") ]]; then
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
  PEXEC_INLINE="set -eo pipefail; ${UMASK_CMD} \
    ${MAKE_PRETTY_FUNCTION}; \
    _PEXEC_PRETTY=\$(_PEXEC_MAKE_PRETTY \"${PEXEC_SCRARG}\"); \
    _PEXEC_TRAP_EXIT() { \
      local RESULT=\$?; \
      if [ \"0\" != \"\${RESULT}\" ]; then \
        local PERMIT=\$((0==${XFAIL}||1==RESULT||255==RESULT)); \
        if [ \"${ALLOW}\" ] && [ \"0\" != \"\${PERMIT}\" ] && \
           [ \"\$(${SED} -En \"/^\${_PEXEC_PRETTY}([[:space:]]|$)/p\" \"${ALLOW}\")\" ]; \
        then \
          if [ \"0\" = \"${QUIET}\" ]; then 1>&2 printf \" -> ALLOW[%03d]: \${_PEXEC_PRETTY}\n\" \${RESULT}; fi; \
        else \
          local ERROR=\"ERROR\"; \
            if [ \"132\" = \"\${RESULT}\" ]; then ERROR=\"ILLEG\"; \
          elif [ \"139\" = \"\${RESULT}\" ]; then ERROR=\"CRASH\"; fi; \
          1>&2 printf \" -> \033[91m\${ERROR}\033[0m[%03d]: \${_PEXEC_PRETTY}\n\" \${RESULT}; \
          if [ \"${BUILD}\" ] && [ \"0\" != \"\${PERMIT}\" ] && [ \"0\" != \"${XFAIL}\" ]; then \
            ${FLOCK} ${BUILD} \"echo \${_PEXEC_PRETTY} >>${BUILD}\"; \
          else \
            exit 1; \
          fi; \
        fi; \
      elif [ ! \"${ALLOW}\" ] || [ \"0\" = \"${SHAKY}\" ] || [ \"no\" = \"${SHAKY}\" ] || \
           [ ! \"\$(${SED} -En \"/^\${_PEXEC_PRETTY}([[:space:]]|$)/p\" \"${ALLOW}\")\" ]; \
      then \
        if [ \"0\" = \"${QUIET}\" ]; then 1>&2 echo -e \" -> \033[92mVALID\033[0m[000]: \${_PEXEC_PRETTY}\"; fi; \
      else \
        1>&2 printf \" -> \033[33mSHAKY\033[0m[%03d]: \${_PEXEC_PRETTY}\n\" \${RESULT}; exit 1; \
      fi; \
      exit 0; \
    }; \
    if [[ ${LOG} != /dev/* ]]; then \
      _PEXEC_LOG=\$(echo \"${LOG}\" | ${SED} -n \"s/\(.*[^.]\)\(\..*\)/\1-\${_PEXEC_PRETTY}\2/p\"); \
    else \
      _PEXEC_LOG=/dev/stdout; \
    fi; \
    trap '_PEXEC_TRAP_EXIT' EXIT; trap 'exit 0' TERM INT; \
    if [[ \$(${FILE} -bL --mime \"\${0%% *}\") =~ ^text/ ]]; then \
      source \"${PEXEC_SCRARG}\"; \
      if [ \"\${PEXEC_PID}\" ]; then \
        for PID in \"\${PEXEC_PID[@]}\"; do wait \"\${PID}\"; done; \
      fi; \
    else \
      eval \"${PEXEC_SCRARG}\"; \
    fi >\"\${_PEXEC_LOG}\" 2>&1"
  if [ "0" = "${QUIET}" ]; then
    if [ "$(command -v tr)" ]; then
      if [ "${TARGET}" ]; then AT=$(echo "@${TARGET}" | tr "[:lower:]" "[:upper:]"); fi
      STEPNAME=${STEPNAME:-${BUILDKITE_LABEL}}
      if [ "${STEPNAME}" ]; then
        LABEL="$(echo "${STEPNAME}" | tr -s "[:punct:][:space:]" - \
        | ${SED} 's/^-//;s/-$//' | tr "[:lower:]" "[:upper:]")${AT} "
      else
        LABEL="$(basename "$(pwd -P)" | tr "[:lower:]" "[:upper:]")${AT} "
      fi
    fi
    1>&2 echo "Execute ${LABEL}with NTASKS=${COUNTER}, NPROCS=${NP}x${NJ}, and OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  fi
  if [ "${BUILD}" ]; then ${CAT} /dev/null >"${BUILD}"; fi # truncate file
  if [ "${NSECS}" ]; then
    NSECS=$(($(date +%s)-NSECS))
  fi
  TIME=$(which time)  # !command
  if [ "0" != "${PEXEC_IL}" ]; then
    if [ "${TIME}" ] && [ -e "${TIME}" ]; then
      echo -e "${COUNTED}" | ${TIME} -p -o "${NAME}.txt" "${XARGS}" >"${LOG_OUTER}" -P${NP} -I{} bash -c "${PEXEC_INLINE}" "{}"
      RESULT=$?
    else
      echo -e "${COUNTED}" | ${XARGS} >"${LOG_OUTER}" -P${NP} -I{} bash -c "${PEXEC_INLINE}" "{}"
      RESULT=$?
    fi
  else
    echo "#!/usr/bin/env bash" >"${PEXEC_SCRIPT}"
    echo "${PEXEC_INLINE}"    >>"${PEXEC_SCRIPT}"
    chmod +x "${PEXEC_SCRIPT}"
    if [ "${TIME}" ] && [ -e "${TIME}" ]; then
      echo -e "${COUNTED}" | ${TIME} -p -o "${NAME}.txt" "${XARGS}" >"${LOG_OUTER}" -P${NP} -I{} "${PEXEC_SCRIPT}" "{}"
      RESULT=$?
    else
      echo -e "${COUNTED}" | ${XARGS} >"${LOG_OUTER}" -P${NP} -I{} "${PEXEC_SCRIPT}" "{}"
      RESULT=$?
    fi
  fi
  if [ -e "${NAME}.txt" ] && [ "$(command -v bc)" ]; then
    read -r -d $'\04' TREAL TUSER TSYST <<<"$(${CUT} -d' ' -f2 "${NAME}.txt")"
    echo "--------------------------------------------------------------------------------"
    SPEEDUP=$(bc 2>/dev/null -l <<<"(${NSECS}+${TUSER}+${TSYST})/(${NSECS}+${TREAL})")
    EFFINCY=$(bc 2>/dev/null -l <<<"100*${SPEEDUP}/(${NP}*${NJ})")
    printf "Executed ${COUNTER} tasks with %.0f%% parallel efficiency (speedup=%.1fx init=%is)\n" \
      "${EFFINCY}" "${SPEEDUP}" "${NSECS}"
  fi
  if [ "0" != "${RESULT}" ]; then
    if [ ! "${SPEEDUP}" ] && [ ! "${EFFINCY}" ]; then
      echo "--------------------------------------------------------------------------------"
    fi
    if [ "${BUILD}" ] && [ -f "${BUILD}" ] && [ "$(command -v sort)" ]; then
      sort -u "${BUILD}" -o "${BUILD}"
    fi
    exit ${RESULT}
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
