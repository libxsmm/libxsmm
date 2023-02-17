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
# shellcheck disable=SC2012

# check if logfile is given
if [ ! "${LOGFILE}" ]; then
  if [ "$1" ]; then
    LOGFILE=$1
  else
    LOGFILE=/dev/stdin
  fi
fi
if [ ! -e "${LOGFILE}" ]; then
  >&2 echo "ERROR: logfile \"${LOGFILE}\" does not exist!"
  exit 1
fi

# automatically echoing input
if [ ! "${LOGRPT_ECHO}" ]; then
  if [ "/dev/stdin" = "${LOGFILE}" ]; then
    LOGRPT_ECHO=1
  else
    LOGRPT_ECHO=0
  fi
fi

# ensure proper permissions
if [ "${UMASK}" ]; then
  UMASK_CMD="umask ${UMASK};"
  eval "${UMASK_CMD}"
fi

# based on https://stackoverflow.com/a/20401674/3001239
flush() {
  if [ "$(command -v sync)" ]; then sync; fi # e.g., async NFS
  if [ "$(command -v script)" ]; then
    script -qefc "$(printf "%q " "$@")" /dev/null
  else
    eval "$@"
  fi
}

# optionally enable script debug
if [ "${LOGRPT_DEBUG}" ] && [ "0" != "${LOGRPT_DEBUG}" ]; then
  echo "*** DEBUG ***"
  if [[ ${LOGRPT_DEBUG} =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
    set -xv
  else
    set "${LOGRPT_DEBUG}"
  fi
  PYTHON=$(command -v python3)
  if [ ! "${PYTHON}" ]; then
    PYTHON=$(command -v python)
  fi
  if [ "${PYTHON}" ]; then
    ${PYTHON} -m site --user-site 2>&1 && echo
  fi
  env
  echo "*** DEBUG ***"
fi

# determine artifact directory
if [ "${LOGRPTDIR}" ] && [ -d "${LOGRPTDIR}" ]; then
  LOGDIR=${LOGRPTDIR}
else
  if [ "${HOME}" ] && [ -d "${HOME}/artifacts" ]; then
    LOGDIR=${HOME}/artifacts
  elif [ "${HOME_REMOTE}" ] && [ -d "${HOME_REMOTE}/artifacts" ]; then
    LOGDIR=${HOME_REMOTE}/artifacts
  elif [ "$(command -v cut)" ] && [ "$(command -v getent)" ]; then
    ARTUSER=$(ls -g "${LOGFILE}" | cut 2>/dev/null -d' ' -f3) # group
    ARTROOT=$(getent passwd "${ARTUSER}" 2>/dev/null | cut -d: -f6 2>/dev/null)
    if [ ! "${ARTROOT}" ]; then ARTROOT=$(dirname "${HOME}")/${ARTUSER}; fi
    if [ -d "${ARTROOT}/artifacts" ]; then
      LOGDIR=${ARTROOT}/artifacts
    elif [ "/dev/stdin" != "${LOGFILE}" ]; then
      LOGDIR=$(cd "$(dirname "${LOGFILE}")" && pwd -P)
    else # debug purpose
      LOGDIR=.
    fi
  fi
fi

# prerequisites for report and opting-out from artifacts
HERE=$(cd "$(dirname "$0")" && pwd -P)
if [ "${LOGDIR}" ] && [ "0" != "${LOGRPT}" ] && \
   [ -e "${HERE}/tool_logperf.sh" ];
then
  PIPELINE=${PIPELINE:-${BUILDKITE_PIPELINE_SLUG}}
  JOBID=${JOBID:-${BUILDKITE_BUILD_NUMBER}}
  STEPNAME=${STEPNAME:-${BUILDKITE_LABEL}}
  if [ ! "${PIPELINE}" ] && \
     [ "$(pwd -P)" = "$(cd "$(dirname "${LOGDIR}")" && pwd -P)" ];
  then
    PIPELINE="debug"
  fi
  if [ "${PIPELINE}" ]; then
    if [ -e "${LOGDIR}/tool_report.sh" ]; then
      DBSCRT=${LOGDIR}/tool_report.sh
    elif [ -e "${HERE}/tool_report.sh" ]; then
      DBSCRT=${HERE}/tool_report.sh
    fi
  fi
  if [ ! "${DBSCRT}" ]; then
    LOGDIR=""
  fi
fi

# determine non-default weights-file (optional)
if [ "${LOGDIR}" ] && [ "${PPID}" ] && \
   [ "$(command -v tail)" ] && \
   [ "$(command -v sed)" ] && \
   [ "$(command -v ps)" ];
then
  PARENT_PID=${PPID}
  while [ "${PARENT_PID}" ]; do
    PARENT=$(ps -o args= ${PARENT_PID} \
      | sed -n "s/[^[:space:]][^[:space:]]*[[:space:]][[:space:]]*\([^.][^.]*\)[.[:space:]]*.*/\1/p")
    if [ "${PARENT}" ]; then
      PARENT_PID=$(ps -oppid ${PARENT_PID} | tail -n1)
      if [ -e "${PARENT}.weights.json" ]; then
        WEIGHTS=${PARENT}.weights.json
      else
        PARENT_DIR=$(dirname "${PARENT}")
        if [ -e "${PARENT_DIR}/../weights.json" ]; then
          WEIGHTS=${PARENT_DIR}/../weights.json
        fi
      fi
      if [ "${WEIGHTS}" ]; then
        DBSCRT="${DBSCRT} -w ${WEIGHTS}"
        break;
      fi
    else
      PARENT_PID=""
    fi
  done
fi

# post-process logfile and generate report
if [ "${LOGDIR}" ]; then
  if [ ! "${LOGRPTSUM}" ] || \
     [[ ${LOGRPTSUM} =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]];
  then # "telegram" format
    if ! FINPUT=$(flush "${HERE}/tool_logperf.sh" "${LOGFILE}");
    then FINPUT=""; fi
    SUMMARY=${LOGRPTSUM:-1}
    QUERY=${STEPNAME}
    RESULT="ms"
  else # JSON-format
    if ! FINPUT=$(flush "${HERE}/tool_logperf.sh" -j "${LOGFILE}");
    then FINPUT=""; fi
    RESULT=${LOGRPTSUM}
    SUMMARY=0
    QUERY=""
  fi
  if [ "${FINPUT}" ]; then
    if [ "${LOGRPT_ECHO}" ] && [ "0" != "${LOGRPT_ECHO}" ] && \
       [ "$(command -v tail)" ];
    then
      VERBOSITY=-1
    else
      VERBOSITY=1
    fi
    mkdir -p "${LOGDIR}/${PIPELINE}/${JOBID}"
    if ! OUTPUT=$(echo "${FINPUT}" | ${DBSCRT} \
      -f "${LOGDIR}/${PIPELINE}.json" \
      -g "${LOGDIR}/${PIPELINE}/${JOBID}" \
      -i /dev/stdin -j "${JOBID}" \
      -x -y "${QUERY}" -r "${RESULT}" \
      -z -v ${VERBOSITY});
    then
      OUTPUT=""
    fi
  fi
  if [ "${OUTPUT}" ] && \
     [ "$(command -v base64)" ] && \
     [ "$(command -v cut)" ];
  then
    if [ "0" != "$((0>VERBOSITY))" ]; then
      FIGURE=$(echo "${OUTPUT}" | tail -n1 | cut -d' ' -f1)
    else
      FIGURE=$(echo "${OUTPUT}" | cut -d' ' -f1)
    fi
    if [ "${FIGURE}" ] && [ -e "${FIGURE}" ]; then
      if ! FIGURE=$(base64 -w0 "${FIGURE}");
      then FIGURE=""; fi
      if [ "${FIGURE}" ]; then
        if [ "0" != "${SUMMARY}" ]; then echo "${FINPUT}"; fi
        printf "\n\033]1338;url=\"data:image/png;base64,%s\";alt=\"%s\"\a\n" \
          "${FIGURE}" "${STEPNAME:-${RESULT}}"
      fi
    fi
  fi
fi
