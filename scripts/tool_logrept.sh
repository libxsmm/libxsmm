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
# shellcheck disable=SC2012,SC2153,SC2206

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
  if [ ! "${PIPELINE}" ]; then
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
    if PSOUT=$(ps -o args= ${PARENT_PID} 2>/dev/null); then
      PARENT=$(echo "${PSOUT}" \
        | sed -n "s/[^[:space:]][^[:space:]]*[[:space:]][[:space:]]*\([^.][^.]*\)[.[:space:]]*.*/\1/p")
      if [ "${PARENT}" ]; then
        PARENT_PID=$(ps -oppid ${PARENT_PID} | tail -n1)
        if [ -e "${PARENT}.weights.json" ]; then
          WEIGHTS=${PARENT}.weights.json
        else
          PARENT_DIR=$(dirname "${PARENT}" 2>/dev/null)
          if [ "${PARENT_DIR}" ] && [ -e "${PARENT_DIR}/../weights.json" ]; then
            WEIGHTS=${PARENT_DIR}/../weights.json
          fi
        fi
        if [ "${WEIGHTS}" ]; then # break
          DBSCRT="${DBSCRT} -w ${WEIGHTS}"
          PARENT_PID=""
        fi
      else # break
        PARENT_PID=""
      fi
    else # break
      PARENT_PID=""
    fi
  done
fi

# process logfile and generate report
if [ "${LOGDIR}" ]; then
  SYNC=$(command -v sync)
  ${SYNC} # optional
  ERROR=""

  # extract data from log (tool_logperf.sh)
  if [ ! "${LOGRPTSUM}" ] || \
     [[ ${LOGRPTSUM} =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]];
  then # "telegram" format
    if ! FINPUT=$("${HERE}/tool_logperf.sh" ${LOGFILE});
    then FINPUT=""; fi
    SUMMARY=${LOGRPTSUM:-1}
    RESULT="ms"
  else # JSON-format
    if ! FINPUT=$("${HERE}/tool_logperf.sh" -j ${LOGFILE});
    then FINPUT=""; fi
    RESULT=${LOGRPTSUM}
    SUMMARY=0
  fi

  # capture result in database and generate report
  if [ "${FINPUT}" ] && [ "$(command -v sed)" ]; then
    QUERY=${LOGRPTQRY-${STEPNAME}}
    if [ "${LOGRPTQRX}" ] && [ "0" != "${LOGRPTQRX}" ]; then
      EXACT="-e"
    fi
    if [ "${LOGRPTSEP}" ] && [ "0" != "${LOGRPTSEP}" ]; then
      UNTIED="-u"
    fi
    if [ "${LOGRPT_ECHO}" ] && [ "0" != "${LOGRPT_ECHO}" ]; then
      VERBOSITY=-1
    else
      VERBOSITY=1
    fi
    mkdir -p "${LOGDIR}/${PIPELINE}/${JOBID}"
    if ! OUTPUT=$(echo "${FINPUT}" | ${DBSCRT} \
      -p "${PIPELINE}" -b "${LOGRPTBRN}" \
      -f "${LOGDIR}/${PIPELINE}.json" \
      -g "${LOGDIR}/${PIPELINE}/${JOBID} ${LOGRPTFMT}" \
      -i /dev/stdin -j "${JOBID}" ${EXACT} \
      -x -y "${QUERY}" -r "${RESULT}" -z \
      -q "${LOGRPTQOP}" ${UNTIED} \
      -t "${LOGRPTBND}" \
      -v ${VERBOSITY});
    then  # ERROR=$?
      ERROR=1
    fi
    FIGPAT="[[:space:]][[:space:]]*created\."
    FIGURE=$(echo "${OUTPUT}" | sed -n "/${FIGPAT}/p" | sed '$!d')
    if [ "${FIGURE}" ]; then
      OUTPUT=$(echo "${OUTPUT}" | sed "/${FIGPAT}/d")
    fi
    if [ "${OUTPUT}" ] && [[ ("${ERROR}") || ("0" != "$((0>VERBOSITY))") ]]; then
      echo "${OUTPUT}"
    fi
    OUTPUT=${FIGURE}
  fi

  # embed report into log (base64)
  if [ "${OUTPUT}" ]; then
    if [ "$(command -v base64)" ] && \
       [ "$(command -v cut)" ];
    then
      FIGURE=$(echo "${OUTPUT}" | cut -d' ' -f1)  # filename
      if [ "${FIGURE}" ] && [ -e "${FIGURE}" ]; then
        if ! OUTPUT=$(base64 -w0 "${FIGURE}");
        then OUTPUT=""; fi
        if [ "${OUTPUT}" ]; then
          FORMAT=(${LOGRPTFMT:-${FIGURE##*.}})
          REPORT=${FIGURE%."${FORMAT[0]}"}.pdf
          if [ "$(command -v mimetype)" ]; then
            MIMETYPE=$(mimetype -b "${FIGURE}")
          else
            if [ "svgz" = "${FORMAT[0]}" ]; then
              MIMETYPE="image/svg+xml-compressed"
            elif [ "svg" = "${FORMAT[0]}" ]; then
              MIMETYPE="image/svg+xml"
            else  # fallback
              MIMETYPE="image/${FORMAT[0]}"
            fi
          fi
          if [ "0" != "${SUMMARY}" ]; then echo "${FINPUT}"; fi
          if [ -e "${REPORT}" ]; then  # print after summary
            printf "\n\033]1339;url=\"artifact://%s\";content=\"Report\"\a\n" "${REPORT}"
          fi
          printf "\n\033]1338;url=\"data:%s;base64,%s\";alt=\"%s\"\a\n" \
            "${MIMETYPE}" "${OUTPUT}" "${STEPNAME:-${RESULT}}"
          if [ "${ERROR}" ] && [ "0" != "${ERROR}" ]; then
            >&2 echo "WARNING: deviation of latest value exceeds margin."
            exit "${ERROR}"
          fi
        else
          >&2 echo "WARNING: encoding failed (\"${FIGURE}\")."
        fi
      else
        >&2 echo "WARNING: report not ready (\"${OUTPUT}\")."
      fi
    else
      >&2 echo "WARNING: missing prerequisites for report."
    fi
  fi
fi
