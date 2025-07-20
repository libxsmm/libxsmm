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
# shellcheck disable=SC2012,SC2086,SC2153,SC2206

if [ "$1" ]; then  # argument takes precedence
  LOGFILE=$1
elif [ ! "${LOGFILE}" ]; then  # logfile given?
  if [ "$1" ]; then
    LOGFILE=$1
  else
    LOGFILE=/dev/stdin
  fi
fi

if [ ! -e "${LOGFILE}" ]; then
  # keep output in sync, i.e., avoid ">&2 echo"
  echo -e "ERROR: logfile \"${LOGFILE}\" does not exist!\n"
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
  PYTHON=$(command -v python3 || true)
  if [ ! "${PYTHON}" ]; then
    PYTHON=$(command -v python || true)
  fi
  if [ "${PYTHON}" ]; then
    ${PYTHON} -m site --user-site 2>&1 && echo
  fi
  env
  echo "*** DEBUG ***"
  if [[ ${LOGRPT_DEBUG} =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
    set -xv
  else
    set "${LOGRPT_DEBUG}"
  fi
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
    ARTUSER=$(ls -g "${LOGFILE}" | cut 2>/dev/null -d' ' -f3)  # group
    ARTROOT=$(getent passwd "${ARTUSER}" 2>/dev/null | cut -d: -f6 2>/dev/null)
    if [ ! "${ARTROOT}" ]; then ARTROOT=$(dirname "${HOME}")/${ARTUSER}; fi
    if [ -d "${ARTROOT}/artifacts" ]; then
      LOGDIR=${ARTROOT}/artifacts
    elif [ "/dev/stdin" != "${LOGFILE}" ]; then
      LOGDIR=$(cd "$(dirname "${LOGFILE}")" && pwd -P)
    fi
  fi
  if [ ! "${LOGDIR}" ]; then  # debug purpose
    LOGDIR=.
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
    elif [ -e "${HERE}/tool_report.py" ]; then
      DBSCRT=${HERE}/tool_report.py
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
      PARENT=$(sed -n \
          "s/[^[:space:]][^[:space:]]*[[:space:]][[:space:]]*\([^.][^.]*\)[.[:space:]]*.*/\1/p" \
        <<<"${PSOUT}")
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
        if [ "${WEIGHTS}" ]; then  # break
          DBSCRT="${DBSCRT} -w ${WEIGHTS}"
          PARENT_PID=""
        fi
      else  # break
        PARENT_PID=""
      fi
    else  # break
      PARENT_PID=""
    fi
  done
fi

# process logfile and generate report
if [ "${LOGDIR}" ]; then
  SYNC=$(command -v sync)
  ${SYNC}  # optional
  ERROR=""

  # extract data from log (tool_logperf.sh)
  if [ ! "${LOGRPTSUM}" ] || \
     [[ ${LOGRPTSUM} =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]];
  then  # "telegram" format
    if ! FINPUT=$("${HERE}/tool_logperf.sh" ${LOGFILE});
    then FINPUT=""; fi
    SUMMARY=${LOGRPTSUM:-1}
    RESULT="ms"
  fi
  if [ ! "${FINPUT}" ]; then  # JSON-format
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
      UNTIED="-u ${LOGRPTSEP}"
    fi
    if [ "${LOGRPT_ECHO}" ] && [ "0" != "${LOGRPT_ECHO}" ]; then
      VERBOSITY=-1
    else
      VERBOSITY=1
    fi
    ARTDIR=${LOGDIR}/${PIPELINE}/${JOBID}
    mkdir -p "${ARTDIR}"
    if ! OUTPUT=$(${DBSCRT} -v "${VERBOSITY}" \
      -p "${PIPELINE}" -b "${LOGRPTBRN}" -t "${LOGRPTBND}" \
      -f "${LOGDIR}/${PIPELINE}.json" -g "${ARTDIR} ${LOGRPTFMT}" \
      -i /dev/stdin -j "${JOBID}" -q "${LOGRPTQOP}" \
      -x -y "${QUERY}" -r "${RESULT}" -z \
      ${EXACT} ${UNTIED} \
      <<<"${FINPUT}");
    then  # ERROR=$?
      ERROR=1
    fi
    FIGPAT="[[:space:]][[:space:]]*created\."
    FIGURE=$(sed -n "/${FIGPAT}/p" <<<"${OUTPUT}" | sed '$!d')
    if [ "${FIGURE}" ]; then
      OUTPUT=$(sed "/${FIGPAT}/d" <<<"${OUTPUT}")
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
      FIGURE=$(cut -d' ' -f1 <<<"${OUTPUT}")  # filename
      if [ "${FIGURE}" ] && [ -e "${FIGURE}" ]; then
        RPTFMT=${LOGRPTDOC:-pdf}
        FORMAT=(${LOGRPTFMT:-${FIGURE##*.}})
        REPORT=${FIGURE%."${FORMAT[0]}"}.${RPTFMT}
        # echo parsed/captured JSON
        if [ "0" != "${SUMMARY}" ]; then echo "${FINPUT}"; fi
        if [ -e "${REPORT}" ]; then  # print after summary
          # normalize path to report file (buildkite)
          REPDIR="$(cd "$(dirname "${REPORT}")" && pwd -P)"
          REPFLE=$(basename "${REPORT}")
          if [ "$(command -v tr)" ]; then
            LABEL=$(tr "[:lower:]" "[:upper:]" <<<"${RPTFMT}")
          else
            LABEL=${RPTFMT^^}
          fi
          if [ -e "${REPDIR}/${REPFLE}" ]; then
            printf "\n\033]1339;url=\"artifact://%s\";content=\"%s\"\a\n\n" \
              "${REPFLE}" "${LABEL}"
          fi
        fi
        # embed figure if report is not exclusive
        if [ -e "${FIGURE}" ] && [ "${FIGURE}" != "${REPORT}" ]; then
          BASE64_FLAG=-w0
          if base64 ${BASE64_FLAG} </dev/null 2>&1 | grep -q invalid; then BASE64_FLAG=""; fi
          if ! OUTPUT=$(eval "base64 ${BASE64_FLAG} <${FIGURE}");
          then OUTPUT=""; fi
          if [ "${OUTPUT}" ]; then
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
            printf "\n\033]1338;url=\"data:%s;base64,%s\";alt=\"%s\"\a\n" \
              "${MIMETYPE}" "${OUTPUT}" "${STEPNAME:-${RESULT}}"
          else
            # keep output in sync, i.e., avoid ">&2 echo"
            echo -e "WARNING: encoding failed (\"${FIGURE}\").\n"
          fi
        fi
        if [ "${ERROR}" ] && [ "0" != "${ERROR}" ]; then
          # keep output in sync, i.e., avoid ">&2 echo"
          echo -e "WARNING: deviation of latest value exceeds margin.\n"
          exit "${ERROR}"
        fi
      else
        # keep output in sync, i.e., avoid ">&2 echo"
        echo -e "WARNING: report not ready (\"${OUTPUT}\").\n"
      fi
    else
      # keep output in sync, i.e., avoid ">&2 echo"
      echo -e "WARNING: missing prerequisites for report.\n"
    fi
  fi
fi
