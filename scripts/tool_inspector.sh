#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

RPT=inspector
KIND=mi1

BASENAME=$(command -v basename)
TOOL=$(command -v inspxe-cl)
GREP=$(command -v grep)
SED=$(command -v sed)
TR=$(command -v tr)
RM=$(command -v rm)

if [ "${TOOL_ENABLED}" != "" ] && [ "${TOOL_ENABLED}" != "0" ]; then
  if [ "" != "$1" ]    && [ "" != "${BASENAME}" ] && [ "" != "${TOOL}" ] && \
     [ "" != "${TR}" ] && [ "" != "${GREP}" ]     && [ "" != "${SED}" ]  && \
     [ "" != "${RM}" ];
  then
    HERE=$(cd "$(dirname "$0")"; pwd -P)
    if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
      export TRAVIS_BUILD_DIR=${HERE}/..
    fi
    if [ "" != "${TESTID}" ]; then
      ID=${TESTID}
    fi
    if [ "" = "${ID}" ]; then
      ID=${COVID}
    fi
    if [ "" != "${ID}" ]; then
      RPTNAME=$(${BASENAME} $1)-${KIND}-${ID}
    else
      RPTNAME=$(${BASENAME} $1)-${KIND}
    fi

    DIR=${TRAVIS_BUILD_DIR}/${RPT}
    ${RM} -rf ${DIR}/${ID}

    ${TOOL} -collect ${KIND} -r ${DIR}/${ID} -no-auto-finalize -return-app-exitcode -- "$@"
    RESULT=$?

    if [ "0" = "${RESULT}" ]; then
      ${TOOL} -report problems -r ${DIR}/${ID} > ${DIR}/${RPTNAME}.txt
      RESULT2=$?

      if [ "" = "${TOOL_REPORT_ONLY}" ] && [ "0" != "$((2<RESULT2))" ]; then
        FN=$(${GREP} 'Function' ${DIR}/${RPTNAME}.txt | \
             ${SED} -e 's/..* Function \(..*\):..*/\1/')
        XFLT=$(echo "${TOOL_XFILTER}" | ${TR} -s " " | ${TR} " " "|")
        YFLT=$(echo "${TOOL_FILTER}" | ${TR} -s " " | ${TR} " " "|")
        MATCH=${FN}

        if [ "" != "${XFLT}" ]; then MATCH=$(echo "${MATCH}" | ${GREP} -Ev ${XFLT}); fi
        if [ "" = "${YFLT}" ]  || [ "" != "$(echo "${MATCH}" | ${GREP} -E  ${YFLT})" ]; then
          RESULT=${RESULT2}
        fi
      fi
    fi
    exit ${RESULT}
  else
    "$@"
  fi
else
  "$@"
fi

