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

CURL=$(command -v curl)
DOMAIN=com
TOKEN=$1
JOBID=$2

if [ ! "${CURL}" ]; then
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi

if [ ! "${JOBID}" ] || [ ! "${TOKEN}" ]; then
  echo "Usage: $0 <token> <jobid>"
  echo "       token: API authentication; see https://travis-ci.${DOMAIN}/account/preferences"
  echo "       jobid: not the build-id; expand \"Build system information\" (job's log)"
  echo "See also https://docs.travis-ci.com/user/running-build-in-debug-mode/"
  exit 1
fi

${CURL} -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Travis-API-Version: 3" \
  -H "Authorization: token ${TOKEN}" \
  -d "{\"quiet\": true}" \
  https://api.travis-ci.${DOMAIN}/job/${JOBID}/debug

RESULT=$?
if [ "0" != "${RESULT}" ]; then
  echo "Error (code ${RESULT}): failed to enable debug mode!"
  exit ${RESULT}
fi

