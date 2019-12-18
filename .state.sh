#!/bin/sh
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

TOUCH=$(command -v touch)
DIFF=$(command -v diff)
SED=$(command -v sed)
TR=$(command -v tr)

if [ "" != "${TOUCH}" ] && [ "" != "${DIFF}" ] && [ "" != "${SED}" ] && [ "" != "${TR}" ]; then
  if [ "$1" = "" ]; then
    STATEFILE=./.state
  else
    STATEFILE=$1/.state
  fi

  STATE=$(${TR} '?' '\n' | ${TR} '"' \' | ${SED} -e 's/^ */\"/' -e 's/   */ /g' -e 's/ *$/\\n\"/')
  if [ ! -e ${STATEFILE} ]; then
    if [ "" = "${NOSTATE}" ] || [ "0" = "${NOSTATE}" ]; then
      printf "%s\n" "${STATE}" > ${STATEFILE}
    fi
    echo "$0"
    # only needed to execute body of .state-rule
    ${TOUCH} $0
  else # difference must be determined
    STATE_DIFF=$(echo "${STATE}" | ${DIFF} --new-line-format="" --unchanged-line-format="" ${STATEFILE} - 2>/dev/null)
    if [ "0" != "$?" ] || [ "" != "${STATE_DIFF}" ]; then
      if [ "" = "${NOSTATE}" ] || [ "0" = "${NOSTATE}" ]; then
        printf "%s\n" "${STATE}" > ${STATEFILE}
      fi
      echo "$0 $(echo "${STATE_DIFF}" | ${SED} -e 's/=..*$//' -e 's/\"//g' -e '/^$/d')"
      # only needed to execute body of .state-rule
      ${TOUCH} $0
    fi
  fi
else
  echo "Error: missing prerequisites!"
  exit 1
fi

