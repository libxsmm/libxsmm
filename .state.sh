#!/usr/bin/env sh
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

MKDIR=$(command -v mkdir)
DIFF=$(command -v diff)
SED=$(command -v sed)
TR=$(command -v tr)

if [ "" != "${MKDIR}" ] && [ "" != "${DIFF}" ] && [ "" != "${SED}" ] && [ "" != "${TR}" ]; then
  HERE=$(cd "$(dirname "$0")"; pwd -P)
  if [ "" = "$1" ]; then
    STATEFILE=${HERE}/.state
  else
    ${MKDIR} -p "$1"
    STATEFILE=$1/.state
    shift
  fi

  STATE=$(${TR} '?' '\n' | ${TR} '"' \' | ${SED} -e 's/^ */\"/' -e 's/   */ /g' -e 's/ *$/\\n\"/')
  TOUCH=$(command -v touch)
  if [ ! -e "${STATEFILE}" ]; then
    if [ "" = "${NOSTATE}" ] || [ "0" = "${NOSTATE}" ]; then
      printf "%s\n" "${STATE}" > "${STATEFILE}"
    fi
    echo "$0"
    # only needed to execute body of .state-rule
    if [ "" != "${TOUCH}" ]; then ${TOUCH} $0; fi
  else # difference must be determined
    if [ "$@" ]; then
      EXCLUDE="-e /\($(echo "$@" | ${SED} "s/[[:space:]][[:space:]]*/\\\|/g" | ${SED} "s/\\\|$//")\)/d"
    fi
    STATE_DIFF=$(echo "${STATE}" \
               | ${DIFF} --new-line-format="" --unchanged-line-format="" "${STATEFILE}" - 2>/dev/null \
               | ${SED} -e 's/=..*$//' -e 's/\"//g' -e '/^$/d' ${EXCLUDE})
    if [ "0" != "$?" ] || [ "" != "${STATE_DIFF}" ]; then
      if [ "" = "${NOSTATE}" ] || [ "0" = "${NOSTATE}" ]; then
        printf "%s\n" "${STATE}" > "${STATEFILE}"
      fi
      echo "$0 $(echo "${STATE_DIFF}")"
      # only needed to execute body of .state-rule
      if [ "" != "${TOUCH}" ]; then ${TOUCH} $0; fi
    fi
  fi
else
  echo "Error: missing prerequisites!"
  exit 1
fi

