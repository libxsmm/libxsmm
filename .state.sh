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
UNIQ=$(command -v uniq)
SED=$(command -v sed)
TR=$(command -v tr)

if [ "" != "${MKDIR}" ] && [ "" != "${SED}" ] && [ "" != "${TR}" ] && \
   [ "" != "${DIFF}" ] && [ "" != "${UNIQ}" ];
then
  HERE=$(cd "$(dirname "$0")"; pwd -P)
  if [ "" != "$1" ]; then
    STATEFILE=$1/.state
    ${MKDIR} -p "$1"
    shift
  else
    STATEFILE=.state
  fi

  STATE=$(${TR} '?' '\n' | ${TR} '"' \' | ${SED} -e 's/^ */\"/' -e 's/   */ /g' -e 's/ *$/\\n\"/')
  TOUCH=$(command -v touch)
  if [ -e "${STATEFILE}" ]; then
    if [ "$@" ]; then
      EXCLUDE="-e /\($(echo "$@" | ${SED} "s/[[:space:]][[:space:]]*/\\\|/g" | ${SED} "s/\\\|$//")\)/d"
    fi
    # BSD's diff does not support --unchanged-line-format=""
    STATE_DIFF=$(printf "%s\n" "${STATE}" \
               | ${DIFF} "${STATEFILE}" - 2>/dev/null | ${SED} -n 's/[<>] \(..*\)/\1/p' \
               | ${SED} -e 's/=..*$//' -e 's/\"//g' -e '/^$/d' ${EXCLUDE} | ${UNIQ})
    if [ "0" != "$?" ] || [ "" != "${STATE_DIFF}" ]; then
      if [ "" = "${NOSTATE}" ] || [ "0" = "${NOSTATE}" ]; then
        printf "%s\n" "${STATE}" > "${STATEFILE}"
      fi
      echo "$0 $(echo "${STATE_DIFF}")"
      # only needed to execute body of .state-rule
      if [ "" != "${TOUCH}" ]; then ${TOUCH} "$0"; fi
    fi
  else # difference must not be determined
    if [ "" = "${NOSTATE}" ] || [ "0" = "${NOSTATE}" ]; then
      printf "%s\n" "${STATE}" > "${STATEFILE}"
    fi
    echo "$0"
    # only needed to execute body of .state-rule
    if [ "" != "${TOUCH}" ]; then ${TOUCH} "$0"; fi
  fi
else
  echo "Error: missing prerequisites!"
  exit 1
fi

