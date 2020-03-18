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

DIFF=$(command -v diff)
SED=$(command -v gsed)

# GNU sed is desired (macOS)
if [ "" = "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "" != "${DIFF}" ] && \
   [ "" != "${SED}" ];
then
  ENVFILE=$1
  shift
  if [ -e "${ENVFILE}" ]; then
    # no need to have unique values in ENVDIFF aka "sort -u"
    ENVDIFF=$(declare -px | ${DIFF} ${ENVFILE} - | ${SED} -n 's/[<>] \(..*\)/\1/p' | ${SED} -n 's/declare -x \(..*\)=..*/\1/p')
    for ENV in ${ENVDIFF}; do # restore environment
      ENVVAR=$(${SED} -n "/declare \-x ${ENV}=/p" ${ENVFILE})
      if [ "" != "${ENVVAR}" ]; then
        eval ${ENVVAR}
      else
        unset ${ENV}
      fi
    done
  else
    echo "Error: missing name of backup-file generated with \"declare -px\"!"
    exit 1
  fi
else
  echo "Error: missing prerequisites!"
  exit 1
fi

