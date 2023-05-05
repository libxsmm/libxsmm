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

DIFF=$(command -v diff)
SED=$(command -v gsed)

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${DIFF}" ] && [ "${SED}" ]; then
  ENVFILE=$1
  if [ -e "${ENVFILE}" ]; then
    shift
    ENVSRCF=$1
    if [ "${ENVSRCF}" ]; then
      if [ ! "${UNIQ}" ] && [ "$(command -v sort)" ]; then UNIQ="| sort -u"; fi
      if [ ! "${UNIQ}" ] && [ "$(command -v uniq)" ]; then UNIQ="| uniq"; fi
      echo "#!/usr/bin/env bash" >"${ENVSRCF}"
      shift
    fi
    # no need to have unique values in ENVDIFF in general
    ENVDIFF="declare -px | ${DIFF} ${ENVFILE} - | ${SED} -n 's/[<>] \(..*\)/\1/p' | ${SED} -n 's/declare -x \(.[^=]*\)=..*/\1/p' ${UNIQ}"
    for ENV in $(eval "${ENVDIFF}"); do # restore environment
      DEF=$(${SED} -n "/declare \-x ${ENV}=/p" "${ENVFILE}")
      if [ "$(echo "${DEF}" | ${SED} -n "/\".*[^\]\"/p")" ]; then
        if [ "${ENVSRCF}" ]; then
          VAL=$(echo "${DEF}" | ${SED} "s/declare -x ${ENV}=\(..*\)/\1/")
          if [ "$(echo "${ENV}" | ${SED} -n "/PATH$/p")" ]; then
            echo "declare -x ${ENV}=$(echo "${VAL}" | ${SED} -e "s/^\":*/\"\${${ENV}}:/" -e "s/:*\"$/\"/")" >>"${ENVSRCF}"
          else
            echo "declare -x ${ENV}=${VAL}" >>"${ENVSRCF}"
          fi
        fi
        eval "${DEF}"
      else
        unset "${ENV}"
      fi
    done
  else
    >&2 echo "ERROR: missing name of backup-file generated with \"declare -px\"!"
    exit 1
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
