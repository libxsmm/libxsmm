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
  STRICT=0
  if [ "-s" = "$1" ] || [ "--strict" = "$1" ]; then
    STRICT=1
    shift
  fi
  ENVFILE=$1
  if [ -e "${ENVFILE}" ]; then
    shift
    ENVSRCF=$1
    if [ "${ENVSRCF}" ]; then
      echo "#!/usr/bin/env bash" >"${ENVSRCF}"
      shift
    fi
    ENVDIFF="declare -px | ${DIFF} --old-line-format='' - ${ENVFILE} \
           | ${SED} -n 's/declare -x \(.[^=]*\)=..*/\1/p'"
    for ENV in $(eval "${ENVDIFF}"); do # restore environment
      DEF=$(${SED} -n "/declare \-x ${ENV}=/p" "${ENVFILE}")
      if [ "$(${SED} -n "/\".*[^\]\"/p" <<<"${DEF}")" ]; then
        VAL=$(${SED} "s/declare -x ${ENV}=\(..*\)/\1/" <<<"${DEF}")
        if [ "$(${SED} -n "/^\"\//p" <<<"${VAL}")" ]; then
          VALS="" && IFS=':"' && for DIR in ${VAL}; do
            if [ "${DIR}" ] && [ -d "$(dirname "${DIR}")" ]; then
              if [ "${VALS}" ]; then VALS="${VALS}:${DIR}";
              else VALS="${DIR}"; fi
            fi
          done && unset IFS
          if [ "${VALS}" ]; then VAL="\"${VALS}\""; else VAL=""; fi
        fi
        if [ "${STRICT}" ] && [ "0" != "${STRICT}" ] && \
           [ "$(${SED} -n "/\//p" <<<"${VAL}")" ] && \
           [ "$(declare -px | ${SED} -n "/${ENV}/p")" ];
        then
          VAL=""
        fi
        if [ "${VAL}" ]; then
          if [ "$(${SED} -n "/PATH$/p" <<<"${ENV}")" ]; then
            DEF="declare -x ${ENV}=$(${SED} -e "s/^\":*/\"\${${ENV}}:/" -e "s/:*\"$/\"/" <<<"${VAL}")"
          else
            DEF="declare -x ${ENV}=${VAL}"
          fi
          if [ "${ENVSRCF}" ]; then
            echo "${DEF}" >>"${ENVSRCF}"
          else
            eval "${DEF}"
          fi
        fi
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
