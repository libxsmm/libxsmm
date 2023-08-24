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
DIRPAT="s/\//\\\\\//g"
STRICT=1
FROM=()
TO=()

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ ! "${DIFF}" ] || [ ! "${SED}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

# ensure proper permissions
if [ "${UMASK}" ]; then
  UMASK_CMD="umask ${UMASK};"
  eval "${UMASK_CMD}"
fi

while test $# -gt 0; do
  case "$1" in
  -h|--help)
    HELP=1
    shift $#;;
  -i|-f|--envfile)
    IFILE=$2
    shift 2;;
  -o|--srcfile)
    OFILE=$2
    shift 2;;
  -u|--unrestricted)
    STRICT=0
    shift 1;;
  -s|--strict)
    STRICT=1
    shift 1;;
  -t|--fromto)
    shift && for ARG in "$@"; do
      if [[ "${ARG}" = *":"* ]]; then
        IFS=':' read -ra PAIR <<<"${ARG}" && unset IFS
        FROM+=("$(${SED} "${DIRPAT}" <<<"${PAIR[0]}")")
        TO+=("$(${SED} "${DIRPAT}" <<<"${PAIR[1]}")")
        shift 1
      fi
    done;;
  *)  # positional arguments and rest
    if [ ! "${IFILE}" ]; then IFILE=$1;
    elif [ ! "${OFILE}" ]; then OFILE=$1; fi
    shift 1;;
  esac
done

if [[ (! "${HELP}") || ("0" = "${HELP}") ]] && [ -e "${IFILE}" ]; then
  if [ "${OFILE}" ]; then echo "#!/usr/bin/env bash" >"${OFILE}"; fi
  # diff --old-line-format='' is not portable
  ENVDIFF="declare -px \
         | ${DIFF} - ${IFILE} 2>/dev/null \
         | ${SED} -n 's/declare -x \(.[^=]*\)=..*/\1/p' \
         | ${SED} -n 's/[>] \(..*\)/\1/p'"
  for ENV in $(eval "${ENVDIFF}"); do # restore environment
    DEF=$(${SED} -n "/declare \-x ${ENV}=/p" "${IFILE}")
    if [ "$(${SED} -n "/\".*[^\]\"/p" <<<"${DEF}")" ]; then
      IS_PATH=$(${SED} -n "/PATH$/p" <<<"${ENV}")
      KEY=$(declare -px | ${SED} -n "/${ENV}=/p")
      VAL=$(${SED} "s/declare -x ${ENV}=\(..*\)/\1/" <<<"${DEF}")
      # perform from-to translation before checking path-existence
      if [ "0" != "${#FROM[@]}" ]; then
        for I in $(seq ${#FROM[@]}); do
          VAL=$(${SED} "s/${FROM[I-1]}/${TO[I-1]}/g" <<<"${VAL}" 2>/dev/null)
        done
      fi
      # handle path strictly, i.e., filter for existing parent
      if [ "${STRICT}" ] && [ "0" != "${STRICT}" ] && \
         [ "$(${SED} -n "/^\"\//p" <<<"${VAL}")" ];
      then
        if [ "${KEY}" ] && [ ! "${IS_PATH}" ]; then
          VAL=""  # drop path values with existing key
        else
          # filter paths and ensure existing parent directory
          VALS="" && IFS=':"' && for DIR in ${VAL}; do
            if [ "${DIR}" ] && [ -d "$(dirname "${DIR}")" ]; then
              if [ "${VALS}" ]; then VALS="${VALS}:${DIR}";
              else VALS="${DIR}"; fi
            fi
          done && unset IFS
          if [ "${VALS}" ]; then VAL="\"${VALS}\""; else VAL=""; fi
        fi
      fi
      if [ "${VAL}" ]; then
        if [ "${KEY}" ] && [ "${IS_PATH}" ]; then  # prepend to existing values
          VALEXT=$(${SED} -e "s/:*\"$/:\${${ENV}}\"/" -e "s/::*\"$/\"/" <<<"${VAL}")
          DEF="declare -x ${ENV}=${VALEXT}"
        else  # introduce or replace values
          DEF="declare -x ${ENV}=${VAL}"
        fi
        if [ "${OFILE}" ]; then  # write source'able script
          echo "${DEF}" >>"${OFILE}"
        else  # evaluate definition directly
          eval "${DEF}"
        fi
      fi
    fi
  done
else
  echo "Usage: $0 [options] IFILE [OFILE]"
  echo "       -i|-f|--envfile file: filename of environment file generated with \"declare -px\""
  echo "       -o|--srcfile file: filename of script to be generated (can be source'd)"
  echo "       -r|--replace: replace environment (as opposed to --strict environment)"
  echo "       -s|--strict: keep existing environment variables with paths"
  echo "                    only keep paths where parent directory exists"
  echo "       -t|--fromto a:b [b:c [...]]: replace \"a\" with \"b\", etc."
  echo
  echo "Examples: source $0 -s my.env || true"
  echo "          $0 -t /data/nfs_home:/Users my.env /dev/stdout"
  echo "          $0 -t /data/nfs_home:/Users -s my.env myenv.sh"
  echo
  if [ "${HELP}" ] && [ "0" != "${HELP}" ]; then
    exit 0
  else
    >&2 echo "ERROR: environment-file generated with \"declare -px\" not specified!"
    exit 1
  fi
fi
