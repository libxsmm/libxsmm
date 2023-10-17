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

PATTERNS="*.c *.cc *.cpp *.cxx *.h *.hpp *.hxx *.f *.F90 *.fh *.py *.sh *.env *.yml *.txt *.slurm"
BANNED_CHARS="\t"

PATPRE="s/^[[:space:]][[:space:]]*#/"
PATSPC="s/[[:space:]][[:space:]]*$/"
PATBAN="s/[${BANNED_CHARS}]/"
PATCMT="s/^[[:space:]][[:space:]]*\/\//"
PATEOL="s/\r$/"

HERE=$(cd "$(dirname "$0")" && pwd -P)
REPO=${HERE}/..
CODEFILE=${REPO}/.codefile
MKTEMP=${REPO}/.mktmp.sh
SRCDIR="src"
# separate multiple patterns with space
FMTDIRS=${2:-"${SRCDIR} tests samples"}
FMTXPAT="/gxm/ /mlpcell/"
# limiter
DIR=$1

FMTBIN=$(command -v clang-format)
SHELLC=$(command -v shellcheck)
FLAKE8=$(command -v flake8)
ICONV=$(command -v iconv)
BLACK=$(command -v black)
MYPY=$(command -v mypy)
DIFF=$(command -v diff)
SED=$(command -v gsed)
GIT=$(command -v git)
CUT=$(command -v cut)
CPP=$(command -v cpp)
TR=$(command -v tr)
CP=$(command -v cp)
RM=$(command -v rm)

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${ICONV}" ]; then
  CAT="${ICONV} -t ASCII"
else
  CAT=$(command -v cat)
fi

# If CPP, then -fpreprocessed shall be available
if [ "${CPP}" ] && [ "$(${CPP} -fpreprocessed /dev/null)" ]; then
  unset CPP
fi

if [ "${FLAKE8}" ] && [ "0" != "$(${FLAKE8} 2>&1 >/dev/null; echo $?)" ]; then
  unset FLAKE8
fi

if [ "${MYPY}" ] && [ "0" != "$(${MYPY} 2>&1 >/dev/null; echo $?)" ]; then
  unset MYPY
fi

if [ "${CAT}" ] && [ -e "${CODEFILE}" ]; then
  PATTERNS="$(${CAT} "${CODEFILE}")"
fi

if [ ! "${FMTBIN}" ] || [ "$(${FMTBIN} --style=file -dump-config 2>&1 >/dev/null)" ]; then
  echo "Warning: missing compatible \"clang-format\" command!"
  FMTBIN=""
fi

if [ "${SED}" ] && [ "${CUT}" ] && [ "${TR}" ] && \
   [ "${GIT}" ] && [ "${CP}" ] && [ "${RM}" ] && \
   [ "${CAT}" ] && [ "${MKTEMP}" ];
then
  WARNINGS=0
  TMPF=$("${MKTEMP}" .libxsmm_XXXXXX.txt)
  trap '${RM} ${TMPF}' EXIT
  # disable glob in Shell
  set -f
  # Search the content of the diffs matching the given file types
  for PATTERN in ${PATTERNS} *Makefile*; do
  for FILE in $(${GIT} ls-files "${PATTERN}"); do
    # FILE must be located in DIR (if given) and FILE must exist
    if [[ "${DIR}" && (${FILE} != "${DIR}/"*) ]] || [ ! -e "${FILE}" ]; then continue; fi
    echo -n "${FILE}"
    #
    # Reformat code (fallback: check for banned characters, etc.).
    #
    REFORMAT=0
    if [[ (${FILE} = *".c"*) || (${FILE} = *".h"*) ]]; then
      if [ "${FMTBIN}" ] && [ -e "${REPO}/.clang-format" ]; then
        if [ ! "${FMTDIRS}" ]; then REFORMAT=1; fi
        for FMTDIR in ${FMTDIRS}; do
          if [[ ${FILE} = "${FMTDIR}/"* ]]; then
            REFORMAT=1; break
          fi
        done
      fi
      EXCLUDE=0
      for XPAT in ${FMTXPAT}; do
        if [[ ${FILE} = *"${XPAT}"* ]]; then
          EXCLUDE=1; break
        fi
      done
      if [ "0" != "${EXCLUDE}" ]; then
        REFORMAT=0
      elif [[ (${FILE} = *".c") || (${FILE} = *".h") ]] && \
           [ "$(${SED} -n "${PATCMT}x/p" "${FILE}")" ];
      then
        echo " : has C++ comments"
        exit 1
      fi
    fi
    # remove or comment the following line to enable reformat (do not set REFORMAT=1)
    REFORMAT=0
    if [ "0" != "${REFORMAT}" ]; then
      if [ "0" = "$(${FMTBIN} --style=file "${FILE}" >"${TMPF}"; echo $?)" ] && \
         [ "1" = "$(${DIFF} "${FILE}" "${TMPF}" >/dev/null; echo $?)" ];
      then
        ${CP} "${TMPF}" "${FILE}"
        echo -n " : reformatted"
      else
        REFORMAT=0
      fi
    elif [[ ${FILE} != *"Makefile"* ]] && \
         [ "$(${SED} -n "${PATBAN}x/p" "${FILE}" 2>/dev/null)" ];
    then
      echo " : has banned characters"
      exit 1
    elif [[ ${FILE} = "${SRCDIR}/"* ]] && \
         [[ (${FILE} = *".c"*) || (${FILE} = *".h"*) ]] && \
         [ "$(${SED} -n "${PATPRE}x/p" "${FILE}" 2>/dev/null)" ];
    then
      echo " : white space leads '#' (malformed preprocessor command)"
      exit 1
    elif [ "$(${SED} -n "s/\([^[:space:]]\)\t/\1 /gp" "${FILE}")" ]; then
      ${SED} -e "s/\([^[:space:]]\)\t/\1 /g" "${FILE}" >"${TMPF}"
      ${CP} "${TMPF}" "${FILE}"
      echo -n " : removed tabs"
      REFORMAT=1
    fi
    #
    # Check for non-UNIX line-endings.
    #
    if [ "$(${SED} -n "${PATEOL}x/p" "${FILE}" 2>/dev/null | ${TR} -d "\n")" ]; then
      echo " : has non-UNIX line endings"
      exit 1
    fi
    #
    # Check and fix for trailing spaces.
    #
    if [ "$(${SED} -n "${PATSPC}x/p" "${FILE}")" ]; then
      ${SED} -e "${PATSPC}/" "${FILE}" >"${TMPF}"
      ${CP} "${TMPF}" "${FILE}"
      echo -n " : removed trailing spaces"
      REFORMAT=1
    fi
    #
    # Check and fix executable flag of file under source control.
    # Black-format Flake8-check, and MyPy-check Python file.
    # Shellcheck any Shell script.
    #
    FLAGS=$(${GIT} ls-files -s "${FILE}" | ${CUT} -d' ' -f1)
    if [ "*.sh" = "${PATTERN}" ] || [ "*.slurm" = "${PATTERN}" ] || [ "*.py" = "${PATTERN}" ]; then
      if [ "*.py" = "${PATTERN}" ]; then
        if [ "${BLACK}" ]; then
          if ! ${BLACK} -l79 --check "${FILE}" 2>/dev/null; then
            if [ "${HERE}" = "$(dirname "${FILE}")" ]; then
              ${BLACK} -l79 "${FILE}" 2>/dev/null
              echo -n " : reformatted"
            else
              echo -n " : reformat using \"black -l 79\""
            fi
            REFORMAT=1
          fi
        fi
        if [ "${FLAKE8}" ]; then
          if [ "0" != "$(${FLAKE8} "${FILE}" 2>&1 >/dev/null; echo $?)" ]; then
            echo -n " : fix issues pointed out by Flake8"
            REFORMAT=1
          fi
        fi
        if [ "${MYPY}" ]; then
          if [ "0" != "$(${MYPY} "${FILE}" 2>&1 >/dev/null; echo $?)" ]; then
            echo -n " : fix issues pointed out by MyPy"
            REFORMAT=1
          fi
        fi
      elif [ "${SHELLC}" ]; then
        if ! ${SHELLC} "${FILE}" >/dev/null; then
          echo -n " : fix issues pointed out by Shellcheck"
          REFORMAT=1
        fi
      fi
      if [ "$(${SED} -n '1!b;/#!/p' "${FILE}")" ] && \
         [ "100755" != "${FLAGS}" ];
      then
        ${GIT} update-index --chmod=+x "${FILE}"
        echo -n " : marked executable"
        REFORMAT=1
      fi
    elif [ "100644" != "${FLAGS}" ] && [ "120000" != "${FLAGS}" ]; then
      ${GIT} update-index --chmod=-x "${FILE}"
      echo -n " : marked non-executable"
      REFORMAT=1
    fi
    #
    # Check (naive) for scripts relying on in-place edit (sed).
    #
    if [ "$(basename "${FILE}")" != "$(basename "$0")" ] && \
       [[ ("*.sh" = "${PATTERN}") || ("*.slurm" = "${PATTERN}") ]] && \
       [ "$(${SED} -n '/sed[[:space:]][[:space:]]*-i/p' "${FILE}")" ];
    then
      echo " : use of sed -i is not portable"
      exit 1
    fi
    #
    # Reject code calling "exit" directly
    #
    if [ "${CPP}" ] && [[ ${FILE} = "${SRCDIR}/"* ]] && \
       [ "$(${CPP} 2>/dev/null -fpreprocessed "${FILE}" | ${SED} -n "/[^[:alnum:]_]exit[[:space:]]*(..*)/p")" ];
    then
      echo " : use LIBXSMM_EXIT_ERROR() instead of exit(...)"
      exit 1
    fi
    if [ "0" != "${REFORMAT}" ]; then
      WARNINGS=$((WARNINGS+1))
      echo
    else
      echo " : OK"
    fi
  done
  done
  if [ "0" != "${WARNINGS}" ]; then
    echo "Completed with format warnings."
  else
    echo "Completed successfully."
  fi
  exit 0
fi

>&2 echo "ERROR: missing prerequisites!"
exit 1
