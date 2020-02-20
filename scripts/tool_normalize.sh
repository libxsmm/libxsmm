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

PATTERNS="*.c *.cc *.cpp *.cxx *.h *.hpp *.hxx *.f *.F90 *.fh *.py *.sh *.env *.yml *.txt *.slurm"
BANNED_CHARS="\t"

PATBAN="s/[${BANNED_CHARS}]/"
PATEOL="s/\r$/"
PATSPC="s/[[:space:]][[:space:]]*$/"

HERE=$(cd "$(dirname "$0")"; pwd -P)
REPO=${HERE}/..
CODEFILE=${REPO}/.codefile
MKTEMP=${REPO}/.mktmp.sh

FLAKE8=$(command -v flake8)
ICONV=$(command -v iconv)
GIT=$(command -v git)
SED=$(command -v sed)
CUT=$(command -v cut)
TR=$(command -v tr)
CP=$(command -v cp)
RM=$(command -v rm)

if [ -e "${CODEFILE}" ]; then
  PATTERNS="$(cat "${CODEFILE}")"
fi

if [ "" != "${FLAKE8}" ] && [ "0" = "$(${FLAKE8} 2>&1 >/dev/null; echo $?)" ] && \
   [ "0" != "$(${FLAKE8} ${HERE}/*.py 2>&1 >/dev/null; echo $?)" ];
then
  echo "Warning: some Python scripts do not pass flake8 check (${HERE})!"
fi

if [ "" != "${GIT}" ] && [ "" != "${CP}" ] && [ "" != "${RM}" ] && \
   [ "" != "${SED}" ] && [ "" != "${CUT}" ] && [ "" != "${TR}" ];
then
  if [ "" != "${ICONV}" ]; then
    CAT="${ICONV} -t ASCII"
  else
    CAT=$(command -v cat)
  fi
  if [ "" != "${CAT}" ]; then
    TMPF=$("${MKTEMP}" .libxsmm_XXXXXX.txt)
    # disable glob in Shell
    set -f
    # Search the content of the diffs matching the given file types
    for PATTERN in ${PATTERNS} *Makefile*; do
      for FILE in $(${GIT} ls-files ${PATTERN}); do
        if [[ ${FILE} != *"Makefile"* ]]; then
          if [ "" != "$(${SED} -n "${PATBAN}x/p" ${FILE} 2>/dev/null)" ]; then
            echo "Warning: ${FILE} contains banned characters!"
          fi
        else
          if [ "" != "$(${SED} -n "s/\([^[:space:]]\)\t/\1 /gp" ${FILE})" ]; then
            ${CAT} ${FILE} | ${SED} -e "s/\([^[:space:]]\)\t/\1 /g" > ${TMPF}
            ${CP} ${TMPF} ${FILE}
            echo "${FILE}: removed inner tabs."
          fi
        fi
        if [ "" != "$(${SED} -n "${PATEOL}x/p" ${FILE} 2>/dev/null | ${TR} -d "\n")" ]; then
          echo "Warning: ${FILE} uses non-UNIX line endings!"
        fi
        if [ "" != "$(${SED} -n "${PATSPC}x/p" ${FILE})" ]; then
          ${CAT} ${FILE} | ${SED} -e "${PATSPC}/" > ${TMPF}
          ${CP} ${TMPF} ${FILE}
          echo "${FILE}: removed trailing white spaces."
        fi
        FLAGS=$(${GIT} ls-files -s ${FILE} | ${CUT} -d' ' -f1)
        if [ "*.sh" = "${PATTERN}" ] || [ "*.py" = "${PATTERN}" ]; then
          if [ "" != "$(${SED} -n '1!b;/#!/p' ${FILE})" ] && \
             [ "100755" != "${FLAGS}" ];
          then
            ${GIT} update-index --chmod=+x ${FILE}
            echo "${FILE}: added executable flag."
          fi
        elif [ "100644" != "${FLAGS}" ] && [ "120000" != "${FLAGS}" ]; then
          ${GIT} update-index --chmod=-x ${FILE}
          echo "${FILE}: removed executable flag."
        fi
      done
    done
    ${RM} ${TMPF}
    echo "Successfully Completed."
    exit 0
  fi
fi

echo "Error: missing prerequisites!"
exit 1

