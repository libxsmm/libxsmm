#!/bin/bash
#############################################################################
# Copyright (c) 2017-2019, Intel Corporation                                #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions        #
# are met:                                                                  #
# 1. Redistributions of source code must retain the above copyright         #
#    notice, this list of conditions and the following disclaimer.          #
# 2. Redistributions in binary form must reproduce the above copyright      #
#    notice, this list of conditions and the following disclaimer in the    #
#    documentation and/or other materials provided with the distribution.   #
# 3. Neither the name of the copyright holder nor the names of its          #
#    contributors may be used to endorse or promote products derived        #
#    from this software without specific prior written permission.          #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#############################################################################
# Hans Pabst (Intel Corp.)
#############################################################################

PATTERNS="*.c *.cpp *.h *.hpp *.f *.F90 *.fh *.sh *.py *.yml *.txt"
BANNED_CHARS="\t"

PATBAN="s/[${BANNED_CHARS}]/"
PATEOL="s/\r$/"
PATSPC="s/[[:space:]][[:space:]]*$/"

HERE=$(cd $(dirname $0); pwd -P)
REPO=${HERE}/..
CODEFILE=${REPO}/.codefile
MKTEMP=${REPO}/.mktmp.sh

FLAKE8=$(command -v flake8)
ICONV=$(command -v iconv)
GIT=$(command -v git)
SED=$(command -v sed)
TR=$(command -v tr)
CP=$(command -v cp)
RM=$(command -v rm)

if [ -e ${CODEFILE} ]; then
  PATTERNS="$(cat ${CODEFILE})"
fi

if [ "" != "${FLAKE8}" ] && [ "0" = "$(${FLAKE8} 2>&1 >/dev/null; echo $?)" ] && \
   [ "0" != "$(${FLAKE8} ${HERE}/*.py 2>&1 >/dev/null; echo $?)" ];
then
  echo "Warning: some Python scripts do not pass flake8 check (${HERE})!"
fi

if [ "" != "${GIT}" ] && [ "" != "${CP}" ] && [ "" != "${RM}" ] && \
   [ "" != "${SED}" ] && [ "" != "${TR}" ];
then
  if [ "" != "${ICONV}" ]; then
    CAT="${ICONV} -t ASCII"
  else
    CAT=$(command -v cat)
  fi
  if [ "" != "${CAT}" ]; then
    TMPF=$(${MKTEMP} .libxsmm_XXXXXX.txt)
    # disable glob in Shell
    set -f
    # Search the content of the diffs matching the given file types
    for PATTERN in ${PATTERNS}; do
      for FILE in $(${GIT} ls-files ${PATTERN}); do
        if [ "" != "$(${SED} -n "${PATBAN}x/p" ${FILE} 2>/dev/null)" ]; then
          echo "Warning: ${FILE} contains banned characters!"
        fi
        if [ "" != "$(${SED} -n "${PATEOL}x/p" ${FILE} 2>/dev/null | ${TR} -d "\n")" ]; then
          echo "Warning: ${FILE} uses non-UNIX line endings!"
        fi
        if [ "" != "$(${SED} -n "${PATSPC}x/p" ${FILE})" ]; then
          ${CAT} ${FILE} | ${SED} -e "${PATSPC}/" > ${TMPF}
          ${CP} ${TMPF} ${FILE}
          echo "${FILE}: removed trailing white spaces."
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

