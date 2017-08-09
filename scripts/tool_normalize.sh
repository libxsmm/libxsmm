#!/bin/sh
#############################################################################
# Copyright (c) 2017, Intel Corporation                                     #
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

HERE=$(cd $(dirname $0); pwd -P)
REPO=${HERE}/..
CODEFILE=${REPO}/.codefile
MKTEMP=${REPO}/.mktmp.sh

ICONV=$(which iconv 2> /dev/null)
ECHO=$(which echo 2> /dev/null)
GIT=$(which git 2> /dev/null)
SED=$(which sed 2> /dev/null)
CP=$(which cp 2> /dev/null)
RM=$(which rm 2> /dev/null)

if [ -e ${CODEFILE} ]; then
  PATTERNS="$(cat ${CODEFILE})"
fi

if [ "" != "${ICONV}" ] && [ "" != "${ECHO}" ] && [ "" != "${GIT}" ] && [ "" != "${CP}" ] && [ "" != "${RM}" ]; then
  TMPF=$(${MKTEMP} .libxsmm_XXXXXX.txt)

  # disable glob in Shell
  set -f
  # Search the content of the diffs matching the given file types
  for PATTERN in ${PATTERNS}; do
    for FILE in $("${GIT}" ls-files ${PATTERN}); do
      if [ "" != "$(${SED} -n /[${BANNED_CHARS}]/p ${FILE} 2> /dev/null)" ]; then
        ${ECHO} "Warning: ${FILE} contains banned characters!"
      fi
      ${ICONV} -t ASCII ${FILE} | ${SED} -e "s/\s\s*$//" > ${TMPF}
      ${CP} ${TMPF} ${FILE}
    done
  done

  ${RM} ${TMPF}
  ${ECHO} "Successfully Completed."
else
  ${ECHO} "Error: missing prerequisites!"
  exit 1
fi

