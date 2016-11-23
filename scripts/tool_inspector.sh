#!/bin/sh
#############################################################################
# Copyright (c) 2016, Intel Corporation                                     #
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

RPT=inspector
KIND=mi1

BASENAME=$(which basename 2> /dev/null)
TOOL=$(which inspxe-cl 2> /dev/null)
GREP=$(which grep 2> /dev/null)
SED=$(which sed 2> /dev/null)
RM=$(which rm 2> /dev/null)

if [ "" != "$1" ] && [ "" != "${BASENAME}" ] && [ "" != "${TOOL}" ] \
                  && [ "" != "${GREP}" ]     && [ "" != "${SED}" ] \
                  && [ "" != "${RM}" ];
then
  HERE=$(cd $(dirname $0); pwd -P)
  if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
    export TRAVIS_BUILD_DIR=${HERE}/..
  fi
  if [ "" != "${TESTID}" ]; then
    ID=${TESTID}
  fi
  if [ "" = "${ID}" ]; then
    ID=${COVID}
  fi
  if [ "" != "${ID}" ]; then
    RPTNAME=$(${BASENAME} $1)-${KIND}-${ID}
  else
    RPTNAME=$(${BASENAME} $1)-${KIND}
  fi

  DIR=${TRAVIS_BUILD_DIR}/${RPT}
  ${RM} -rf ${DIR}/${ID}

  ${TOOL} -collect ${KIND} -r ${DIR}/${ID} -no-auto-finalize -return-app-exitcode -- $*
  RESULT=$?

  if [ "0" = "${RESULT}" ]; then
    ${TOOL} -report problems -r ${DIR}/${ID} > ${DIR}/${RPTNAME}.txt
    RESULT2=$?

    if [ "" = "${TOOL_REPORT_ONLY}" ] && [ "0" != "$((2<RESULT2))" ]; then
      if [ "" = "${TOOL_FILTER}" ] || \
         [ "" != "$(${GREP} 'Function' ${DIR}/${RPTNAME}.txt   | \
                    ${SED} -e 's/..* Function \(..*\):..*/\1/' | \
                    ${GREP} ${TOOL_FILTER})" ];
      then
        RESULT=${RESULT2}
      fi
    fi
  fi
  exit ${RESULT}
else
  $*
fi

