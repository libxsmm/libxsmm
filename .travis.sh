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

HERE=$(cd $(dirname $0); pwd -P)
SED=$(which sed)
TR=$(which tr)

if [ "" != "${SED}" ] && [ "" != "${TR}" ]; then
  if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
    export TRAVIS_BUILD_DIR=${HERE}
  fi
  if [ "" = "${TRAVIS_OS_NAME}" ] && [ "" != "$(which uname)" ]; then
    export TRAVIS_OS_NAME=$(uname)
  fi

  # set the case number
  if [ "" != "$1" ]; then
    export TESTID=$1
  else
    export TESTID=1
  fi

  # should be source'd after the above variables are set
  source ${HERE}/.travis.env
  source ${HERE}/.buildkite.env

  # clear build log
  if [ -e ${HERE}/log.txt ]; then
    if [ "" = "$1" ] || [ "0" = "$1" ]; then
      cat /dev/null > ${HERE}/log.txt
    fi
  fi

  while TEST=$(eval " \
    ${SED} -e '/^\s*script:\s*$/,\$!d' -e '/^\s*script:\s*$/d' ${HERE}/.travis.yml | \
    ${SED} -nr \"/^\s*-\s*/H;//,/^\s*$/G;s/\n(\n[^\n]*){\${TESTID}}$//p\" | \
    ${SED} -e 's/^\s*-\s*//' -e 's/^\s\s*//' | ${TR} '\n' ' ' | \
    ${SED} -e 's/\s\s*$//'") && [ "" != "${TEST}" ];
  do
    # print header if all test cases are selected
    if [ "" = "$1" ]; then
      echo "================================================================================"
      echo "Test Case #${TESTID}"
    fi

    # run the actual test case
    eval ${TEST}
    RESULT=$?

    # increment the case number if all cases are selected or leave the loop
    if [ "0" = "${RESULT}" ] && [ "" = "$1" ]; then
      TESTID=$((TESTID+1))
    else # dummy/exit case
      exit ${RESULT}
    fi
  done
fi

