#!/bin/bash
#############################################################################
# Copyright (c) 2016-2017, Intel Corporation                                #
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
MKTEMP=$(which mktemp 2> /dev/null)
CHMOD=$(which chmod 2> /dev/null)
SED=$(which sed 2> /dev/null)
TR=$(which tr 2> /dev/null)
RM=$(which rm 2> /dev/null)

if [ "" != "${MKTEMP}" ] && [ "" != "${CHMOD}" ] && [ "" != "${SED}" ] && [ "" != "${TR}" ] && [ "" != "${RM}" ]; then
  if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
    export TRAVIS_BUILD_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH}
  fi
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

  # setup PARTITIONS for multi-tests
  if [ "" = "${PARTITIONS}" ]; then
    PARTITIONS=none
  fi

  # setup batch execution
  if [ "" = "${LAUNCH}" ] && [ "" != "${SRUN}" ]; then
    if [ "" = "${SRUN_CPUS_PER_TASK}" ]; then SRUN_CPUS_PER_TASK=2; fi
    TESTSCRIPT=$(${MKTEMP} ${HERE}/.libxsmm_XXXXXX.sh)
    ${CHMOD} a+rwx ${TESTSCRIPT}
    LAUNCH="${SRUN} \
      --ntasks=1 --cpus-per-task=${SRUN_CPUS_PER_TASK} \
      --partition=\${PARTITION} --preserve-env --pty bash -l ${TESTSCRIPT}"
  else # avoid temporary script in case of non-batch execution
    LAUNCH=\${TEST}
  fi
  if [ "" != "${LAUNCH_USER}" ]; then
    LAUNCH="su ${LAUNCH_USER} -c \'${LAUNCH}\'"
  fi

  RESULT=0
  while TEST=$(eval " \
    ${SED} -e '/^\s*script:\s*$/,\$!d' -e '/^\s*script:\s*$/d' ${HERE}/.travis.yml | \
    ${SED} -nr \"/^\s*-\s*/H;//,/^\s*$/G;s/\n(\n[^\n]*){\${TESTID}}$//p\" | \
    ${SED} -e 's/^\s*-\s*//' -e 's/^\s\s*//' | ${TR} '\n' ' ' | \
    ${SED} -e 's/\s\s*$//'") && [ "" != "${TEST}" ];
  do
    for PARTITION in ${PARTITIONS}; do
      # print some header if all tests are selected or in case of multi-tests
      if [ "" = "$1" ] || [ "none" != "${PARTITION}" ]; then
        echo "================================================================================"
        if [ "none" != "${PARTITION}" ]; then
          echo "Test Case #${TESTID} (${PARTITION})"
        else
          echo "Test Case #${TESTID}"
        fi
      fi

      # prepare temporary script
      if [ "" != "${TESTSCRIPT}" ] && [ -e ${TESTSCRIPT} ]; then
        echo "#!/bin/bash" > ${TESTSCRIPT}
        # re-source the required environment
        echo "source ${TRAVIS_BUILD_DIR}/.travis.env" >> ${TESTSCRIPT}
        echo "source ${TRAVIS_BUILD_DIR}/.buildkite.env" >> ${TESTSCRIPT}
        # record the actual test case
        echo "${TEST}" >> ${TESTSCRIPT}
      fi

      # run the prepared test case/script
      eval $(eval echo ${LAUNCH})

      # capture test status
      RESULT=$?

      # exit the loop in case of an error
      if [ "0" = "${RESULT}" ]; then
        echo "--------------------------------------------------------------------------------"
        echo "SUCCESS"
        echo
      else
        break
      fi
    done

    # increment the case number, or exit the script
    if [ "" = "$1" ] && [ "0" = "${RESULT}" ]; then
      TESTID=$((TESTID+1))
    else # finish
      break
    fi
  done

  # remove temporary script (if it exists)
  if [ "" != "${TESTSCRIPT}" ] && [ -e ${TESTSCRIPT} ]; then
    ${RM} ${TESTSCRIPT}
  fi

  if [ "0" != "${RESULT}" ]; then
    echo "--------------------------------------------------------------------------------"
    echo "FAILURE"
    echo
  fi

  exit ${RESULT}
fi

