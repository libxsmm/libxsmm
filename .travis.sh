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
#MKTEMP=$(which mktemp 2> /dev/null)
MKTEMP=${HERE}/.mktmp.sh
MKDIR=$(which mkdir 2> /dev/null)
CHMOD=$(which chmod 2> /dev/null)
SORT=$(which sort 2> /dev/null)
GREP=$(which grep 2> /dev/null)
SED=$(which sed 2> /dev/null)
TR=$(which tr 2> /dev/null)
WC=$(which wc 2> /dev/null)
RM=$(which rm 2> /dev/null)
CP=$(which cp 2> /dev/null)

if [ "" != "${MKTEMP}" ] && [ "" != "${MKDIR}" ]&& [ "" != "${CHMOD}" ] && \
   [ "" != "${SED}" ] && [ "" != "${TR}" ] && [ "" != "${WC}" ] && \
   [ "" != "${RM}" ] && [ "" != "${CP}" ]; \
then
  HOST=$(hostname -s 2> /dev/null)

  if [ "" != "${GREP}" ] && [ "" != "${SORT}" ] && [ -e /proc/cpuinfo ]; then
    export NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)
    export NC=$((NS*$(${GREP} "core id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)))
    export NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l)
  fi
  if [ "" != "${NC}" ] && [ "" != "${NT}" ]; then
    export HT=$((NT/(NC)))
    export MAKEJ="-j ${NC}"
  else
    export NS=1 NC=1 NT=1 HT=1
  fi

  if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
    export TRAVIS_BUILD_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH}
  fi
  if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
    export BUILDKITE_BUILD_CHECKOUT_PATH=${HERE}
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
  source ${HERE}/.env/travis.env
  source ${HERE}/.env/buildkite.env

  # setup PARTITIONS for multi-tests
  if [ "" = "${PARTITIONS}" ]; then
    if [ "" != "${PARTITION}" ]; then
      PARTITIONS=${PARTITION}
    else
      PARTITIONS=none
    fi
  fi

  # setup CONFIGS (multiple configurations)
  if [ "" = "${CONFIGS}" ]; then
    if [ "" != "${CONFIG}" ]; then
      CONFIGS=${CONFIG}
    else
      CONFIGS=none
    fi
  fi

  # select test-set ("travis" by default)
  if [ "" = "${TESTSET}" ]; then
    TESTSET=travis
  fi

  # setup batch execution
  if [ "" = "${LAUNCH}" ] && [ "" != "${SRUN}" ]; then
    if [ "" != "${SRUN_CPUS_PER_TASK}" ]; then
      SRUN_CPUS_PER_TASK_FLAG="--cpus-per-task=${SRUN_CPUS_PER_TASK}"
    fi
    umask 007
    TESTSCRIPT=$(${MKTEMP} ${HERE}/.libxsmm_XXXXXX.sh)
    ${CHMOD} +rx ${TESTSCRIPT}
    LAUNCH="${SRUN} --ntasks=1 ${SRUN_FLAGS} ${SRUN_CPUS_PER_TASK_FLAG} \
      --partition=\${PARTITION} --preserve-env --pty ${TESTSCRIPT} 2\> /dev/null"
  else # avoid temporary script in case of non-batch execution
    LAUNCH=\${TEST}
  fi
  if [ "" != "${LAUNCH_USER}" ]; then
    LAUNCH="su ${LAUNCH_USER} -p -c \'${LAUNCH}\'"
  fi

  RESULT=0
  while TEST=$(eval " \
    ${SED} -e '/^\s*script:\s*$/,\$!d' -e '/^\s*script:\s*$/d' ${HERE}/.${TESTSET}.yml | \
    ${SED} -nr \"/^\s*-\s*/H;//,/^\s*$/G;s/\n(\n[^\n]*){\${TESTID}}$//p\" | \
    ${SED} -e 's/^\s*-\s*//' -e 's/^\s\s*//' | ${TR} '\n' ' ' | \
    ${SED} -e 's/\s\s*$//'") && [ "" != "${TEST}" ];
  do
    for PARTITION in ${PARTITIONS}; do
    for CONFIG in ${CONFIGS}; do
      # print some header if all tests are selected or in case of multi-tests
      if [ "" = "$1" ] || [ "none" != "${PARTITION}" ]; then
        echo "================================================================================"
        if [ "none" != "${PARTITION}" ] && [ "0" != "${SHOW_PARTITION}" ]; then
          echo "Test Case #${TESTID} (${PARTITION})"
        else
          echo "Test Case #${TESTID}"
        fi
      fi

      # prepare temporary script for remote environment/execution
      if [ "" != "${TESTSCRIPT}" ] && [ -e ${TESTSCRIPT} ]; then
        echo "#!/bin/bash" > ${TESTSCRIPT}
        # make execution environment available
        if [ "" != "${HOST}" ] && [ "none" != "${CONFIG}" ] && \
           [ -e ${TRAVIS_BUILD_DIR}/.env/${HOST}_${CONFIG}.env ]; \
        then
          ${MKDIR} -p ${TRAVIS_BUILD_DIR}/licenses
          ${CP} -u /opt/intel/licenses/* ${TRAVIS_BUILD_DIR}/licenses 2> /dev/null
          echo "export INTEL_LICENSE_FILE=${TRAVIS_BUILD_DIR}/licenses" >> ${TESTSCRIPT}
          echo "source ${TRAVIS_BUILD_DIR}/.env/${HOST}_${CONFIG}.env" >> ${TESTSCRIPT}
        fi
        # record the actual test case
        echo "${TEST} 2>&1" >> ${TESTSCRIPT}
      else # make execution environment locally available
        if [ "" != "${HOST}" ] && [ "none" != "${CONFIG}" ] && \
           [ -e ${TRAVIS_BUILD_DIR}/.env/${HOST}_${CONFIG}.env ]; \
        then
          source ${TRAVIS_BUILD_DIR}/.env/${HOST}_${CONFIG}.env
        fi
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
    done # CONFIGS
    done # PARTITIONS

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

  # override result code (alternative outcome)
  if [ "" != "${RESULTCODE}" ]; then
    RESULT=${RESULTCODE}
  fi

  exit ${RESULT}
fi

