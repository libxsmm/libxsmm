#!/bin/bash
#############################################################################
# Copyright (c) 2016-2018, Intel Corporation                                #
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
set -o pipefail

HERE=$(cd $(dirname $0); pwd -P)
MKDIR=$(which mkdir 2>/dev/null)
CHMOD=$(which chmod 2>/dev/null)
UNAME=$(which uname 2>/dev/null)
ECHO=$(which echo 2>/dev/null)
SYNC=$(which sync 2>/dev/null)
SORT=$(which sort 2>/dev/null)
GREP=$(which grep 2>/dev/null)
WGET=$(which wget 2>/dev/null)
GIT=$(which git 2>/dev/null)
SED=$(which sed 2>/dev/null)
CUT=$(which cut 2>/dev/null)
TR=$(which tr 2>/dev/null)
WC=$(which wc 2>/dev/null)
RM=$(which rm 2>/dev/null)
CP=$(which cp 2>/dev/null)

MKTEMP=${HERE}/../.mktmp.sh
FASTCI=$2

RUN_CMD="--session-command"
#RUN_CMD="-c"

if [ "" != "${WGET}" ] && \
   [ "" != "${BUILDKITE_ORGANIZATION_SLUG}" ] && \
   [ "" != "${BUILDKITE_PIPELINE_SLUG}" ] && \
   [ "" != "${BUILDKITE_AGENT_ACCESS_TOKEN}" ];
then
  REVSTART=$(${WGET} -qO- \
  https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${BUILDKITE_PIPELINE_SLUG}/builds?access_token=${BUILDKITE_AGENT_ACCESS_TOKEN} \
  | ${SED} -n '/ *\"state\": \"passed\"/,/ *\"commit\": / {0,/ *\"commit\": / s/ *\"commit\": \"\(..*\)\".*/\1/p}')
fi
if [ "" = "${REVSTART}" ]; then
  REVSTART="HEAD^"
fi

if [ "" = "${FULLCI}" ] || [ "0" = "${FULLCI}" ]; then
  FULLCI="\[full ci\]"
fi

if [ "" != "${MKTEMP}" ] && [ "" != "${MKDIR}" ] && [ "" != "${CHMOD}" ] && [ "" != "${ECHO}" ] && \
   [ "" != "${GREP}" ] && [ "" != "${SED}" ] && [ "" != "${TR}" ] && [ "" != "${WC}" ] && \
   [ "" != "${RM}" ] && [ "" != "${CP}" ];
then
  # check if full tests are triggered (allows to skip the detailed investigation)
  if [ "webhook" = "${BUILDKITE_SOURCE}" ] && \
     [ "" != "${FASTCI}" ] && [ -e ${FASTCI} ] && [ "" != "${GIT}" ] && [ "1" != "${FULLCI}" ] && \
     [ "" = "$(${GIT} log ${REVSTART}...HEAD 2>/dev/null | ${GREP} -e "${FULLCI}")" ];
  then
    # transform wild-card patterns to regular expressions
    PATTERNS="$(${SED} -e 's/\./\\./g' -e 's/\*/..*/g' -e 's/?/./g' -e 's/$/\$/g' ${FASTCI} 2>/dev/null)"
    DOTESTS=0
    if [ "" != "${PATTERNS}" ]; then
      for FILENAME in $(${GIT} diff --name-only ${REVSTART} HEAD 2>/dev/null); do
        # check if the file is supposed to impact a build (source code or script)
        for PATTERN in ${PATTERNS}; do
          MATCH=$(${ECHO} "${FILENAME}" | ${GREP} -e "${PATTERN}" 2>/dev/null)
          if [ "" != "${MATCH}" ]; then # file would impact the build
            DOTESTS=1
            break
          fi
        done
        if [ "0" != "${DOTESTS}" ]; then
          break
        fi
      done
    else
      DOTESTS=1
    fi
    if [ "0" = "${DOTESTS}" ]; then
      ${ECHO} "================================================================================"
      ${ECHO} "Skipped test(s) due to FASTCI option."
      ${ECHO} "================================================================================"
      exit 0 # skip tests
    fi
  fi

  HOST=$(hostname -s 2>/dev/null)
  if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
    export TRAVIS_BUILD_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH}
  fi
  if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
    export BUILDKITE_BUILD_CHECKOUT_PATH=${HERE}/..
    export TRAVIS_BUILD_DIR=${HERE}/..
  fi
  if [ "" = "${TRAVIS_OS_NAME}" ] && [ "" != "${UNAME}" ]; then
    export TRAVIS_OS_NAME=$(${UNAME})
  fi

  if [ "" != "${SORT}" ] && [ "" != "${WC}" ] && [ -e /proc/cpuinfo ]; then
    export NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)
    export NC=$((NS*$(${GREP} "core id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)))
    export NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l)
  elif [ "" != "${UNAME}" ] && [ "" != "${CUT}" ] && [ "Darwin" = "$(${UNAME})" ]; then
    export NS=$(sysctl hw.packages | ${CUT} -d: -f2 | tr -d " ")
    export NC=$(sysctl hw.physicalcpu | ${CUT} -d: -f2 | tr -d " ")
    export NT=$(sysctl hw.logicalcpu | ${CUT} -d: -f2 | tr -d " ")
  fi
  if [ "" != "${NC}" ] && [ "" != "${NT}" ]; then
    export HT=$((NT/(NC)))
    export MAKEJ="-j ${NC}"
  else
    export NS=1 NC=1 NT=1 HT=1
  fi
  if [ "" != "${CUT}" ] && [ "" != "$(which numactl 2>/dev/null)" ]; then
    export NN=$(numactl -H | ${GREP} available: | ${CUT} -d' ' -f2)
  else
    export NN=${NS}
  fi

  # set the case number
  if [ "" != "$1" ]; then
    export TESTID=$1
  else
    export TESTID=1
  fi

  # should be source'd after the above variables are set
  source ${HERE}/../.env/travis.env
  source ${HERE}/../.env/buildkite.env

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
  if [ -e .${TESTSET}.yml ]; then
    TESTSETFILE=.${TESTSET}.yml
  elif [ -e ${TESTSET}.yml ]; then
    TESTSETFILE=${TESTSET}.yml
  elif [ -e ${TESTSET} ]; then
    TESTSETFILE=${TESTSET}
  else
    ${ECHO} "ERROR: Cannot find file with test set!"
    exit 1
  fi

  # setup batch execution
  if [ "" = "${LAUNCH}" ] && [ "" != "${SRUN}" ]; then
    if [ "" != "${BUILDKITE_LABEL}" ]; then
      LABEL=$(${ECHO} "${BUILDKITE_LABEL}" | ${TR} -s [:punct:][:space:] - | ${SED} -e "s/^-//" -e "s/-$//")
    fi
    if [ "" != "${LABEL}" ]; then
      SRUN_FLAGS="${SRUN_FLAGS} -J ${LABEL}"
      TESTSCRIPT=${HERE}/../.libxsmm_test-${LABEL}.sh
    fi
    umask 007
    if [ "" != "${TESTSCRIPT}" ]; then
      touch ${TESTSCRIPT}
    else
      TESTSCRIPT=$(${MKTEMP} ${HERE}/../.libxsmm_XXXXXX.sh)
    fi
    ${CHMOD} +rx ${TESTSCRIPT}
    LAUNCH="${SRUN} --ntasks=1 --partition=\${PARTITION} ${SRUN_FLAGS} --preserve-env --pty ${TESTSCRIPT} 2\>/dev/null"
  else # avoid temporary script in case of non-batch execution
    LAUNCH=\${TEST}
  fi
  if [ "" != "${LAUNCH_USER}" ]; then
    LAUNCH="su ${LAUNCH_USER} -p ${RUN_CMD} \'${LAUNCH}\'"
  fi

  RESULT=0
  while TEST=$(eval " \
    ${SED} -n -e '/^ *script: *$/,\$p' ${HERE}/../${TESTSETFILE} | ${SED} -e '/^ *script: *$/d' | \
    ${SED} -n -E \"/^ *- */H;//,/^ *$/G;s/\n(\n[^\n]*){\${TESTID}}$//p\" | \
    ${SED} -e 's/^ *- *//' -e 's/^  *//' | ${TR} '\n' ' ' | \
    ${SED} -e 's/  *$//'") && [ "" != "${TEST}" ];
  do
    for PARTITION in ${PARTITIONS}; do
    for CONFIG in ${CONFIGS}; do
      # print some header if all tests are selected or in case of multi-tests
      if [ "" = "$1" ] || [ "none" != "${PARTITION}" ]; then
        ${ECHO} "================================================================================"
        if [ "none" != "${PARTITION}" ] && [ "0" != "${SHOW_PARTITION}" ]; then
          ${ECHO} "Test Case #${TESTID} (${PARTITION})"
        else
          ${ECHO} "Test Case #${TESTID}"
        fi
      fi
      ${ECHO} "^^^ +++"

      # make execution environment locally available (always)
      if [ "" != "${HOST}" ] && [ "none" != "${CONFIG}" ] && \
         [ -e ${TRAVIS_BUILD_DIR}/.env/${HOST}/${CONFIG}.env ];
      then
        source ${TRAVIS_BUILD_DIR}/.env/${HOST}/${CONFIG}.env
      fi

      # prepare temporary script for remote environment/execution
      if [ "" != "${TESTSCRIPT}" ] && [ -e ${TESTSCRIPT} ]; then
        ${ECHO} "#!/bin/bash" > ${TESTSCRIPT}
        # make execution environment available
        if [ "" != "${HOST}" ] && [ "none" != "${CONFIG}" ] && \
           [ -e ${TRAVIS_BUILD_DIR}/.env/${HOST}/${CONFIG}.env ];
        then
          LICSDIR=$(which icc 2>/dev/null | ${SED} -e "s/\(\/.*intel\)\/.*$/\1/")
          ${MKDIR} -p ${TRAVIS_BUILD_DIR}/licenses
          ${CP} -u /opt/intel/licenses/* ${TRAVIS_BUILD_DIR}/licenses 2>/dev/null
          ${CP} -u ${LICSDIR}/licenses/* ${TRAVIS_BUILD_DIR}/licenses 2>/dev/null
          ${ECHO} "export INTEL_LICENSE_FILE=${TRAVIS_BUILD_DIR}/licenses" >> ${TESTSCRIPT}
          ${ECHO} "source ${TRAVIS_BUILD_DIR}/.env/${HOST}/${CONFIG}.env" >> ${TESTSCRIPT}
        fi
        # record the current test case
        ${ECHO} "${TEST}" >> ${TESTSCRIPT}

        if [ "" != "${SYNC}" ]; then # flush asynchronous NFS mount
          ${SYNC}
        fi
      fi

      COMMAND=$(eval ${ECHO} ${LAUNCH})
      # run the prepared test case/script
      if [ "" != "${LABEL}" ]; then
        eval ${COMMAND} 2>&1 | tee .test-${LABEL}.log
      else
        eval ${COMMAND}
      fi

      # capture test status
      RESULT=$?

      # exit the loop in case of an error
      if [ "0" = "${RESULT}" ]; then
        ${ECHO} "--- ------------------------------------------------------------------------------"
        ${ECHO} "SUCCESS"
        ${ECHO}
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
    ${ECHO} "^^^ +++"
    ${ECHO} "--- ------------------------------------------------------------------------------"
    ${ECHO} "FAILURE"
    ${ECHO}
  fi

  # override result code (alternative outcome)
  if [ "" != "${RESULTCODE}" ]; then
    RESULT=${RESULTCODE}
  fi

  exit ${RESULT}
fi

