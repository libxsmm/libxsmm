#!/bin/bash
#############################################################################
# Copyright (c) 2016-2019, Intel Corporation                                #
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
BASENAME=$(command -v basename)
MKDIR=$(command -v mkdir)
CHMOD=$(command -v chmod)
UNAME=$(command -v uname)
SYNC=$(command -v sync)
SORT=$(command -v sort)
GREP=$(command -v grep)
WGET=$(command -v wget)
GIT=$(command -v git)
SED=$(command -v sed)
CUT=$(command -v cut)
REV=$(command -v rev)
TR=$(command -v tr)
WC=$(command -v wc)
RM=$(command -v rm)
CP=$(command -v cp)

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
  | ${SED} -n '/ *\"commit\": / {0,/ *\"commit\": / s/ *\"commit\": \"\(..*\)\".*/\1/p}')
fi
if [ "" = "${REVSTART}" ]; then
  REVSTART="HEAD^"
fi

if [ "" = "${FULLCI}" ] || [ "0" = "${FULLCI}" ]; then
  FULLCI="\[full ci\]"
fi

if [ "" != "${MKTEMP}" ] && [ "" != "${MKDIR}" ] && [ "" != "${CHMOD}" ] && \
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
          MATCH=$(echo "${FILENAME}" | ${GREP} -e "${PATTERN}" 2>/dev/null)
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
      echo "================================================================================"
      echo "Skipped test(s) due to FASTCI option."
      echo "================================================================================"
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

  # set the case number
  if [ "" != "$1" ] && [ -e $1 ]; then
    export TESTSETFILE=$1
    if [ "" != "${BASENAME}" ] && [ "" != "${REV}" ] && [ "" != "${CUT}" ]; then
      export TESTID=$(${BASENAME} ${TESTSETFILE%.*})
      if [ "" = "${TESTID}" ]; then
        export TESTID=$(${BASENAME} ${TESTSETFILE} | ${REV} | ${CUT} -d. -f1 | ${REV})
      fi
    else
      export TESTID=${TESTSETFILE}
    fi
    export TESTSET=${TESTID}
  else
    if [ "" != "$1" ]; then
      export TESTID=$1
    else
      export TESTID=1
    fi
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
  # setup ENVS (multiple environments)
  if [ "" = "${ENVS}" ]; then
    if [ "" != "${ENV}" ]; then
      ENVS=${ENV}
    else
      ENVS=none
    fi
  fi

  # select test-set ("travis" by default)
  if [ "" = "${TESTSET}" ]; then
    TESTSET=travis
  fi
  if [ "" = "${TESTSETFILE}" ] || [ ! -e ${TESTSETFILE} ]; then
    if [ -e .${TESTSET}.yml ]; then
      TESTSETFILE=.${TESTSET}.yml
    elif [ -e ${TESTSET}.yml ]; then
      TESTSETFILE=${TESTSET}.yml
    elif [ -e ${TESTSET} ]; then
      TESTSETFILE=${TESTSET}
    else
      echo "ERROR: Cannot find file with test set!"
      exit 1
    fi
  else
    TEST=${TESTSETFILE}
  fi

  # setup batch execution (TEST may be a singular test given by filename)
  if [ "" = "${LAUNCH}" ] && [ "" != "${SRUN}" ] && [ "0" != "${SLURM}" ]; then
    if [ "" != "${BUILDKITE_LABEL}" ]; then
      LABEL=$(echo "${BUILDKITE_LABEL}" | ${TR} -s [:punct:][:space:] - | ${SED} -e "s/^-//" -e "s/-$//")
    fi
    if [ "" != "${LABEL}" ]; then
      SRUN_FLAGS="${SRUN_FLAGS} -J ${LABEL}"
    fi
    if [ "" != "${LIMITRUN}" ]; then
      if [ "" = "${LIMIT}" ] || [ "0" = "${LIMIT}" ]; then
        SRUN_FLAGS="${SRUN_FLAGS} --time=${LIMITRUN}"
      else # seconds -> minutes
        SRUN_FLAGS="${SRUN_FLAGS} --time=$((LIMIT/60))"
      fi
    fi
    umask 007
    # eventually cleanup run-script from terminated sessions
    ${RM} -f ${HERE}/../.tool_??????.sh
    TESTSCRIPT=$(${MKTEMP} ${HERE}/../.tool_XXXXXX.sh)
    ${CHMOD} +rx ${TESTSCRIPT}
    LAUNCH="${SRUN} --ntasks=1 --partition=\${PARTITION} ${SRUN_FLAGS} --preserve-env --unbuffered ${TESTSCRIPT}"
  else # avoid temporary script in case of non-batch execution
    if [ "" = "${MAKEJ}" ]; then
      export MAKEJ="-j $(eval ${HERE}/tool_cpuinfo.sh -nc)"
    fi
    SHOW_PARTITION=0
    LAUNCH="\${TEST}"
  fi
  if [ "" != "${LAUNCH_USER}" ] && [ "0" != "${SLURM}" ]; then
    LAUNCH="su ${LAUNCH_USER} -p ${RUN_CMD} \'${LAUNCH}\'"
  fi

  RESULT=0
  # control log
  echo && echo "^^^ +++"
  while [ "" != "${TEST}" ] || TEST=$(eval " \
    ${SED} -n -e '/^ *script: *$/,\$p' ${HERE}/../${TESTSETFILE} | ${SED} -e '/^ *script: *$/d' | \
    ${SED} -n -E \"/^ *- */H;//,/^ *$/G;s/\n(\n[^\n]*){\${TESTID}}$//p\" | \
    ${SED} -e 's/^ *- *//' -e 's/^  *//' | ${TR} '\n' ' ' | \
    ${SED} -e 's/  *$//'") && [ "" != "${TEST}" ];
  do
    if [ -d "${TEST}" ]; then
      SLURMDIR=${TEST}
    else # dummy
      SLURMDIR=$0
    fi
    for SLURMFILE in $(ls -1 ${SLURMDIR}); do
    if [ -d ${SLURMDIR} ]; then
      SLURMFILE=${SLURMDIR}/${SLURMFILE}
      TESTID=$(${BASENAME} ${SLURMFILE%.*})
    fi
    if [ "$0" != "${SLURMFILE}" ] && [ -e ${SLURMFILE} ]; then
      if [ "" != "${LIMIT}" ] && [ "0" != "${LIMIT}" ] && \
         [ "" != "$(command -v touch)" ] && \
         [ "" != "$(command -v stat)" ] && \
         [ "" != "$(command -v date)" ];
      then
        NOW=$(date +%s)
        if [ "" != "${LIMITDIR}" ] && [ -d ${LIMITDIR} ]; then
          LIMITFILE=${LIMITDIR}/$(basename ${SLURMFILE})
          if [ ! -e ${LIMITFILE} ]; then OLD=${NOW}; fi
        else
          LIMITFILE=${SLURMFILE}
          OLD=$(stat -c %Y ${LIMITFILE})
        fi
        if [ "0" != "$(((OLD+LIMIT)<=NOW))" ]; then
          echo "================================================================================"
          echo "Skipped ${TESTID} due to LIMIT=${LIMIT}."
          echo "================================================================================"
          continue
        else
          TOUCH=${LIMITFILE}
        fi
      fi
      if [ "none" = "${PARTITIONS}" ]; then
        PARTITION=$(sed -n "s/^#SBATCH[[:space:]][[:space:]]*\(--partition=\|-p\)\(..*\)/\2/p" ${SLURMFILE})
        if [ "" != "${PARTITION}" ]; then
          PARTITIONS=${PARTITION}
        fi
      fi
    fi
    for PARTITION in ${PARTITIONS}; do
    for CONFIG in ${CONFIGS}; do
    # make execution environment locally available (always)
    if [ "" != "${HOST}" ] && [ "none" != "${CONFIG}" ] && \
       [ -e ${TRAVIS_BUILD_DIR}/.env/${HOST}/${CONFIG}.env ];
    then
      source ${TRAVIS_BUILD_DIR}/.env/${HOST}/${CONFIG}.env
    fi
    for ENV in ${ENVS}; do
      if [ "none" != "${ENV}" ]; then
        if [ "" != "${CUT}" ]; then ENVVAL=$(echo "${ENV}" | ${CUT} -d= -f2); fi
        ENVSTR=${ENV}
      fi
      # print some header if all tests are selected or in case of multi-tests
      if [ "" = "$1" ] || [ "none" != "${PARTITION}" ] || [ "none" != "${ENV}" ]; then
        if [ "none" != "${PARTITION}" ] && [ "0" != "${SHOW_PARTITION}" ]; then
          if [ "" != "${ENVVAL}" ]; then
            echo "+++ TEST ${TESTID} (${PARTITION}/${ENVVAL})"
          else
            echo "+++ TEST ${TESTID} (${PARTITION})"
          fi
        elif [ "" != "${ENVVAL}" ]; then
          echo "+++ TEST ${TESTID} (${ENVVAL})"
        else
          echo "+++ TEST ${TESTID}"
        fi
      fi
      # prepare temporary script for remote environment/execution
      if [ "" != "${TESTSCRIPT}" ] && [ -e ${TESTSCRIPT} ]; then
        echo "#!/bin/bash" > ${TESTSCRIPT}
        echo "set -o pipefail" >> ${TESTSCRIPT}
        echo "if [ \"\" = \"\${MAKEJ}\" ]; then MAKEJ=\"-j \$(eval ${HERE}/tool_cpuinfo.sh -nc)\"; fi" >> ${TESTSCRIPT}
        # make execution environment available
        if [ "" != "${HOST}" ] && [ "none" != "${CONFIG}" ] && \
           [ -e ${TRAVIS_BUILD_DIR}/.env/${HOST}/${CONFIG}.env ];
        then
          LICSDIR=$(command -v icc | ${SED} -e "s/\(\/.*intel\)\/.*$/\1/")
          ${MKDIR} -p ${TRAVIS_BUILD_DIR}/licenses
          ${CP} -u /opt/intel/licenses/* ${TRAVIS_BUILD_DIR}/licenses 2>/dev/null
          ${CP} -u ${LICSDIR}/licenses/* ${TRAVIS_BUILD_DIR}/licenses 2>/dev/null
          echo "export INTEL_LICENSE_FILE=${TRAVIS_BUILD_DIR}/licenses" >> ${TESTSCRIPT}
          echo "source ${TRAVIS_BUILD_DIR}/.env/${HOST}/${CONFIG}.env" >> ${TESTSCRIPT}
        fi
        if [ "" != "${SLURMSCRIPT}" ] && [ "0" != "${SLURMSCRIPT}" ] && [ -e "${TEST}" ]; then
          SLURMFILE=${TEST}
        fi
        # record the current test case
        if [ "$0" != "${SLURMFILE}" ] && [ -e ${SLURMFILE} ]; then
          DIR=$(cd $(dirname ${SLURMFILE}); pwd -P)
          if [ -e ${DIR}/../Makefile ]; then
            DIR=${DIR}/..
          fi
          echo "cd ${TRAVIS_BUILD_DIR} && make \${MAKEJ} && cd ${DIR} && make \${MAKEJ}" >> ${TESTSCRIPT}
          echo "RESULT=\$?" >> ${TESTSCRIPT}
          echo "if [ \"0\" != \"\${RESULT}\" ]; then exit \${RESULT}; fi" >> ${TESTSCRIPT}
          # control log
          echo "echo \"--- RUN ${TESTID}\"" >> ${TESTSCRIPT}
          if [ "" != "${LIMITLOG}" ] && [ "" != "$(command -v cat)" ] && [ "" != "$(command -v tail)" ]; then
            echo "(" >> ${TESTSCRIPT}
          fi
          DIRSED=$(echo "${DIR}" | ${SED} "s/\//\\\\\//g")
          ${SED} \
            -e "/^#\!..*/d" \
            -e "/^#SBATCH/d" \
            -e "/^[[:space:]]*$/d" \
            -e "s/\.\//${DIRSED}\//" \
            -e "s/^[./]*\([[:print:]][[:print:]]*\/\)*slurm[[:space:]][[:space:]]*//" \
            ${SLURMFILE} >> ${TESTSCRIPT}
          if [ "" != "${LIMITLOG}" ] && [ "" != "$(command -v cat)" ] && [ "" != "$(command -v tail)" ]; then
            echo ") | cat -s | tail -n ${LIMITLOG}" >> ${TESTSCRIPT}
          fi
          # clear captured test
          TEST=""
        else
          echo "${TEST}" >> ${TESTSCRIPT}
        fi

        if [ "" != "${SYNC}" ]; then # flush asynchronous NFS mount
          ${SYNC}
        fi
      fi

      COMMAND=$(eval echo "${ENVSTR} ${LAUNCH}")
      # run the prepared test case/script
      if [ "" != "${LABEL}" ] && [ "" != "$(command -v tee)" ]; then
        if [ -t 0 ]; then
          eval "${COMMAND} 2>&1 | tee .test-${LABEL}.log"
        else
          eval "${COMMAND} 2>&1 | ${GREP} -v '^srun: error:' | tee .test-${LABEL}.log"
        fi
      else
        eval "${COMMAND}"
      fi
      # capture test status
      RESULT=$?

      # exit the loop in case of an error
      if [ "0" != "${RESULT}" ]; then
        break 4
      fi
    done # ENVS
    done # CONFIGS
    done # PARTITIONS
    if [ "" != "${TOUCH}" ] && [ -e ${TOUCH} ]; then
      touch ${TOUCH}
      TOUCH=""
    fi
    done # SLURMFILE

    # increment the case number, or exit the script
    if [ "" = "$1" ] && [ "0" = "${RESULT}" ]; then
      TESTID=$((TESTID+1))
    else # finish
      break
    fi
  done # TEST

  # remove temporary script (if it exists)
  if [ "" != "${TESTSCRIPT}" ] && [ -e ${TESTSCRIPT} ]; then
    ${RM} ${TESTSCRIPT}
  fi

  # control log
  if [ "0" = "${RESULT}" ]; then
    echo "+++ ------------------------------------------------------------------------------"
    echo "SUCCESS"
  else
    echo "^^^ +++"
    echo "+++ ------------------------------------------------------------------------------"
    echo "FAILURE"
    echo
  fi

  # override result code (alternative outcome)
  if [ "" != "${RESULTCODE}" ]; then
    RESULT=${RESULTCODE}
  fi

  exit ${RESULT}
fi

