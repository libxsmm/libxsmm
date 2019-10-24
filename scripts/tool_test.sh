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
GREP=$(command -v grep)
WGET=$(command -v wget)
GIT=$(command -v git)
SED=$(command -v sed)
CUT=$(command -v cut)
LS=$(command -v ls)
TR=$(command -v tr)
RM=$(command -v rm)
CP=$(command -v cp)

MKTEMP=${HERE}/../.mktmp.sh
RUN_CMD="--session-command"
#RUN_CMD="-c"

if [ "" != "${WGET}" ] && [ "" != "${PIPELINE}" ] && \
   [ "" != "${BUILDKITE_ORGANIZATION_SLUG}" ] && \
   [ "" != "${BUILDKITE_AGENT_ACCESS_TOKEN}" ];
then
  REVSTART=$(${WGET} -qO- \
  https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${PIPELINE}/builds?access_token=${BUILDKITE_AGENT_ACCESS_TOKEN} \
  | ${SED} -n '/ *\"commit\": / {0,/ *\"commit\": / s/ *\"commit\": \"\(..*\)\".*/\1/p}')
fi
if [ "" = "${REVSTART}" ]; then
  REVSTART="HEAD^"
fi

if [ "" != "${MKTEMP}" ] && [ "" != "${MKDIR}" ] && [ "" != "${CHMOD}" ] && \
   [ "" != "${GREP}" ] && [ "" != "${SED}" ] && [ "" != "${LS}" ] && \
   [ "" != "${TR}" ] && [ "" != "${RM}" ] && [ "" != "${CP}" ];
then
  # check if full/unlimited tests are triggered
  if [ "" != "${FULLCI}" ] && [ "0" != "${FULLCI}" ]; then
    LIMIT=0
  fi
  if [ "0" != "${LIMIT}" ] && [ "" != "${GIT}" ] && \
     [ "" != "$(${GIT} log ${REVSTART}...HEAD 2>/dev/null | ${GREP} -e "\[full ci\]")" ];
  then
    LIMIT=0
  fi

  # set the case number
  if [ "" != "$1" ] && [ -e $1 ]; then
    export TESTSETFILE=$1
    if [ "" != "${BASENAME}" ]; then
      export TESTID=$(${BASENAME} ${TESTSETFILE%.*})
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
  source ${HERE}/../.env/buildkite.env ""
  #source ${HERE}/../.env/travis.env ""

  # support yml-files for Travis-CI that depend on TRAVIS_* variables
  if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
    export TRAVIS_BUILD_DIR=${REPOROOT}
  fi
  if [ "" = "${TRAVIS_OS_NAME}" ] && [ "" != "${UNAME}" ]; then
    export TRAVIS_OS_NAME=$(${UNAME})
  fi
  HOST=$(hostname -s 2>/dev/null)

  # setup PARTITIONS for multi-tests
  if [ "" = "${PARTITIONS}" ]; then
    if [ "" != "${PARTITION}" ]; then
      PARTITIONS=${PARTITION}
    else
      PARTITIONS=none
    fi
  fi
  if [ "random" = "${PARTITION}" ]; then
    if [ "random" != "${PARTITIONS}" ]; then
      PARTITIONS=(${PARTITIONS})
      NPARTITIONS=${#PARTITIONS[@]}
      PARTITIONS=${PARTITIONS[RANDOM%NPARTITIONS]}
    else
      PARTITIONS=none
    fi
  fi
  export PARTITIONS
  export PARTITION

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

  if [ "" != "${LIMITRUN}" ] && [ "0" != "${LIMITRUN}" ] && \
     [ "" != "${LIMIT}" ] && [ "0" != "${LIMIT}" ];
  then
    LIMITRUN=$((LIMIT<LIMITRUN?LIMIT:LIMITRUN))
  fi

  # setup batch execution (TEST may be a singular test given by filename)
  if [ "" = "${LAUNCH}" ] && [ "" != "${SRUN}" ] && [ "0" != "${SLURM}" ]; then
    if [ "" != "${BUILDKITE_LABEL}" ]; then
      LABEL=$(echo "${BUILDKITE_LABEL}" | ${TR} -s [:punct:][:space:] - | ${SED} -e "s/^-//" -e "s/-$//")
    fi
    if [ "" != "${LABEL}" ]; then
      SRUN_FLAGS="${SRUN_FLAGS} -J ${LABEL}"
    fi
    if [ "" != "${LIMITRUN}" ] && [ "0" != "${LIMITRUN}" ]; then
      # convert: seconds -> minutes
      SRUN_FLAGS="${SRUN_FLAGS} --time=$((LIMITRUN/60))"
    fi
    umask 007
    # eventually cleanup run-script of terminated/previous sessions
    ${RM} -f ${HERE}/../.tool_??????.sh
    TESTSCRIPT=$(${MKTEMP} ${HERE}/../.tool_XXXXXX.sh)
    ${CHMOD} +rx ${TESTSCRIPT}
    LAUNCH="${SRUN} --ntasks=1 --partition=\${PARTITION} ${SRUN_FLAGS} \
                    --preserve-env --unbuffered ${TESTSCRIPT}"
  elif [ "" != "${SLURMSCRIPT}" ] && [ "0" != "${SLURMSCRIPT}" ]; then
    umask 007
    # eventually cleanup run-script of terminated/previous sessions
    ${RM} -f ${HERE}/../.tool_??????.sh
    TESTSCRIPT=$(${MKTEMP} ${HERE}/../.tool_XXXXXX.sh)
    ${CHMOD} +rx ${TESTSCRIPT}
    LAUNCH="${TESTSCRIPT}"
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
    for SLURMFILE in $(${LS} -1 ${SLURMDIR}); do
    if [[ (-d ${SLURMDIR}) && ("" = "${SLURMSCRIPT}" || "0" = "${SLURMSCRIPT}") ]]; then
      SLURMFILE=${SLURMDIR}/${SLURMFILE}
      TESTID=$(${BASENAME} ${SLURMFILE%.*})
    elif [ -e "${TEST}" ]; then
      SLURMFILE=${TEST}
    fi
    if [ "none" = "${PARTITIONS}" ] && [ "$0" != "${SLURMFILE}" ] && [ -e ${SLURMFILE} ]; then
      PARTITION=$(${SED} -n "s/^#SBATCH[[:space:]][[:space:]]*\(--partition=\|-p\)\(..*\)/\2/p" ${SLURMFILE})
      if [ "" != "${PARTITION}" ]; then PARTITIONS=${PARTITION}; fi
    fi
    if [ "" != "${LIMIT}" ] && [ "0" != "${LIMIT}" ] && \
       [ "" != "$(command -v touch)" ] && \
       [ "" != "$(command -v stat)" ] && \
       [ "" != "$(command -v date)" ];
    then
      NOW=$(date +%s)
      LIMITFILE=$(echo "${LABEL}" | ${SED} -e "s/[^A-Za-z0-9._-]//g")
      if [ "" = "${LIMITFILE}" ]; then
        LIMITFILE=$(echo "${TESTID}" | ${SED} -e "s/[^A-Za-z0-9._-]//g")
      fi
      if [ "" != "${LIMITFILE}" ]; then
        if [ "" != "${PIPELINE}" ]; then LIMITBASE="${PIPELINE}-"; fi
        if [ "" != "${LIMITDIR}" ] && [ -d ${LIMITDIR} ]; then
          LIMITFILE=${LIMITDIR}/${LIMITBASE}${LIMITFILE}
        else
          LIMITFILE=${REPOROOT}/${LIMITBASE}${LIMITFILE}
        fi
      fi
      if [ "" != "${LIMITFILE}" ] && [ -e ${LIMITFILE} ]; then
        OLD=$(stat -c %Y ${LIMITFILE})
      else # ensure build is not skipped
        OLD=${NOW}
        LIMIT=0
      fi
    fi
    if [ "" = "${NOW}" ]; then NOW=0; fi
    if [ "" = "${OLD}" ]; then OLD=0; fi
    if [ "0" != "$((NOW<(OLD+LIMIT)))" ]; then
      echo "================================================================================"
      echo "Skipped ${TESTID} due to LIMIT=${LIMIT} seconds."
      echo "================================================================================"
      continue
    else
      TOUCHFILE=${LIMITFILE}
    fi
    for PARTITION in ${PARTITIONS}; do
    for CONFIG in ${CONFIGS}; do
    # make execution environment locally available (always)
    if [ "" != "${HOST}" ] && [ "none" != "${CONFIG}" ]; then
      if [ -e ${REPOROOT}/.env/${HOST}/${CONFIG}.env ]; then
        source ${REPOROOT}/.env/${HOST}/${CONFIG}.env ""
      else
        echo "WARNING: configuration \"${CONFIG}\" not found!"
      fi
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
        echo "set -eo pipefail" >> ${TESTSCRIPT}
        echo "if [ \"\" = \"\${MAKEJ}\" ]; then MAKEJ=\"-j \$(eval ${HERE}/tool_cpuinfo.sh -nc)\"; fi" >> ${TESTSCRIPT}
        # make execution environment available
        if [ "" != "${HOST}" ] && [ "none" != "${CONFIG}" ] && \
           [ -e ${REPOROOT}/.env/${HOST}/${CONFIG}.env ];
        then
          LICSDIR=$(command -v icc | ${SED} -e "s/\(\/.*intel\)\/.*$/\1/")
          ${MKDIR} -p ${REPOROOT}/licenses
          ${CP} -u /opt/intel/licenses/* ${REPOROOT}/licenses 2>/dev/null
          ${CP} -u ${LICSDIR}/licenses/* ${REPOROOT}/licenses 2>/dev/null
          echo "export INTEL_LICENSE_FILE=${REPOROOT}/licenses" >> ${TESTSCRIPT}
          echo "source ${REPOROOT}/.env/${HOST}/${CONFIG}.env """ >> ${TESTSCRIPT}
        fi
        # record the current test case
        if [ "$0" != "${SLURMFILE}" ] && [ -e ${SLURMFILE} ]; then
          DIR=$(cd $(dirname ${SLURMFILE}); pwd -P)
          if [ -e ${DIR}/../Makefile ]; then
            DIR=${DIR}/..
          fi
          echo "cd ${REPOROOT} && make \${MAKEJ} && cd ${DIR} && make \${MAKEJ}" >> ${TESTSCRIPT}
          echo "RESULT=\$?" >> ${TESTSCRIPT}
          echo "if [ \"0\" != \"\${RESULT}\" ]; then exit \${RESULT}; fi" >> ${TESTSCRIPT}
          # control log
          echo "echo \"--- RUN ${TESTID}\"" >> ${TESTSCRIPT}
          DIRSED=$(echo "${DIR}" | ${SED} "s/\//\\\\\//g")
          ${SED} \
            -e "s/#\!..*/#\!\/bin\/bash\nset -eo pipefail/" -e "s/\.\//${DIRSED}\//" \
            -e "s/^[./]*\([[:print:]][[:print:]]*\/\)*slurm[[:space:]][[:space:]]*//" \
            -e "/^#SBATCH/d" -e "/^[[:space:]]*$/d" \
            ${SLURMFILE} > ${SLURMFILE}.run && ${CHMOD} +rx ${SLURMFILE}.run
          RUNFILE=$(readlink -f ${SLURMFILE}.run)
          if [ "" = "${TOOL_ENABLED}" ] || [ "0" != "${TOOL_ENABLED}" ]; then
            echo -n "${TOOL_COMMAND} ${RUNFILE} ${TOOL_COMMAND_POST}" >> ${TESTSCRIPT}
          else
            echo -n "${RUNFILE}" >> ${TESTSCRIPT}
          fi
          if [ "" != "${LIMITLOG}" ] && [ "0" != "${LIMITLOG}" ] && \
             [ "" != "$(command -v cat)" ] && [ "" != "$(command -v tail)" ];
          then
            echo " | cat -s | tail -n ${LIMITLOG}" >> ${TESTSCRIPT}
          else
            echo >> ${TESTSCRIPT}
          fi
          echo "${RM} -f ${RUNFILE}" >> ${TESTSCRIPT}
        else
          echo "${TEST}" >> ${TESTSCRIPT}
        fi
        echo >> ${TESTSCRIPT}
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
        if [ "" != "${TOUCHFILE}" ]; then
          ${RM} ${TOUCHFILE}
          TOUCHFILE=""
        fi
        break 4
      fi
    done # ENVS
    done # CONFIGS
    done # PARTITIONS
    if [ "" != "${TOUCHFILE}" ]; then
      touch ${TOUCHFILE}
      TOUCHFILE=""
    fi
    done # SLURMFILE

    # increment the case number, or exit the script
    if [ "" = "$1" ] && [ "0" = "${RESULT}" ]; then
      TESTID=$((TESTID+1))
    else # finish
      break
    fi
    # clear captured test
    TEST=""
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

