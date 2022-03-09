#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################
# shellcheck disable=SC1090,SC2129,SC2155,SC2164,SC2178,SC2206,SC2207
set -o pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
BASENAME=$(command -v basename)
MKDIR=$(command -v mkdir)
CHMOD=$(command -v chmod)
UNAME=$(command -v uname)
DIFF=$(command -v diff)
# flush asynchronous NFS mount
SYNC=$(command -v sync)
GREP=$(command -v grep)
WGET=$(command -v wget)
GIT=$(command -v git)
SED=$(command -v gsed)
CUT=$(command -v cut)
LS=$(command -v ls)
TR=$(command -v tr)
RM=$(command -v rm)
CP=$(command -v cp)

MKTEMP=${HERE}/../.mktmp.sh
RUN_CMD="--session-command"
#RUN_CMD="-c"

# GNU sed is desired (macOS)
if [ "" = "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${WGET}" ] && [ "${SED}" ] && [ "${PIPELINE}" ] && \
   [ "${BUILDKITE_ORGANIZATION_SLUG}" ] && \
   [ "${BUILDKITE_AGENT_ACCESS_TOKEN}" ];
then
  REVSTART=$(${WGET} -qO- \
  "https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${PIPELINE}/builds?access_token=${BUILDKITE_AGENT_ACCESS_TOKEN}" \
  | ${SED} -n '/ *\"commit\": / {0,/ *\"commit\": / s/ *\"commit\": \"\(..*\)\".*/\1/p}')
fi
if [ "" = "${REVSTART}" ]; then
  REVSTART="HEAD^"
fi

if [ "${MKTEMP}" ] && [ "${MKDIR}" ] && [ "${CHMOD}" ] && \
   [ "${DIFF}" ] && [ "${GREP}" ] && [ "${SED}" ] && \
   [ "${LS}" ] && [ "${TR}" ] && \
   [ "${RM}" ] && [ "${CP}" ];
then
  # check if full/unlimited tests are triggered
  if [ "${FULLCI}" ] && [ "0" != "${FULLCI}" ]; then
    LIMIT=0
  fi
  if [ "0" != "${LIMIT}" ] && [ "${GIT}" ] && \
     [ "$(${GIT} log ${REVSTART}...HEAD 2>/dev/null | ${GREP} -e "\[full ci\]")" ];
  then
    LIMIT=0
  fi

  # set the case number
  if [ "$1" ] && [ -e "$1" ]; then
    export TESTSETFILE=$1
    if [ "${BASENAME}" ]; then
      export TESTID=$(${BASENAME} ${TESTSETFILE%.*})
    else
      export TESTID=${TESTSETFILE}
    fi
    export TESTSET=${TESTID}
  else # case number given
    if [ "$1" ] && [ "0" != "$1" ]; then
      export TESTID=$1
    else
      export TESTID=1
    fi
  fi

  # should be source'd after the above variables are set
  source "${HERE}/../.env/buildkite.env" ""
  #source "${HERE}/../.env/travis.env" ""

  # support yml-files for Travis-CI that depend on TRAVIS_* variables
  if [ "" = "${TRAVIS_BUILD_DIR}" ]; then
    export TRAVIS_BUILD_DIR=${REPOROOT}
  fi
  if [ "" = "${TRAVIS_OS_NAME}" ] && [ "${UNAME}" ]; then
    export TRAVIS_OS_NAME=$(${UNAME})
  fi
  if [ "" = "${HOSTNAME}" ]; then
    HOSTNAME=$(hostname -s 2>/dev/null)
  fi

  # setup PARTITIONS for multi-tests
  if [ "" = "${PARTITIONS}" ]; then
    if [ "${PARTITION}" ]; then
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

  # setup CONFIGS (multiple configurations)
  if [ "" = "${CONFIGS}" ]; then
    if [ "${CONFIG}" ]; then
      CONFIGS=${CONFIG}
    else
      CONFIGS=none
    fi
  elif [ "${CONFIG}" ]; then
    # singular CONFIG replaces set of CONFIGS
    CONFIGS=${CONFIG}
  fi
  # setup ENVS (multiple environments)
  if [ "" = "${ENVS}" ]; then
    if [ "${ENV}" ]; then
      ENVS=${ENV}
    else
      ENVS=none
    fi
  fi

  # select test-set ("travis" by default)
  if [ "" = "${TESTSET}" ]; then
    TESTSET=travis
  fi
  if [ "" = "${TESTSETFILE}" ] || [ ! -e "${TESTSETFILE}" ]; then
    if [ -e ".${TESTSET}.yml" ]; then
      TESTSETFILE=.${TESTSET}.yml
    elif [ -e "${TESTSET}.yml" ]; then
      TESTSETFILE=${TESTSET}.yml
    elif [ -e "${TESTSET}" ]; then
      TESTSETFILE=${TESTSET}
    else
      echo "ERROR: Cannot find file with test set!"
      exit 1
    fi
  else
    TEST=${TESTSETFILE}
  fi

  if [ "${LIMITRUN}" ] && [ "0" != "${LIMITRUN}" ] && \
     [ "${LIMIT}" ] && [ "0" != "${LIMIT}" ];
  then
    LIMITRUN=$((LIMIT<LIMITRUN?LIMIT:LIMITRUN))
  fi

  # eventually cleanup run-script of terminated/previous sessions
  ${RM} -f "${HERE}"/../.tool_??????.sh
  # setup batch execution (TEST may be a singular test given by filename)
  if [ "" = "${LAUNCH_CMD}" ] && [ "" = "${LAUNCH}" ] && [ "${SRUN}" ] && [ "0" != "${SLURM}" ]; then
    if [ "${BUILDKITE_LABEL}" ]; then
      LABEL=$(echo "${BUILDKITE_LABEL}" | ${TR} -s [[:punct:][:space:]] - | ${SED} -e "s/^-//" -e "s/-$//")
    fi
    if [ "${LABEL}" ]; then
      SRUN_FLAGS="${SRUN_FLAGS} -J ${LABEL}"
    fi
    if [ "${LIMITRUN}" ] && [ "0" != "${LIMITRUN}" ]; then
      # convert: seconds -> minutes
      SRUN_FLAGS="${SRUN_FLAGS} --time=$((LIMITRUN/60))"
    fi
    #SRUN_FLAGS="${SRUN_FLAGS} --preserve-env"
    umask 007
    TESTSCRIPT=$(${MKTEMP} "${HERE}/../.tool_XXXXXX.sh")
    ${CHMOD} +rx "${TESTSCRIPT}"
    LAUNCH="${SRUN} --ntasks=1 --partition=\${PARTITION} ${SRUN_FLAGS} \
                    --unbuffered ${TESTSCRIPT}"
  elif [[ ("${LAUNCH_CMD}") || (-d "$1") || ("${SLURMSCRIPT}" && "0" != "${SLURMSCRIPT}") ]]; then
    umask 007
    TESTSCRIPT=$(${MKTEMP} "${HERE}/../.tool_XXXXXX.sh")
    ${CHMOD} +rx "${TESTSCRIPT}"
    LAUNCH="${LAUNCH_CMD} ${TESTSCRIPT}"
  else # avoid temporary script in case of non-batch execution
    if [ "" = "${MAKEJ}" ]; then
      export MAKEJ="-j $(eval "${HERE}/tool_cpuinfo.sh" -nc)"
    fi
    SHOW_PARTITION=0
    LAUNCH="\${TEST}"
  fi
  if [ "${LAUNCH_USER}" ] && [ "0" != "${SLURM}" ]; then
    LAUNCH="su ${LAUNCH_USER} -p ${RUN_CMD} \'${LAUNCH}\'"
  fi

  # backup current environment (snapshot)
  ${RM} -f "${HERE}"/../.env_??????
  ENVFILE=$(${MKTEMP} "${HERE}/../.env_XXXXXX")
  ${CHMOD} +r "${ENVFILE}"
  declare -px > "${ENVFILE}"

  RESULT=0
  # control log
  echo && echo "^^^ +++"
  while [ "${TEST}" ] || TEST=$(eval " \
    ${SED} -n -e '/^ *script: *$/,\$p' ${HERE}/../${TESTSETFILE} | ${SED} -e '/^ *script: *$/d' | \
    ${SED} -n -E \"/^ *- */H;//,/^ *$/G;s/\n(\n[^\n]*){\${TESTID}}$//p\" | \
    ${SED} -e 's/^ *- *//' -e 's/^  *//' | ${TR} '\n' ' ' | \
    ${SED} -e 's/  *$//'") && [ "${TEST}" ];
  do
    if [ -d "${TEST}" ]; then
      SLURMDIR=${TEST}
    else # dummy
      SLURMDIR=$0
    fi
    for SLURMFILE in $(${LS} -1 "${SLURMDIR}"); do
    if [[ (-d ${SLURMDIR}) && ("" = "${SLURMSCRIPT}" || "0" = "${SLURMSCRIPT}") ]]; then
      SLURMFILE=${SLURMDIR}/${SLURMFILE}
      TESTID=$(${BASENAME} ${SLURMFILE%.*})
    elif [ -e "${TEST}" ]; then
      SLURMFILE=${TEST}
    fi
    if [ "none" = "${PARTITIONS}" ] && [ "$0" != "${SLURMFILE}" ] && [ -e "${SLURMFILE}" ]; then
      PARTITION=$(${SED} -n "s/^#SBATCH[[:space:]][[:space:]]*\(--partition=\|-p\)\(..*\)/\2/p" "${SLURMFILE}")
      if [ "${PARTITION}" ]; then PARTITIONS=${PARTITION}; fi
    fi
    if [ "${LIMIT}" ] && [ "0" != "${LIMIT}" ] && \
       [ "$(command -v stat)" ] && \
       [ "$(command -v date)" ];
    then
      NOW=$(date +%s)
      LIMITFILE=$(echo "${LABEL}" | ${SED} -e "s/[^A-Za-z0-9._-]//g")
      if [ "" = "${LIMITFILE}" ]; then
        LIMITFILE=$(echo "${TESTID}" | ${SED} -e "s/[^A-Za-z0-9._-]//g")
      fi
      if [ "${LIMITFILE}" ]; then
        if [ "${PIPELINE}" ]; then LIMITBASE="${PIPELINE}-"; fi
        if [ "${LIMITDIR}" ] && [ -d "${LIMITDIR}" ]; then
          LIMITFILE=${LIMITDIR}/${LIMITBASE}${LIMITFILE}
        else
          LIMITFILE=${REPOROOT}/${LIMITBASE}${LIMITFILE}
        fi
      fi
      if [ "${LIMITFILE}" ] && [ -e "${LIMITFILE}" ]; then
        OLD=$(stat -c %Y "${LIMITFILE}")
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
    CONFIGFILE=""
    if [ "${HOSTNAME}" ] && [ "none" != "${CONFIG}" ]; then
      CONFIGPAT=$(echo "${CONFIGEX}" | ${SED} "s/[[:space:]][[:space:]]*/\\\|/g" | ${SED} "s/\\\|$//")
      if [ "${CONFIGPAT}" ]; then
        CONFIGFILES=($(bash -c "${LS} -1 ${REPOROOT}/.env/${HOSTNAME}/${CONFIG}.env 2>/dev/null" | ${SED} "/\(${CONFIGPAT}\)/d"))
      else
        CONFIGFILES=($(bash -c "${LS} -1 ${REPOROOT}/.env/${HOSTNAME}/${CONFIG}.env 2>/dev/null"))
      fi
      CONFIGCOUNT=${#CONFIGFILES[@]}
      if [ "0" != "${CONFIGCOUNT}" ]; then
        CONFIGFILE=${CONFIGFILES[RANDOM%CONFIGCOUNT]}
        CONFIG=$(${BASENAME} "${CONFIGFILE}" .env)
      else
        echo "WARNING: configuration \"${CONFIG}\" not found!"
        CONFIGFILE=""
      fi
    fi
    for ENV in ${ENVS}; do
      if [ "none" != "${ENV}" ]; then
        if [ "${CUT}" ]; then ENVVAL=$(echo "${ENV}" | ${CUT} -d= -f2); fi
        ENVSTR=${ENV}
      fi
      # print some header if all tests are selected or in case of multi-tests
      if [[ "none" != "${CONFIG}" && ("" = "$1" || "none" != "${PARTITION}" || "none" != "${ENV}") ]]; then
        if [ "none" != "${PARTITION}" ] && [ "0" != "${SHOW_PARTITION}" ]; then
          if [ "${ENVVAL}" ]; then
            echo "+++ TEST ${TESTID} (${PARTITION}/${CONFIG}/${ENVVAL})"
          else
            echo "+++ TEST ${TESTID} (${PARTITION}/${CONFIG})"
          fi
        elif [ "${ENVVAL}" ]; then
          echo "+++ TEST ${TESTID} (${CONFIG}/${ENVVAL})"
        else
          echo "+++ TEST ${TESTID} (${CONFIG})"
        fi
      fi
      # prepare temporary script for remote environment/execution
      if [ "${TESTSCRIPT}" ] && [ -e "${TESTSCRIPT}" ]; then
        echo "#!/usr/bin/env bash" > "${TESTSCRIPT}"
        echo "set -eo pipefail" >> "${TESTSCRIPT}"
        echo "cd ${REPOROOT}" >> "${TESTSCRIPT}"
        echo "if [ \"\$(command -v sync)\" ]; then sync; fi" >> "${TESTSCRIPT}"
        if [ "0" != "${SHOW_PARTITION}" ]; then echo "echo \"-> \${USER}@\${HOSTNAME} (\${PWD})\"" >> "${TESTSCRIPT}"; fi
        echo "if [ \"\" = \"\${MAKEJ}\" ]; then MAKEJ=\"-j \$(eval ${HERE}/tool_cpuinfo.sh -nc)\"; fi" >> "${TESTSCRIPT}"
        # make execution environment available
        if [ "" = "${INTEL_LICENSE_FILE}" ]; then
          LICSDIR=$(command -v icc | ${SED} -e "s/\(\/.*intel\)\/.*$/\1/")
          ${MKDIR} -p "${REPOROOT}/licenses"
          ${CP} -u "${HOME}"/intel/licenses/* "${REPOROOT}/licenses" 2>/dev/null
          ${CP} -u "${LICSDIR}"/licenses/* "${REPOROOT}/licenses" 2>/dev/null
          ${CP} -u /opt/intel/licenses/* "${REPOROOT}/licenses" 2>/dev/null
          echo "export INTEL_LICENSE_FILE=${REPOROOT}/licenses" >> "${TESTSCRIPT}"
        fi
        # setup environment on a per-test basis
        echo "if [ -e \"${ENVFILE}\" ]; then" >> "${TESTSCRIPT}"
        if [ "${LAUNCH_CMD}" ]; then
          echo "  eval ${HERE}/tool_envrestore.sh \"${ENVFILE}\" \"${HERE}/../.env.sh\"" >> "${TESTSCRIPT}"
          echo "  source \"${HERE}/../.env.sh\"" >> "${TESTSCRIPT}"
        else
          echo "  eval ${HERE}/tool_envrestore.sh \"${ENVFILE}\"" >> "${TESTSCRIPT}"
        fi
        echo "fi" >> "${TESTSCRIPT}"
        if [ -e "${CONFIGFILE}" ]; then
          echo "  source \"${CONFIGFILE}\" \"\"" >> "${TESTSCRIPT}"
        fi
        # record the current test case
        if [ "$0" != "${SLURMFILE}" ] && [ -e "${SLURMFILE}" ]; then
          ABSDIR=$(dirname "${SLURMFILE}")
          if [ ! -e "${ABSDIR}/Makefile" ] && [ -d "${ABSDIR}" ] && [ -e "${ABSDIR}/../Makefile" ]; then
            ABSDIR=${ABSDIR}/..
          fi
          ABSDIR=$(cd "${ABSDIR}"; pwd -P)
          echo "cd ${REPOROOT} && make -e \${MAKEJ} && cd ${ABSDIR} && make -e \${MAKEJ}" >> "${TESTSCRIPT}"
          echo "RESULT=\$?" >> "${TESTSCRIPT}"
          echo "if [ \"0\" != \"\${RESULT}\" ]; then exit \${RESULT}; fi" >> "${TESTSCRIPT}"
          # control log
          echo "echo \"--- RUN ${TESTID}\"" >> "${TESTSCRIPT}"
          DIRSED=$(echo "${ABSDIR}" | ${SED} "s/\//\\\\\//g")
          ${SED} \
            -e "s/#\!..*/#\!\/bin\/bash\nset -eo pipefail/" -e "s/\(^\|[[:space:]]\)\(\.\|\.\.\)\//\1${DIRSED}\/\2\//" \
            -e "s/^[./]*\([[:print:]][[:print:]]*\/\)*slurm[[:space:]][[:space:]]*//" \
            -e "/^#SBATCH/d" -e "/^[[:space:]]*$/d" \
            "${SLURMFILE}" > "${SLURMFILE}.run" && ${CHMOD} +rx "${SLURMFILE}.run"
          RUNFILE=$(readlink -f "${SLURMFILE}.run")
          if [ "${TOOL_COMMAND}" ]; then
            if [ "0" = "${TOOL_INJECT}" ] || [ "" = "$(${SED} -n "/^taskset/p" "${RUNFILE}")" ]; then
              echo -n "${TOOL_COMMAND} ${RUNFILE} ${TOOL_COMMAND_POST}" >> "${TESTSCRIPT}"
            else # inject TOOL_COMMAND
              TOOL_COMMAND_SED1="$(echo "${TOOL_COMMAND}" | ${SED} "s/\//\\\\\//g") "
              if [ "${TOOL_COMMAND_POST}" ]; then
                TOOL_COMMAND_SED2=" $(echo "${TOOL_COMMAND_POST}" | ${SED} "s/\//\\\\\//g")"
              fi
              ${SED} -i "s/\(^taskset[[:space:]]..*\)/${TOOL_COMMAND_SED1}\1${TOOL_COMMAND_SED2}/" "${RUNFILE}"
              echo -n "${RUNFILE}" >> "${TESTSCRIPT}"
            fi
          else
            echo -n "${RUNFILE}" >> "${TESTSCRIPT}"
          fi
          if [ "${LIMITLOG}" ] && [ "0" != "${LIMITLOG}" ] && \
             [ "$(command -v cat)" ] && [ "$(command -v tail)" ];
          then
            echo " | cat -s | tail -n ${LIMITLOG}" >> "${TESTSCRIPT}"
          elif [ "0" = "${LIMITLOG}" ]; then
            echo " >/dev/null" >> "${TESTSCRIPT}"
          else
            echo >> "${TESTSCRIPT}"
          fi
          echo "${RM} -f ${RUNFILE}" >> "${TESTSCRIPT}"
        else
          echo "${TEST}" >> "${TESTSCRIPT}"
        fi
        echo >> "${TESTSCRIPT}"
        if [ "${SYNC}" ]; then ${SYNC}; fi
      else
        # setup environment on a per-test basis
        if [ "${CONFIGFILE}" ]; then
          if [ -e "${ENVFILE}" ]; then
            eval "${HERE}/tool_envrestore.sh" "${ENVFILE}"
          fi
          source "${CONFIGFILE}" ""
        fi
      fi

      COMMAND=$(eval echo "${ENVSTR} ${LAUNCH}")
      # run the prepared test case/script
      if [ "${LABEL}" ] && [ "$(command -v tee)" ]; then
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
      if [ "0" != "${RESULT}" ] && [ "1" != "${LIMITHARD}" ]; then
        if [ "${TOUCHFILE}" ]; then
          ${RM} -f "${TOUCHFILE}"
          TOUCHFILE=""
        fi
        break 4
      fi
    done # ENVS
    done # CONFIGS
    done # PARTITIONS
    if [ "${TOUCHFILE}" ]; then
      echo "${JOBID}" > "${TOUCHFILE}"
      TOUCHFILE=""
    fi
    done # SLURMFILE

    # increment the case number, or exit the script
    if [ "0" = "$1" ] && [ "0" = "${RESULT}" ]; then
      TESTID=$((TESTID+1))
    else # finish
      break
    fi
    # clear captured test
    TEST=""
  done # TEST

  # remove temporary files
  if [ "${TESTSCRIPT}" ] && [ -e "${TESTSCRIPT}" ]; then
    ${RM} "${TESTSCRIPT}"
  fi
  if [ "${ENVFILE}" ] && [ -e "${ENVFILE}" ]; then
    ${RM} "${ENVFILE}"
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
  if [ "${RESULTCODE}" ]; then
    RESULT=${RESULTCODE}
  fi

  exit "${RESULT}"
fi
