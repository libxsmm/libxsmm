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
# shellcheck disable=SC1090,SC2129,SC2153,SC2207
set -o pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
ROOT=${HERE}/..
# TODO: map to CI-provider (abstract environment)
source "${ROOT}/.env/buildkite.env" ""

MKTEMP=${ROOT}/.mktmp.sh
MKDIR=$(command -v mkdir)
DIFF=$(command -v diff)
# flush asynchronous NFS mount
SYNC=$(command -v sync)
GREP=$(command -v grep)
SED=$(command -v gsed)
GIT=$(command -v git)
TR=$(command -v tr)

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

RUN_CMD="--session-command"
#RUN_CMD="-c"

if [ "${MKTEMP}" ] && [ "${MKDIR}" ] && [ "${DIFF}" ] && [ "${GREP}" ] && [ "${SED}" ] && [ "${TR}" ]; then
  DIRPAT="s/\//\\\\\//g"
  REMPAT=$(echo "${REPOREMOTE}" | ${SED} "${DIRPAT}")
  REPPAT=$(echo "${REPOROOT}" | ${SED} "${DIRPAT}")
  # ensure proper permissions
  if [ "${UMASK}" ]; then
    UMASK_CMD="umask ${UMASK};"
    eval "${UMASK_CMD}"
  fi

  # check if full/unlimited tests are triggered
  if [ "${FULLCI}" ] && [ "0" != "${FULLCI}" ]; then
    LIMIT=0
  elif [ ! "${LAUNCH_CMD}" ]; then
    if [ ! "${REVSTART}" ]; then
      REVSTART="HEAD^"
    fi
    if [ "0" != "${LIMIT}" ] && [ "${GIT}" ] && \
       [ "$(${GIT} log ${REVSTART}...HEAD 2>/dev/null | ${GREP} "\[full ci\]")" ];
    then
      LIMIT=0
    fi
  fi

  # set the case number
  if [ "$1" ] && [ -e "$1" ]; then
    TESTID=$(basename "${TESTSETFILE%.*}")
    export TESTSETFILE=$1 TESTSET=${TESTID}
  else # case number given
    if [ "$1" ] && [ "0" != "$1" ]; then
      TESTID=$1
    else
      TESTID=1
    fi
  fi
  export TESTID

  # support yml-files for Travis-CI that depend on TRAVIS_* variables
  if [ ! "${TRAVIS_BUILD_DIR}" ]; then
    export TRAVIS_BUILD_DIR=${REPOREMOTE}
  fi
  if [ ! "${TRAVIS_OS_NAME}" ]; then
    if [ "${LAUNCH_CMD}" ]; then
      TRAVIS_OS_NAME=$(${LAUNCH_CMD} "uname")
    else
      TRAVIS_OS_NAME=$(uname)
    fi
    export TRAVIS_OS_NAME
  fi
  if [ "${HOSTNAME}" ]; then
    HOSTNAME=$(echo "${HOSTNAME}" | cut -d. -f1 2>/dev/null)
  fi
  if [ ! "${HOSTNAME}" ]; then
    HOSTNAME=$(hostname -s 2>/dev/null)
  fi
  HOSTDELIMCHAR="-"
  HOSTPREFIX=$(echo "${HOSTNAME}" | cut -d${HOSTDELIMCHAR} -f1 2>/dev/null)
  if [ "${HOSTPREFIX}" ]; then
    HOSTPREFIX="${HOSTPREFIX}${HOSTDELIMCHAR}"
  fi

  # setup PARTITIONS for multi-tests
  if [ ! "${PARTITIONS}" ]; then
    if [ "${PARTITION}" ]; then
      PARTITIONS=${PARTITION}
    else
      PARTITIONS=none
    fi
  fi
  if [ "random" = "${PARTITION}" ]; then
    if [ "random" != "${PARTITIONS}" ]; then
      read -ra ARRAY <<<"${PARTITIONS}"
      NPARTITIONS=${#ARRAY[@]}
      PARTITIONS=${ARRAY[RANDOM%NPARTITIONS]}
    else
      PARTITIONS=none
    fi
  fi
  export PARTITIONS
  read -ra ARRAY <<<"${PARTITIONS}"
  NPARTITIONS=${#ARRAY[@]}

  if [ "${LIBXSMM_TARGETS}" ] && [ ! "${LIBXSMM_TARGET}" ]; then
    read -ra ARRAY <<<"${LIBXSMM_TARGETS}"
    NTARGETS=${#ARRAY[@]}
    export LIBXSMM_TARGET=${ARRAY[RANDOM%NTARGETS]}
  fi

  # setup CONFIGS (multiple configurations)
  # singular CONFIG takes precedence
  if [ "${CONFIG}" ]; then
    CONFIGS=${CONFIG}
  elif [ ! "${CONFIGS}" ]; then
    if [ "${CONFIG}" ]; then
      CONFIGS=${CONFIG}
    else
      CONFIGS=none
    fi
  fi
  read -ra ARRAY <<<"${CONFIGS}"
  NCONFIGS=${#ARRAY[@]}

  # setup ENVS (multiple environments)
  if [ ! "${ENVS}" ]; then
    if [ "${ENV}" ]; then
      ENVS=${ENV}
    else
      ENVS=none
    fi
  fi
  read -ra ARRAY <<<"${ENVS}"
  NENVS=${#ARRAY[@]}

  # select test-set ("travis" by default)
  if [ ! "${TESTSET}" ]; then
    TESTSET=travis
  fi
  if [ ! "${TESTSETFILE}" ] || [ ! -e "${TESTSETFILE}" ]; then
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

  CPUINFO=${HERE}/tool_cpuinfo.sh
  # eventually cleanup run-script of terminated/previous sessions
  rm -f "${REPOROOT}"/.tool_??????.sh
  # setup batch execution (TEST may be a singular test given by filename)
  if [ ! "${LAUNCH_CMD}" ] && [ ! "${LAUNCH}" ] && [ "${SRUN}" ] && [ "0" != "${SLURM}" ]; then
    STEPNAME=${STEPNAME:-${BUILDKITE_LABEL}}
    if [ "${STEPNAME}" ]; then
      LABEL=$(echo "${STEPNAME}" \
        | ${TR} -s "[:punct:][:space:]" "-" \
        | ${SED} "s/^-//;s/-$//" \
        | ${SED} "s/[^A-Za-z0-9._-]//g")
    fi
    if [ "${LABEL}" ]; then
      SRUN_FLAGS="${SRUN_FLAGS} -J ${LABEL}"
    fi
    if [ "${LIMITRUN}" ] && [ "0" != "${LIMITRUN}" ]; then
      # convert: seconds -> minutes
      SRUN_FLAGS="${SRUN_FLAGS} --time=$((LIMITRUN/60))"
    fi
    #SRUN_FLAGS="${SRUN_FLAGS} --preserve-env"
    TESTSCRIPT=$(${MKTEMP} "${REPOROOT}/.tool_XXXXXX.sh")
    chmod +rx "${TESTSCRIPT}"
    LAUNCH="${SRUN} --ntasks=1 --partition=\${PARTITION} ${SRUN_FLAGS} \
                    --unbuffered ${TESTSCRIPT} ${*:2}"
  elif [[ ("${LAUNCH_CMD}") || (-d "$1") || ("${SLURMSCRIPT}" && "0" != "${SLURMSCRIPT}") ]]; then
    TESTSCRIPT=$(${MKTEMP} "${REPOROOT}/.tool_XXXXXX.sh")
    REMSCRIPT=$(echo "${TESTSCRIPT}" | ${SED} "s/${REPPAT}/${REMPAT}/")
    chmod +rx "${TESTSCRIPT}"
    LAUNCH="${LAUNCH_CMD} ${REMSCRIPT} ${*:2}"
  else # avoid temporary script in case of non-batch execution
    if [ ! "${MAKEJ}" ]; then
      MAKEJ="-j $(eval "${CPUINFO}" -nc)"
      export MAKEJ
    fi
    SHOW_PARTITION=0
    LAUNCH="\${TEST}"
  fi
  if [ "${LAUNCH_USER}" ] && [ "0" != "${SLURM}" ]; then
    # avoid preserving environment (wrong HOME)
    LAUNCH="su ${LAUNCH_USER} ${RUN_CMD} \'${LAUNCH}\'"
  fi

  # eventually cleanup environment snapshots
  rm -f "${REPOROOT}"/.env_??????
  # backup current environment (snapshot)
  ENVFILE=$(${MKTEMP} "${REPOROOT}/.env_XXXXXX")
  chmod +r "${ENVFILE}"
  declare -px >"${ENVFILE}"

  if [[ "${UMASK}" && (! "${TESTSCRIPT}" || ! -e "${TESTSCRIPT}") ]]; then
    # TODO: derive permissions from UMASK
    trap 'rm ${TESTSCRIPT} ${ENVFILE} && (chmod -Rf g+u,o=u-w ${REPOROOT} || true)' EXIT
  else
    trap 'rm ${TESTSCRIPT} ${ENVFILE}' EXIT
  fi

  RESULT=0
  LOGFILE_INIT=${LOGFILE}
  while [ "${TEST}" ] || TEST=$(eval " \
    ${SED} -n '/^ *script: *$/,\$p' ${REPOROOT}/${TESTSETFILE} | ${SED} '/^ *script: *$/d' | \
    ${SED} -n -E \"/^ *- */H;//,/^ *$/G;s/\n(\n[^\n]*){\${TESTID}}$//p\" | \
    ${SED} 's/^ *- *//;s/^  *//' | ${TR} '\n' ' ' | \
    ${SED} 's/  *$//'") && [ "${TEST}" ];
  do
    if [ -d "${TEST}" ]; then
      SLURMDIR=${TEST}
    else # dummy
      SLURMDIR=$0
    fi
    for SLURMFILE in "${SLURMDIR}"/*; do
    if [[ (-d ${SLURMDIR}) && (! "${SLURMSCRIPT}" || "0" = "${SLURMSCRIPT}") ]]; then
      SLURMFILE=${SLURMDIR}/${SLURMFILE}
      TESTID=$(basename "${SLURMFILE%.*}")
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
      LIMITFILE=$(echo "${LABEL}" | ${TR} "[:upper:]" "[:lower:]")
      if [ ! "${LIMITFILE}" ]; then
        LIMITFILE=$(echo "${TESTID}" | ${SED} "s/[^A-Za-z0-9._-]//g")
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
    if [ ! "${NOW}" ]; then NOW=0; fi
    if [ ! "${OLD}" ]; then OLD=0; fi
    if [ "0" != "$((NOW<(OLD+LIMIT)))" ]; then
      echo "================================================================================"
      echo "Skipped ${TESTID} due to LIMIT=${LIMIT} seconds."
      echo "================================================================================"
      continue
    else
      TOUCHFILE=${LIMITFILE}
    fi
    COUNT_PRT=0; for PARTITION in ${PARTITIONS}; do
    COUNT_CFG=0; for CONFIG in ${CONFIGS}; do
    # make execution environment locally available (always)
    CONFIGFILE=""
    if [[ ("none" != "${CONFIG}") && ("${HOSTNAME}" || "${HOSTPREFIX}") ]]; then
      CONFIGFILES=($(ls -1 ${ROOT}/.env/${HOSTNAME}/${CONFIG}.env 2>/dev/null))
      if [[ ! "${CONFIGFILES[@]}" ]]; then
        CONFIGFILES=($(ls -1 ${ROOT}/.env/${HOSTPREFIX}*/${CONFIG}.env 2>/dev/null))
      fi
      if [[ "${CONFIGFILES[@]}" ]]; then
        CONFIGPAT=$(echo "${CONFIGEX}" | ${SED} "s/[[:space:]][[:space:]]*/\\\|/g" | ${SED} "s/\\\|$//")
        if [ "${CONFIGPAT}" ]; then
          CONFIGFILES=($(echo "${CONFIGFILES}" | ${SED} "/\(${CONFIGPAT}\)/d"))
        fi
        CONFIGCOUNT=${#CONFIGFILES[@]}
        if [ "0" != "${CONFIGCOUNT}" ]; then
          CONFIGFILE=${CONFIGFILES[RANDOM%CONFIGCOUNT]}
          CONFIG=$(basename "${CONFIGFILE}" .env)
        else
          echo "WARNING: configuration \"${CONFIG}\" not found!"
          CONFIGFILE=""
        fi
      fi
    fi
    COUNT_ENV=0; for ENV in ${ENVS}; do
      if [ "none" != "${ENV}" ]; then
        ENVVAL=$(echo "${ENV}" | cut -d= -f2)
        ENVSTR=${ENV}
      fi
      # print some header if all tests are selected or in case of multi-tests
      HEADER=""
      if [ "none" != "${PARTITION}" ] && [ "0" != "${SHOW_PARTITION}" ]; then HEADER="${PARTITION}"; fi
      if [ "none" != "${CONFIG}" ]; then HEADER="${HEADER} ${CONFIG}"; fi
      if [ "${ENVVAL}" ]; then HEADER="${HEADER} ${ENV}"; fi
      HEADER=$(echo "${HEADER}" \
        | ${SED} "s/^[[:space:]][[:space:]]*//;s/[[:space:]][[:space:]]*$//" \
        | ${TR} "[:lower:]" "[:upper:]" | ${TR} -s " " "/")
      if [ "${TESTID}" ] && [ "test" != "$(echo "${TESTID}" | ${TR} "[:upper:]" "[:lower:]")" ]; then
        if [ "${HEADER}" ]; then
          CAPTION="${TESTID} (${HEADER})"
        else
          CAPTION="${TESTID}"
        fi
      else
        CAPTION="${HEADER}"
      fi
      echo "--- TEST ${CAPTION}"
      # prepare temporary script for remote environment/execution
      if [ "${TESTSCRIPT}" ] && [ -e "${TESTSCRIPT}" ]; then
        echo "#!/usr/bin/env bash" >"${TESTSCRIPT}"
        echo "set -eo pipefail" >>"${TESTSCRIPT}"
        # exact/real name of run-file is not known yet
        EXIT_TRAP="rm -f ${REPOREMOTE}/.env.sh ${REPOREMOTE}/*.run"
        if [ "${UMASK}" ]; then # TODO: derive permissions from UMASK
          EXIT_TRAP="(${EXIT_TRAP}); (chmod -Rf g+u,o=u-w ${REPOREMOTE} || true)"
        fi
        echo "trap '${EXIT_TRAP}' EXIT" >>"${TESTSCRIPT}"
        echo "${UMASK_CMD}" >>"${TESTSCRIPT}"
        echo "cd ${REPOREMOTE}" >>"${TESTSCRIPT}"
        echo "if [ \"\$(command -v sync)\" ]; then sync; fi" >>"${TESTSCRIPT}"
        if [ "0" != "${SHOW_PARTITION}" ]; then echo "echo \"-> \${USER}@\${HOSTNAME} (\${PWD})\"" >>"${TESTSCRIPT}"; fi
        REMINFO=$(echo "${CPUINFO}" | ${SED} "s/${REPPAT}/${REMPAT}/")
        echo "if [ \"\" = \"\${MAKEJ}\" ]; then MAKEJ=\"-j \$(eval ${REMINFO} -nc)\"; fi" >>"${TESTSCRIPT}"
        # make execution environment available
        if [ ! "${INTEL_LICENSE_FILE}" ]; then
          LICSDIR=$(command -v icc | ${SED} "s/\(\/.*intel\)\/.*$/\1/")
          ${MKDIR} -p "${REPOROOT}/licenses"
          cp -u "${HOME}"/intel/licenses/* "${REPOROOT}/licenses" 2>/dev/null
          cp -u "${LICSDIR}"/licenses/* "${REPOROOT}/licenses" 2>/dev/null
          cp -u /opt/intel/licenses/* "${REPOROOT}/licenses" 2>/dev/null
          echo "export INTEL_LICENSE_FILE=${REPOREMOTE}/licenses" >>"${TESTSCRIPT}"
        fi
        # setup environment on a per-test basis
        ENVREM=$(echo "${ENVFILE}" | ${SED} "s/${REPPAT}/${REMPAT}/")
        ENVRST=$(echo "${HERE}/tool_envrestore.sh" | ${SED} "s/${REPPAT}/${REMPAT}/")
        echo "if [ -e \"${ENVREM}\" ]; then" >>"${TESTSCRIPT}"
        if [ "${LAUNCH_CMD}" ]; then
          echo "  eval ${ENVRST} \"${ENVREM}\" \"${REPOREMOTE}/.env.sh\"" >>"${TESTSCRIPT}"
          echo "  source \"${REPOREMOTE}/.env.sh\"" >>"${TESTSCRIPT}"
        else
          echo "  eval ${ENVRST} \"${ENVREM}\"" >>"${TESTSCRIPT}"
        fi
        echo "fi" >>"${TESTSCRIPT}"
        if [ -e "${CONFIGFILE}" ]; then
          echo "  source \"$(echo "${CONFIGFILE}" | ${SED} "s/${REPPAT}/${REMPAT}/")\" \"\"" >>"${TESTSCRIPT}"
        fi
        # record the current test case
        if [ "$0" != "${SLURMFILE}" ] && [ -e "${SLURMFILE}" ]; then
          ABSDIR=$(dirname "${SLURMFILE}")
          if [ ! -e "${ABSDIR}/Makefile" ] && [ -d "${ABSDIR}" ] && [ -e "${ABSDIR}/../Makefile" ]; then
            ABSDIR=${ABSDIR}/..
          fi
          ABSDIR=$(cd "${ABSDIR}" && pwd -P)
          ABSREM=$(echo "${ABSDIR}" | ${SED} "s/${REPPAT}/${REMPAT}/")
          echo "cd ${REPOREMOTE} && make -e \${MAKEJ}" >>"${TESTSCRIPT}"
          echo "RESULT=\$?; if [ \"0\" != \"\${RESULT}\" ]; then exit \${RESULT}; fi" >>"${TESTSCRIPT}"
          if [ "${REPOREMOTE}" != "${ABSREM}" ]; then
            echo "cd ${ABSREM} && make -e \${MAKEJ}" >>"${TESTSCRIPT}"
            echo "RESULT=\$?; if [ \"0\" != \"\${RESULT}\" ]; then exit \${RESULT}; fi" >>"${TESTSCRIPT}"
          fi
          echo "echo \"--- RUN ${PARTITION}\"" >>"${TESTSCRIPT}"
          DIRSED=$(echo "${ABSREM}" | ${SED} "${DIRPAT}")
          ${SED} \
            -e "s/#\!..*/#\!\/bin\/bash\nset -eo pipefail\n${UMASK_CMD}/" -e "s/\(^\|[[:space:]]\)\(\.\|\.\.\)\//\1${DIRSED}\/\2\//" \
            -e "s/^[./]*\([[:print:]][[:print:]]*\/\)*slurm[[:space:]][[:space:]]*//" \
            -e "/^#SBATCH/d" -e "/^[[:space:]]*$/d" \
            -e "s/^srun[[:space:]]//" \
            "${SLURMFILE}" >"${SLURMFILE}.run" && chmod +rx "${SLURMFILE}.run"
          RUNFILE=$(readlink -f "${SLURMFILE}.run")
          RUNREM=$(echo "${RUNFILE}" | ${SED} "s/${REPPAT}/${REMPAT}/")
          CMDREM=$(echo "${TOOL_COMMAND}" | ${SED} "s/${REPPAT}/${REMPAT}/")
          if [ "${TOOL_COMMAND}" ]; then
            if [ "0" = "${TOOL_INJECT}" ] || [ ! "$(${SED} -n "/^taskset/p" "${RUNFILE}")" ]; then
              echo -n "${CMDREM} ${RUNREM} \$@ ${TOOL_COMMAND_POST}" >>"${TESTSCRIPT}"
            else # inject TOOL_COMMAND
              TOOL_COMMAND_SED1="$(echo "${CMDREM}" | ${SED} "${DIRPAT}") "
              if [ "${TOOL_COMMAND_POST}" ]; then
                TOOL_COMMAND_SED2=" $(echo "${TOOL_COMMAND_POST}" | ${SED} "${DIRPAT}")"
              fi
              ${SED} -i "s/\(^taskset[[:space:]]..*\)/${TOOL_COMMAND_SED1}\1${TOOL_COMMAND_SED2}/" "${RUNFILE}"
              echo -n "${RUNREM} \$@" >>"${TESTSCRIPT}"
            fi
          else
            echo -n "${RUNREM} \$@" >>"${TESTSCRIPT}"
          fi
          if [ "${LIMITLOG}" ] && [ "0" != "${LIMITLOG}" ] && \
             [ "$(command -v cat)" ] && [ "$(command -v tail)" ];
          then
            echo " | cat -s | tail -n ${LIMITLOG}" >>"${TESTSCRIPT}"
          elif [ "0" = "${LIMITLOG}" ]; then
            echo " >/dev/null" >>"${TESTSCRIPT}"
          else
            echo >>"${TESTSCRIPT}"
          fi
          echo "rm -f ${RUNREM}" >>"${TESTSCRIPT}"
        else
          echo "${TEST}" >>"${TESTSCRIPT}"
        fi
        # debug test environment
        if [ "${DEBUG_TEST}" ] && [ "0" != "${DEBUG_TEST}" ]; then
          echo "echo \"DEBUG: \$(hostname)\"" >>"${TESTSCRIPT}"
          echo "if [ -d \"${REPOREMOTE}/bin\" ]; then STAT=\$(stat -c %a \"${REPOREMOTE}/bin\"); echo \"  BIN: \${STAT}\"; fi" >>"${TESTSCRIPT}"
          echo "if [ -d \"${REPOREMOTE}/obj\" ]; then STAT=\$(stat -c %a \"${REPOREMOTE}/obj\"); echo \"  OBJ: \${STAT}\"; fi" >>"${TESTSCRIPT}"
          echo "if [ -d \"${REPOREMOTE}/lib\" ]; then STAT=\$(stat -c %a \"${REPOREMOTE}/lib\"); echo \"  LIB: \${STAT}\"; fi" >>"${TESTSCRIPT}"
        fi
        echo >>"${TESTSCRIPT}"
        if [ "${SYNC}" ]; then ${SYNC}; fi
      elif [ "${CONFIGFILE}" ]; then # setup environment on a per-test basis
        if [ -e "${ENVFILE}" ]; then
          eval "${REPOROOT}/scripts/tool_envrestore.sh" "${ENVFILE}"
        fi
        source "${CONFIGFILE}" ""
      fi

      COMMAND=$(eval echo "${ENVSTR} ${LAUNCH}")
      # run the prepared test case/script
      if [ "$(command -v tee)" ]; then
        if [ "${LOGFILE_INIT}" ]; then
          if [ "." = "$(dirname "${LOGFILE_INIT}")" ]; then
            LOGFILE=${PWD}/${LOGFILE_INIT}
          fi
        elif [ "${LABEL}" ]; then
          LOGFILE=${PWD}/.test-$(echo "${LABEL}" | ${TR} "[:upper:]" "[:lower:]").log
        else
          LOGFILE=${PWD}/.test.log
        fi
        LOGPATH=$(dirname "${LOGFILE}")
        LOGBASE=$(basename "${LOGFILE}" .log)
        if [ "1" != "${NPARTITIONS}" ]; then
          LOGBASE=${LOGBASE}-${PARTITION}
        fi
        if [ "1" != "${NCONFIGS}" ]; then
          LOGBASE=${LOGBASE}-${COUNT_CFG}
        fi
        if [ "1" != "${NENVS}" ]; then
          LOGBASE=${LOGBASE}-${COUNT_ENV}
        fi
        export LOGFILE=${LOGPATH}/${LOGBASE}.log
        if [ -t 0 ]; then
          eval "${COMMAND} 2>&1 | tee ${LOGFILE}"
        else
          eval "${COMMAND} 2>&1 | ${GREP} -v '^srun: error:' | tee ${LOGFILE}"
        fi
      else
        eval "${COMMAND}"
      fi
      # capture test status
      RESULT=$?

      # exit the loop in case of an error
      if [ "0" != "${RESULT}" ] && [ "1" != "${LIMITHARD}" ]; then
        if [ "${TOUCHFILE}" ]; then
          rm -f "${TOUCHFILE}"
          TOUCHFILE=""
        fi
        break 4
      fi
    COUNT_ENV=$((COUNT_ENV+1)); done # ENVS
    COUNT_CFG=$((COUNT_CFG+1)); done # CONFIGS
    COUNT_PRT=$((COUNT_PRT+1)); done # PARTITIONS
    if [ "${TOUCHFILE}" ]; then
      echo "${JOBID}" >"${TOUCHFILE}"
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

  # override result code (alternative outcome)
  if [ "${RESULTCODE}" ]; then
    RESULT=${RESULTCODE}
  fi

  exit "${RESULT}"
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
