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
# shellcheck disable=SC1090,SC1091,SC2028,SC2129,SC2153,SC2207
set -o pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
ROOTENV=${HERE}/../.env
ROOT=${HERE}/..

# TODO: map to CI-provider (abstract environment)
source "${ROOTENV}/buildkite.env" ""

CI_AGENT=$(command -v buildkite)
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

# optionally enable script debug
if [ "${DEBUG_TEST}" ] && [ "0" != "${DEBUG_TEST}" ]; then
  echo "*** DEBUG ***"
  env
  echo "*** DEBUG ***"
  if [[ ${DEBUG_TEST} =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
    set -xv
  else
    set "${DEBUG_TEST}"
  fi
fi

if [ "${MKTEMP}" ] && [ "${MKDIR}" ] && [ "${DIFF}" ] && [ "${GREP}" ] && [ "${SED}" ] && [ "${TR}" ]; then
  DIRPAT="s/\//\\\\\//g"
  REMPAT=$(${SED} "${DIRPAT}" <<<"${REPOREMOTE}")
  REPPAT=$(${SED} "${DIRPAT}" <<<"${REPOROOT}")
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

  # attempt to determine SLURMSCRIPT
  if [ "$1" ] && [ ! "${SLURMSCRIPT}" ] && \
    [[ ("$1" != "$(basename "$1" .sh)") || ("$1" != "$(basename "$1" .slurm)") ]];
  then
    SLURMSCRIPT=1
  fi

  # set the case number or (Slurm-)script (may not exist yet)
  if [ "$1" ] && [[ (-e "$1") || (("${SLURMSCRIPT}") && ("0" != "${SLURMSCRIPT}")) ]]; then
    export TESTSETFILE=$1
    TESTID=$(basename "${TESTSETFILE%.*}")
    export TESTSET=${TESTID}
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
    HOSTNAME=$(cut -d. -f1 2>/dev/null <<<"${HOSTNAME}")
  fi
  if [ ! "${HOSTNAME}" ]; then
    HOSTNAME=$(hostname -s 2>/dev/null)
  fi
  HOSTDELIMCHAR="-"
  HOSTPREFIX=$(cut -d${HOSTDELIMCHAR} -f1 2>/dev/null <<<"${HOSTNAME}")
  if [ "${HOSTPREFIX}" ]; then
    HOSTPREFIX="${HOSTPREFIX}${HOSTDELIMCHAR}"
  fi

  if [ "${SRUN}" ] && [ "0" != "${SLURM}" ]; then
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
  else
    PARTITIONS=none
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
    if [ "${ENVI}" ]; then
      ENVS=${ENVI}
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
  if [ ! "${TESTSETFILE}" ] || [[ (! -e "${TESTSETFILE}") && \
     ((! "${SLURMSCRIPT}") || ("0" = "${SLURMSCRIPT}")) ]];
  then
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
    if [ "${STEPNAME}" ]; then
      LABEL=$(${TR} -s "[:punct:][:space:]" "-" <<<"${STEPNAME}" \
        | ${SED} "s/^-//;s/-$//" \
        | ${SED} "s/[^A-Za-z0-9._-]//g")
    fi
    SRUN_LABEL=""
    if [ "${PIPELINE}" ] && [ "${JOBID}" ]; then
      SRUN_LABEL="${PIPELINE}/${JOBID}"
    elif [ "${LABEL}" ]; then
      SRUN_LABEL="${LABEL}"
    fi
    if [ "${SRUN_LABEL}" ]; then
      if [ "${BUILD_USER}" ]; then
        SRUN_LABEL="${SRUN_LABEL}/${BUILD_USER}"
      fi
      SRUN_FLAGS="${SRUN_FLAGS} -J ${SRUN_LABEL}"
    elif [ "${BUILD_USER}" ]; then
      SRUN_FLAGS="${SRUN_FLAGS} -J ${BUILD_USER}"
    fi
    if [ "${LIMITRUN}" ] && [ "0" != "${LIMITRUN}" ]; then
      # convert: seconds -> minutes
      SRUN_FLAGS="${SRUN_FLAGS} --time=$((LIMITRUN/60))"
    fi
    if [ "${SRUN_NODE}" ]; then  # request specific node
      SRUN_FLAGS="${SRUN_FLAGS} -w ${SRUN_NODE}"
    fi
    #SRUN_FLAGS="${SRUN_FLAGS} --preserve-env"
    TESTSCRIPT=$(${MKTEMP} "${REPOROOT}/.tool_XXXXXX.sh")
    chmod +rx "${TESTSCRIPT}"
    LAUNCH="${SRUN} --ntasks=1 --partition=\${PARTITION} ${SRUN_FLAGS} \
                    --unbuffered ${TESTSCRIPT} ${*:2}"
  elif [[ ("${LAUNCH_CMD}") || (-d "$1") || \
         (("${SLURMSCRIPT}") && ("0" != "${SLURMSCRIPT}")) ]];
  then
    TESTSCRIPT=$(${MKTEMP} "${REPOROOT}/.tool_XXXXXX.sh")
    REMSCRIPT=$(${SED} "s/${REPPAT}/${REMPAT}/" <<<"${TESTSCRIPT}")
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

  if [[ ("${UMASK}") && ((! "${TESTSCRIPT}") || (! -e "${TESTSCRIPT}")) ]]; then
    # TODO: derive permissions from UMASK
    trap 'rm ${TESTSCRIPT} ${ENVFILE} && (chmod -Rf g+u,o=u-w ${REPOROOT} || true)' EXIT
  else
    trap 'rm ${TESTSCRIPT} ${ENVFILE}' EXIT
  fi

  # artifact download (ARTIFACT_UPLOAD_DB=1)
  if [ "${CI_AGENT}" ] && [ "${ARTIFACT_ROOT}" ] && [ -d "${ARTIFACT_ROOT}" ] && [ "${PIPELINE}" ] && \
     [ "${ARTIFACT_UPLOAD_DB}" ] && [ "0" != "${ARTIFACT_UPLOAD_DB}" ];
  then
  ( # subshell
    cd "${ARTIFACT_ROOT}" || exit 1
    artifact_download "${PIPELINE}" "json" 1
  )
  fi

  RESULT=0
  LOGFILE_INIT=${LOGFILE}
  while [ "${TEST}" ] || TEST=$(eval " \
    ${SED} -n '/^ *script: *$/,\$p' ${REPOROOT}/${TESTSETFILE} | ${SED} '/^ *script: *$/d' | \
    ${SED} -n -E \"/^ *- */H;//,/^ *$/G;s/\n(\n[^\n]*){\${TESTID}}$//p\" 2>/dev/null | \
    ${SED} 's/^ *- *//;s/^  *//' | ${TR} '\n' ' ' | \
    ${SED} 's/  *$//'") && [ "${TEST}" ];
  do
    if [ -d "${TEST}" ]; then
      SLURMDIR=${TEST}
    else # dummy
      SLURMDIR=$0
    fi
    for SLURMFILE in "${SLURMDIR}"/*; do
    if [[ (-d ${SLURMDIR}) && ((! "${SLURMSCRIPT}") || ("0" = "${SLURMSCRIPT}")) ]]; then
      SLURMFILE=${SLURMDIR}/${SLURMFILE}
      TESTID=$(basename "${SLURMFILE%.*}")
    elif [[ (-e "${TEST}") || (("${SLURMSCRIPT}") && ("0" != "${SLURMSCRIPT}")) ]]; then
      SLURMFILE=${TEST}
    fi
    if [ "none" = "${PARTITIONS}" ] && [ "$0" != "${SLURMFILE}" ] && \
      [[ (-e "${SLURMFILE}") || (("${SLURMSCRIPT}") && ("0" != "${SLURMSCRIPT}")) ]];
    then
      PARTITION=$(${SED} -n "s/^#SBATCH[[:space:]][[:space:]]*\(--partition=\|-p\)\(..*\)/\2/p" "${SLURMFILE}")
      if [ "${PARTITION}" ]; then PARTITIONS=${PARTITION}; fi
    fi
    if [ "${LIMIT}" ] && [ "0" != "${LIMIT}" ] && \
       [ "$(command -v stat)" ] && \
       [ "$(command -v date)" ];
    then
      NOW=$(date +%s)
      LIMITFILE=$(${TR} "[:upper:]" "[:lower:]" <<<"${LABEL}")
      if [ ! "${LIMITFILE}" ]; then
        LIMITFILE=$(${SED} "s/[^A-Za-z0-9._-]//g" <<<"${TESTID}")
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
    # determine configuration files once according to pattern
    if [[ ("none" != "${CONFIG}") && (("${HOSTNAME}") || ("${HOSTPREFIX}")) ]]; then
      CONFIGFILES=($(ls -1 "${ROOTENV}/${HOSTNAME}"/${CONFIG}.env 2>/dev/null))
      if [[ ! "${CONFIGFILES[*]}" ]]; then
        CONFIGFILES=($(ls -1 "${ROOTENV}/${HOSTPREFIX}"*/${CONFIG}.env 2>/dev/null))
      fi
      if [[ "${CONFIGFILES[*]}" ]]; then
        CONFIGPAT=$(${SED} "s/[[:space:]][[:space:]]*/\\\|/g" <<<"${CONFIGEX}" | ${SED} "s/\\\|$//")
        if [ "${CONFIGPAT}" ]; then
          CONFIGFILES=($(printf "%s\n" "${CONFIGFILES[@]}" | ${SED} "/\(${CONFIGPAT}\)/d"))
        fi
        CONFIGCOUNT=${#CONFIGFILES[@]}
      fi
    fi
    # determine actual configuration for every test/iteration
    if [ "${CONFIGCOUNT}" ] && [ "0" != "${CONFIGCOUNT}" ]; then
      CONFIGFILE=${CONFIGFILES[RANDOM%CONFIGCOUNT]}
      CONFIG=$(basename "${CONFIGFILE}" .env)
      # setup Python environment if LAUNCH_USER cannot access orig. user's site-directory
      if [ "${LAUNCH_USER}" ] && [ "0" != "${SLURM}" ]; then
        PYTHONSITE=$(su "${LAUNCH_USER}" ${RUN_CMD} "python3 -m site --user-site 2>/dev/null")
        if [ ! "${PYTHONSITE}" ]; then
          PYTHONSITE=$(su "${LAUNCH_USER}" ${RUN_CMD} "python -m site --user-site 2>/dev/null")
        fi
        if [ "${PYTHONSITE}" ]; then
          export PYTHONPATH=${PYTHONSITE}:${PYTHONPATH}
        fi
      fi
    elif [ "none" != "${CONFIG}" ]; then
      echo "WARNING: configuration \"${CONFIG}\" not found!"
      CONFIGFILE=""
      CONFIG="none"
    fi
    # iterate over all given environments
    COUNT_ENV=0; for ENVI in ${ENVS}; do
      if [ "none" != "${ENVI}" ]; then
        ENVVAL=$(cut -d= -f2 <<<"${ENVI}")
        ENVSTR=${ENVI}
      fi
      # print some header if all tests are selected or in case of multi-tests
      HEADER=""
      if [ "none" != "${PARTITION}" ] && [ "0" != "${SHOW_PARTITION}" ]; then HEADER="${PARTITION}"; fi
      if [ "none" != "${CONFIG}" ]; then HEADER="${HEADER} ${CONFIG}"; fi
      if [ "${ENVVAL}" ]; then HEADER="${HEADER} ${ENVI}"; fi
      HEADER=$(${SED} "s/^[[:space:]][[:space:]]*//;s/[[:space:]][[:space:]]*$//" <<<"${HEADER}" \
        | ${TR} "[:lower:]" "[:upper:]" | ${TR} -s " " "/")
      if [ "${TESTID}" ] && [ "test" != "$(${TR} "[:upper:]" "[:lower:]" <<<"${TESTID}")" ]; then
        if [ "${HEADER}" ]; then
          CAPTION="${TESTID} (${HEADER})"
        else
          CAPTION="${TESTID}"
        fi
      else
        CAPTION="${HEADER}"
      fi
      if [ "${CAPTION}" ]; then
        echo "--- TEST ${CAPTION}"
      fi
      # prepare temporary script for remote environment/execution
      if [ "${TESTSCRIPT}" ] && [ -e "${TESTSCRIPT}" ]; then
        echo "#!/usr/bin/env bash" >"${TESTSCRIPT}"
        echo "SED=\$(command -v gsed); SED=\${SED:-\$(command -v sed)}" >>"${TESTSCRIPT}"
        echo "set -eo pipefail" >>"${TESTSCRIPT}"
        if [ "$0" != "${SLURMFILE}" ] && \
          [[ (-e "${SLURMFILE}") || (("${SLURMSCRIPT}") && ("0" != "${SLURMSCRIPT}")) ]];
        then
          RUNFILE=$(touch "${SLURMFILE}.run" && chmod +rx "${SLURMFILE}.run" && readlink -f "${SLURMFILE}.run")
          ABSDIR=$(dirname "${SLURMFILE}")
        else
          RUNFILE=$(touch "${TESTSCRIPT}.run" && chmod +rx "${TESTSCRIPT}.run" && readlink -f "${TESTSCRIPT}.run")
        fi
        RUNREM=$(${SED} "s/${REPPAT}/${REMPAT}/" <<<"${RUNFILE}")
        # exact/real name of run-file is not known yet
        EXIT_TRAP="rm -f ${RUNREM}"
        if [ "${UMASK}" ]; then # TODO: derive permissions from UMASK
          EXIT_TRAP="(${EXIT_TRAP}); (chmod -Rf g+u,o=u-w ${REPOREMOTE} || true)"
        fi
        echo "trap '${EXIT_TRAP}' EXIT" >>"${TESTSCRIPT}"
        echo "${UMASK_CMD}" >>"${TESTSCRIPT}"
        echo "cd ${REPOREMOTE}" >>"${TESTSCRIPT}"
        echo "if [ \"\$(command -v sync)\" ]; then sync; fi" >>"${TESTSCRIPT}"
        if [ "0" != "${SHOW_PARTITION}" ]; then echo "echo \"-> \${USER}@\${HOSTNAME} (\${PWD})\"" >>"${TESTSCRIPT}"; fi
        REMINFO=$(${SED} "s/${REPPAT}/${REMPAT}/" <<<"${CPUINFO}")
        echo "if [ ! \"\${MAKEJ}\" ]; then MAKEJ=\"-j \$(eval ${REMINFO} -nc)\"; fi" >>"${TESTSCRIPT}"
        # make execution environment available
        if [ ! "${INTEL_LICENSE_FILE}" ]; then
          LICSDIR=$(command -v icc | ${SED} "s/\(\/.*intel\)\/.*$/\1/")
          ${MKDIR} -p "${REPOROOT}/licenses"
          cp -u "${HOME}"/intel/licenses/* "${REPOROOT}/licenses" 2>/dev/null
          cp -u "${LICSDIR}"/licenses/* "${REPOROOT}/licenses" 2>/dev/null
          cp -u /opt/intel/licenses/* "${REPOROOT}/licenses" 2>/dev/null
          echo "export INTEL_LICENSE_FILE=${REPOREMOTE}/licenses" >>"${TESTSCRIPT}"
        fi
        # apply/restore environment on a per-test basis
        if [ ! "${ENV_APPLY}" ] || [ "0" != "${ENV_APPLY}" ]; then
          if [ "${HOME_REMOTE}" != "${HOME}" ]; then
            ENVRST_FLAGS="-s"
          fi
          ENVREM=$(${SED} "s/${REPPAT}/${REMPAT}/" <<<"${ENVFILE}")
          ENVRST=$(${SED} "s/${REPPAT}/${REMPAT}/" <<<"${HERE}/tool_envrestore.sh")
          echo "if [ -e \"${ENVREM}\" ]; then" >>"${TESTSCRIPT}"
          echo "  source ${ENVRST} ${ENVRST_FLAGS} ${ENVREM} || true" >>"${TESTSCRIPT}"
          echo "fi" >>"${TESTSCRIPT}"
        fi
        if [ "${CONFIGFILE}" ]; then
          echo "source \"$(${SED} "s/${REPPAT}/${REMPAT}/" <<<"${CONFIGFILE}")\" \"\"" >>"${TESTSCRIPT}"
        fi
        # install requested Python packages
        if ! [[ ${ENV_PYTHON} =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
          echo "if [ \"\${PYTHON}\" ]; then" >>"${TESTSCRIPT}"
          echo "  eval \"\${PYTHON} -m pip install --upgrade --user \${ENV_PYTHON} >/dev/null\"" >>"${TESTSCRIPT}"
          echo "fi" >>"${TESTSCRIPT}"
        fi
        # record the current test case
        if [ "${ABSDIR}" ]; then
          if [ ! -e "${ABSDIR}/Makefile" ] && [ -d "${ABSDIR}" ] && [ -e "${ABSDIR}/../Makefile" ]; then
            ABSDIR=${ABSDIR}/..
          fi
          ABSDIR=$(cd "${ABSDIR}" && pwd -P)
          ABSREM=$(${SED} "s/${REPPAT}/${REMPAT}/" <<<"${ABSDIR}")
          echo "cd ${REPOREMOTE} && make -e \${MAKEJ} ${MAKETGT}" >>"${TESTSCRIPT}"
          echo "RESULT=\$?; if [ \"0\" != \"\${RESULT}\" ]; then exit \${RESULT}; fi" >>"${TESTSCRIPT}"
          if [ "${REPOREMOTE}" != "${ABSREM}" ]; then
            echo "cd ${ABSREM} && make -e \${MAKEJ} ${MAKETGT}" >>"${TESTSCRIPT}"
            echo "RESULT=\$?; if [ \"0\" != \"\${RESULT}\" ]; then exit \${RESULT}; fi" >>"${TESTSCRIPT}"
          fi
          if [ "none" != "${PARTITION}" ]; then
            echo "echo \"--- RUN ${PARTITION}\"" >>"${TESTSCRIPT}"
          else # suspicious
            echo "echo -n \"--- \"" >>"${TESTSCRIPT}"
          fi
          SLURMREM=$(readlink -f "${SLURMFILE}" | ${SED} "s/${REPPAT}/${REMPAT}/")
          DIRSED=$(${SED} "${DIRPAT}" <<<"${ABSREM}")
          echo "\${SED} \
            -e \"s/#\!..*/#\!\/bin\/bash\nset -eo pipefail\n${UMASK_CMD}/\" \
            -e \"s/\(^\|[[:space:]]\)\(\.\|\.\.\)\//\1${DIRSED}\/\2\//\" \
            -e \"s/^[./]*\([[:print:]][[:print:]]*\/\)*slurm[[:space:]][[:space:]]*//\" \
            -e \"/^#SBATCH/d\" -e \"/#[[:space:]]*shellcheck/d\" -e \"/^[[:space:]]*$/d\" \
            -e \"s/^srun[[:space:]]//\" \
            \"${SLURMREM}\" >>\"${RUNREM}\"" >>"${TESTSCRIPT}"
          if [ "${TOOL_COMMAND}" ]; then # inject TOOL_COMMAND
            CMDREM=$(${SED} "s/${REPPAT}/${REMPAT}/" <<<"${TOOL_COMMAND}")
            echo -n "${CMDREM} ${RUNREM} \$@ ${TOOL_COMMAND_POST}" >>"${TESTSCRIPT}"
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
        else
          echo "${TEST}" >>"${TESTSCRIPT}"
        fi
        echo >>"${TESTSCRIPT}"
        if [ "${SYNC}" ]; then ${SYNC}; fi
      elif [ "${CONFIGFILE}" ]; then # setup environment on a per-test basis
        if [ -e "${ENVFILE}" ]; then
          eval "${REPOROOT}/scripts/tool_envrestore.sh" "${ENVFILE}"
        fi
        source "${CONFIGFILE}" ""
        # install requested Python packages
        if ! [[ ${ENV_PYTHON} =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]] && [ "${PYTHON}" ]; then
          eval "${PYTHON} -m pip install --upgrade --user ${ENV_PYTHON} >/dev/null"
        fi
      fi

      COMMAND=$(eval echo "${ENVSTR} ${LAUNCH}")
      # run the prepared test case/script
      if [ "$(command -v tee)" ]; then
        if [ "${LOGFILE_INIT}" ]; then
          if [ "." = "$(dirname "${LOGFILE_INIT}")" ]; then
            LOGFILE=${PWD}/${LOGFILE_INIT}
          fi
        elif [ "${LABEL}" ]; then
          LOGFILE=${PWD}/.test-$(${TR} "[:upper:]" "[:lower:]" <<<"${LABEL}").log
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

  # artifact upload
  if [ "${CI_AGENT}" ] && [ "${ARTIFACT_ROOT}" ] && [ -d "${ARTIFACT_ROOT}" ] && [ "${PIPELINE}" ]; then
    # upload regular artifacts
    if [ "${JOBID}" ] && [ -d "${ARTIFACT_ROOT}/${PIPELINE}/${JOBID}" ] && \
       [ ! -e "${ARTIFACT_ROOT}/${PIPELINE}/${JOBID}/.uploaded" ] && \
       [ "$(ls -1 "${ARTIFACT_ROOT}/${PIPELINE}/${JOBID}")" ];
    then
    ( # subshell
      cd "${ARTIFACT_ROOT}/${PIPELINE}/${JOBID}" || exit 1
      ${CI_AGENT} artifact upload "*"
      touch ./.uploaded
    )
    fi
    # upload database
    if [ "${ARTIFACT_UPLOAD_DB}" ] && [ "0" != "${ARTIFACT_UPLOAD_DB}" ] && \
       [ -e "${ARTIFACT_ROOT}/${PIPELINE}.json" ];
    then
    ( # subshell
      cd "${ARTIFACT_ROOT}" || exit 1
      ${CI_AGENT} artifact upload "${PIPELINE}.json;${PIPELINE}.weights.json"
    )
    fi
  fi

  # return captured status
  exit "${RESULT}"
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
