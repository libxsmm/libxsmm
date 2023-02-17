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
# shellcheck disable=SC2207

DATAMASH=$(command -v datamash)
CAT=$(command -v cat)
SED=$(command -v sed)
BC=$(command -v bc)
SEP=";"

if [ ! "${CAT}" ] || [ ! "${SED}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

# ensure proper permissions
if [ "${UMASK}" ]; then
  UMASK_CMD="umask ${UMASK};"
  eval "${UMASK_CMD}"
fi

while test $# -gt 0; do
  case "$1" in
  -h|--help)
    HELP=1
    shift $#;;
  -o|--out)
    OFILE=$2
    shift 2;;
  -s|--sep)
    SEP=$2
    shift 2;;
  -e|--echo)
    if [[ $2 =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
      ECHO=$2; shift 2
    else
      ECHO=1; shift 1
    fi;;
  -j|--json)
    if [[ $2 =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
      JSON=$2; shift 2
    else
      JSON=1; shift 1
    fi;;
  *)
    if [ ! "${IFILE}" ]; then IFILE=$1; fi
    shift 1;;
  esac
done

# argument sanity checks
if [ "${IFILE}" ] && [ "${OFILE}" ] && \
   [ "$(cd "$(dirname "${IFILE}")" && pwd)/$(basename "${IFILE}")" = \
     "$(cd "$(dirname "${OFILE}")" && pwd)/$(basename "${OFILE}")" ];
then
  >&2 echo "ERROR: infile and outfile are equal!"
  exit 1
fi

# print usage information and exit
if [ "${HELP}" ] && [ "0" != "${HELP}" ]; then
  echo "Usage: $0 [options] IFILE"
  echo "       -o|--out file: filename of output"
  echo "       -s|--sep char: delimiter (CSV-file)"
  echo "       -e|--echo [0|1]: echo the input"
  echo "       -j|--json [0|1]: JSON-format"
  echo
  echo "Example: ./run_convs.sh 1 10 -1 f32 F 0 0 64 64 1 \\\\"
  echo "         | $0 -o performance.csv /dev/stdin"
  echo
  exit 0
fi

# handle errors more strictly
set -eo pipefail

if [ ! "${IFILE}" ]; then
  IFILE=/dev/stdin
elif [ ! -e "${IFILE}" ]; then
  >&2 echo "ERROR: logfile \"${IFILE}\" does not exist!"
  exit 1
fi

# automatically echoing input
if [ ! "${ECHO}" ]; then
  if [ "/dev/stdin" = "${IFILE}" ]; then
    ECHO=1
  else
    ECHO=0
  fi
fi

# Summarize input into telegram format or JSON-format
BEGIN=""
if [ "${JSON}" ] && [ "0" != "${JSON}" ]; then
  SPCPAT="[[:space:]][[:space:]]*"
  KEYPAT="[[:graph:]][[:graph:]]*"
  VALPAT="[[:print:]][[:print:]]*"
  while read -r LINE; do
    BENCHMARK=$(echo "${LINE}" | ${SED} -n "s/^Benchmark:${SPCPAT}\(${KEYPAT}\).*/\1/p")
    if [ "${BENCHMARK}" ]; then
      if [ "${BENCHMARK}" != "${BEGIN}" ]; then
        if [ "${BEGIN}" ]; then
          echo -e "\n  }," >>"${OFILE}"
        else
          if [ ! "${OFILE}" ]; then
            OFILE=$(mktemp)
            trap 'rm -f ${OFILE}' EXIT
          fi
          if [ "${SLURM_JOB_PARTITION}" ]; then
            echo "+++ REPORT ${SLURM_JOB_PARTITION}" >"${OFILE}"
          else
            echo "+++ REPORT ${HOSTNAME}" >"${OFILE}"
          fi
          echo "{" >>"${OFILE}"
        fi
        echo "  \"${BENCHMARK}\": {" >>"${OFILE}"
        BEGIN=${BENCHMARK}
      fi
      START=""
    elif [ "${BEGIN}" ]; then
      ENTRY=$(echo "${LINE}" \
        | ${SED} -n "s/^[[:space:]]*\(${KEYPAT}\)[[:space:]]*:${SPCPAT}\(${VALPAT}\).*/\"\1\": \"\2\"/p" \
        | ${SED} "s/${SPCPAT}/ /g")
      if [ "${ENTRY}" ]; then
        if [ "${BEGIN}" ] && [ "${START}" ]; then
          echo "," >>"${OFILE}"
        fi
        echo -n "    ${ENTRY}" >>"${OFILE}"
        START=${ENTRY}
      fi
    fi
    if [ "${ECHO}" ] && [ "0" != "${ECHO}" ]; then
      echo "${LINE}"
    fi
  done <"${IFILE}"
  if [ "${BEGIN}" ]; then
    echo -e "\n  }" >>"${OFILE}"
    echo "}" >>"${OFILE}"
    if [ "${OFILE}" ]; then
      echo # separating line
      ${CAT} "${OFILE}"
    fi
  fi
else # CSV-file (GNU Datamash)
  if [ ! "${DATAMASH}" ]; then
    >&2 echo "ERROR: missing GNU Datamash!"
    exit 1
  fi
  if [ ! "${BC}" ]; then
    >&2 echo "ERROR: missing Basic Calculator!"
    exit 1
  fi
  PATTERN="[[:space:]]*=[[:space:]]*\(..*\)/\2/p"
  NVALUES=0
  while read -r LINE; do
    ENTRY=$(echo "${LINE}" \
      | ${SED} -n "s/^\(GFLOP\)${PATTERN};s/^\(fp\|wu\|bp\) time${PATTERN}" 2>/dev/null \
      | ${SED} "s/\r//g")
    if [ "${ENTRY}" ]; then
      NVALUES=$((NVALUES+1))
      if [ "${VALUES}" ]; then
        VALUES=${VALUES}${SEP}${ENTRY}
      else
        VALUES=${ENTRY}
      fi
      if [ "0" = "$((NVALUES%2))" ]; then
        if [ ! "${OFILE}" ]; then
          OFILE=$(mktemp)
          trap 'rm -f ${OFILE}' EXIT
        fi
        # Write CSV-file header line
        if [ ! "${BEGIN}" ]; then
          echo "FLOPS${SEP}TIME" >"${OFILE}"
          BEGIN=${VALUES}
        fi
        # Write CSV-file record
        echo "${VALUES}" >>"${OFILE}"
        VALUES=""
      fi
    fi
    if [ "${ECHO}" ] && [ "0" != "${ECHO}" ]; then
      echo "${LINE}"
    fi
  done <"${IFILE}"
  if [ "${BEGIN}" ] && [ "${OFILE}" ]; then
    NUMPAT="s/\([+-]\{0,1\}[0-9]*\.\{0,1\}[0-9]\{1,\}\)[eE]+\{0,1\}\(-\{0,1\}\)\([0-9]\{1,\}\)/(\1*10^\2\3)/g"
    RESULT=($(${DATAMASH} 2>/dev/null <"${OFILE}" --header-in -t"${SEP}" --output-delimiter=" " sum 1 sum 2 \
      | ${SED} 2>/dev/null "${NUMPAT}"))
  fi
  # Print result in "telegram" format
  if [ "${RESULT[0]}" ] && [ "${RESULT[1]}" ]; then
    echo # separating line
    if [ "${SLURM_JOB_PARTITION}" ]; then
      echo "+++ PERFORMANCE ${SLURM_JOB_PARTITION}"
    else
      echo "+++ PERFORMANCE ${HOSTNAME}"
    fi
    # Print detailed results (per-layer)
    LAYER=0
    while read -r LINE; do
      if [ "0" != "${LAYER}" ]; then # skip header line
        DETAIL=($(echo "${LINE}" | ${SED} 2>/dev/null -e "${NUMPAT}" -e "s/${SEP}/ /"))
        if [ "${DETAIL[0]}" ] && [ "${DETAIL[1]}" ]; then
          printf "Layer %i: %f ms\n" "${LAYER}" "$(${BC} 2>/dev/null -l <<<"1000*${DETAIL[1]}")"
        fi
      fi
      LAYER=$((LAYER+1))
    done <"${OFILE}"
    # Print summary (over all layers)
    printf "%f ms\n" "$(${BC} 2>/dev/null -l <<<"1000*${RESULT[1]}")"
    printf "%.0f GFLOPS/s\n" "$(${BC} 2>/dev/null -l <<<"${RESULT[0]}/${RESULT[1]}")"
    printf "%.0f Hz (fps)\n" "$(${BC} 2>/dev/null -l <<<"1/${RESULT[1]}")"
  fi
fi
