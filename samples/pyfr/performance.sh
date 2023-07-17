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
HERE=$(cd "$(dirname "$0")" && pwd -P)
MATS=${HERE}/mats
#
# Build PyFR sample code with "make all".
# Consider fixing CPU clock frequency, and
# disabling all kinds of "turbo boost".
#
export OMP_PROC_BIND=${OMP_PROC_BIND:-TRUE}
export FSSPMDM_NBLOCK=${FSSPMDM_NBLOCK:-40}
export PERF_R=${PERF_R:-10000}
export PERF_N=${PERF_N:-40}
export LIBXSMM_VERBOSE=0

WAIT=12
if [[ ! -e "${HERE}/pyfr_driver_asp_reg" || ! -e "${HERE}/gimmik" || ("$(command -v ldd)" \
  && ("$(ldd "${HERE}/pyfr_driver_asp_reg" | sed -n '/omp/p')" || \
      "$(ldd "${HERE}/gimmik" | sed -n '/omp/p')")) ]];
then
  echo "Please build the PyFR sample code with \"make all\"!"
  if [ ! -e "${HERE}/pyfr_driver_asp_reg" ] || [ ! -e "${HERE}/gimmik" ]; then exit 1; fi
  if [ "0" != "$((0<WAIT))" ] && [ "$(command -v sleep)" ]; then
    echo
    echo "Benchmark will start in ${WAIT} seconds. Hit CTRL-C to abort."
    sleep ${WAIT}
  fi
fi

# ensure proper permissions
if [ "${UMASK}" ]; then
  UMASK_CMD="umask ${UMASK};"
  eval "${UMASK_CMD}"
fi

# optionally enable script debug
if [ "${PERFORMANCE_DEBUG}" ] && [ "0" != "${PERFORMANCE_DEBUG}" ]; then
  echo "*** DEBUG ***"
  PYTHON=$(command -v python3 || true)
  if [ ! "${PYTHON}" ]; then
    PYTHON=$(command -v python || true)
  fi
  if [ "${PYTHON}" ]; then
    ${PYTHON} -m site --user-site 2>&1 && echo
  fi
  env
  echo "*** DEBUG ***"
  if [[ ${PERFORMANCE_DEBUG} =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
    set -xv
  else
    set "${PERFORMANCE_DEBUG}"
  fi
fi

TMPF=$(mktemp)
trap 'rm -f ${TMPF}' EXIT

SEP=";"
POSTFX="-sp"
PERF_B=1
MATX=$(sed 's/\//\\\//g' <<<"${MATS}")
echo "------------------------------------------------------------------"
echo "LIBXSMM"
echo "------------------------------------------------------------------"
echo "MATRIX${SEP}N${SEP}NREP${SEP}BETA${SEP}SPARSE${SEP}DENSE${SEP}BLAS"
for MTX in "${MATS}"/p*/{pri,hex}/m{3,6}"${POSTFX}".mtx; do
  MAT=$(sed "s/^${MATX}\///" <<<"${MTX}" | sed 's/\(.*\)\..*/\1/' | sed "s/${POSTFX}$//")
  RESULT=$("${HERE}/pyfr_driver_asp_reg" "${MTX}" "${PERF_N}" "${PERF_R}" "${PERF_B}")
  SPARSE=$(sed -n "s/[[:space:]][[:space:]]*LIBXSMM GFLOPS : \(..*\) (sparse)/\1/p" <<<"${RESULT}")
  DENSE=$(sed -n "s/[[:space:]][[:space:]]*LIBXSMM GFLOPS : \(..*\) (dense)/\1/p" <<<"${RESULT}")
  BLAS=$(sed -n "s/[[:space:]][[:space:]]*BLAS GFLOPS    : \(..*\)/\1/p" <<<"${RESULT}")
  echo "${MAT}${SEP}${PERF_N}${SEP}${PERF_R}${SEP}${PERF_B}${SEP}${SPARSE}${SEP}${DENSE}${SEP}${BLAS}"
done | tee -a "${TMPF}"

PERF_B=0
export FSSPMDM_NTS=0
for MTX in "${MATS}"/p*/{pri,hex}/m{0,132,460}"${POSTFX}".mtx; do
  MAT=$(sed "s/^${MATX}\///" <<<"${MTX}" | sed 's/\(.*\)\..*/\1/' | sed "s/${POSTFX}$//")
  RESULT=$("${HERE}/pyfr_driver_asp_reg" "${MTX}" "${PERF_N}" "${PERF_R}" "${PERF_B}")
  SPARSE=$(sed -n "s/[[:space:]][[:space:]]*LIBXSMM GFLOPS : \(..*\) (sparse)/\1/p" <<<"${RESULT}")
  DENSE=$(sed -n "s/[[:space:]][[:space:]]*LIBXSMM GFLOPS : \(..*\) (dense)/\1/p" <<<"${RESULT}")
  BLAS=$(sed -n "s/[[:space:]][[:space:]]*BLAS GFLOPS    : \(..*\)/\1/p" <<<"${RESULT}")
  echo "${MAT}${SEP}${PERF_N}${SEP}${PERF_R}${SEP}${PERF_B}${SEP}${SPARSE}${SEP}${DENSE}${SEP}${BLAS}"
done | tee -a "${TMPF}"

echo "MATRIX${SEP}N${SEP}NREP${SEP}BETA${SEP}SPARSE${SEP}DENSE${SEP}BLAS" >"${HERE}/libxsmm.csv"
sort -t"${SEP}" -k1 "${TMPF}" >>"${HERE}/libxsmm.csv"

echo
echo "------------------------------------------------------------------"
echo "Gimmik"
echo "------------------------------------------------------------------"
echo "MATRIX${SEP}GFLOPS${SEP}MEMBW"
"${HERE}/gimmik" "${PERF_R}" | tee "${TMPF}"
echo "MATRIX${SEP}GFLOPS${SEP}MEMBW" >"${HERE}/gimmik.csv"
sort -t"${SEP}" -k1 "${TMPF}" >>"${HERE}/gimmik.csv"

cut -d"${SEP}" -f1,2 "${HERE}/gimmik.csv" | sed "1s/GFLOPS/GIMMIK/" \
| join -t"${SEP}" \
  "${HERE}/libxsmm.csv" \
  - \
>"${HERE}/performance.csv"

RESULT=$?
if [ "$(command -v datamash)" ]; then
  if ! datamash --headers -t"${SEP}" geomean 5-8 \
      <"${HERE}/performance.csv" >"${TMPF}" 2>/dev/null \
    || [ ! -s "${TMPF}" ];
  then
    datamash --headers -t"${SEP}" mean 5-8 \
      <"${HERE}/performance.csv" >"${TMPF}"
  fi
  if [ "-r" != "$1" ] && [ "--report" != "$1" ]; then
    echo
    echo "------------------------------------------------------------------"
    echo "Performance"
    echo "------------------------------------------------------------------"
    cat "${TMPF}"
    echo
  else
    read -r -d $'\04' HEADER VALUES <"${TMPF}" || true
    if [ "${HEADER}" ] && [ "${VALUES}" ]; then
      IFS="${SEP}"; N=0
      read -ra ENTRIES <<<"${VALUES}"
      COUNT=${#ENTRIES[@]}
      { echo
        echo "------------------------------------------------------------------"
        echo "Benchmark: PyFR"
        echo
        for LABEL in ${HEADER}; do
          if [ "0" != "$((COUNT<=N))" ]; then break; fi
          echo "${LABEL}: ${ENTRIES[N]} GFLOPS/s"
          N=$((N+1))
        done
        echo
      } >"${TMPF}"
      unset IFS COUNT N
      if [ ! "${LOGRPT_ECHO}" ]; then export LOGRPT_ECHO=1; fi
      # post-process result further (graphical report)
      eval "${HERE}/../../scripts/tool_logrept.sh ${TMPF}"
      RESULT=$?
    fi
  fi
fi

exit ${RESULT}
