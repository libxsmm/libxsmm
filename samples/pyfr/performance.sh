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
# shellcheck disable=SC2143

HERE=$(cd "$(dirname "$0")" && pwd -P)
BASE=$(echo "$0" | sed 's/\(.[^/]*\/\)*//' | sed 's/\(.*\)\..*/\1/')
MATS=${HERE}/mats
#
# Build PyFR sample code with "make all".
# Consider fixing CPU clock frequency, and
# disabling all kinds of "turbo boost".
#
export OMP_PROC_BIND=${OMP_PROC_BIND:-TRUE}
export FSSPMDM_NBLOCK=${FSSPMDM_NBLOCK:-40}
export PERF_R=${PERF_R:-200000}
export PERF_N=${PERF_N:-40}

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

TMPF=$(mktemp)
trap 'rm -f ${TMPF}' EXIT

SEP=";"
POSTFX="-sp"
PERF_B=1
MATX=$(echo "${MATS}" | sed 's/\//\\\//g')
echo "------------------------------------------------------------------"
echo "LIBXSMM"
echo "------------------------------------------------------------------"
echo "MATRIX${SEP}N${SEP}NREP${SEP}BETA${SEP}SPARSE${SEP}DENSE${SEP}BLAS"
for MTX in "${MATS}"/p*/{pri,hex}/m{3,6}"${POSTFX}".mtx; do
  MAT=$(echo "${MTX}" | sed "s/^${MATX}\///" | sed 's/\(.*\)\..*/\1/' | sed "s/${POSTFX}$//")
  RESULT=$("${HERE}/pyfr_driver_asp_reg" "${MTX}" "${PERF_N}" "${PERF_R}" "${PERF_B}")
  SPARSE=$(echo "${RESULT}" | sed -n "s/[[:space:]][[:space:]]*LIBXSMM GFLOPS : \(..*\) (sparse)/\1/p")
  DENSE=$(echo "${RESULT}" | sed -n "s/[[:space:]][[:space:]]*LIBXSMM GFLOPS : \(..*\) (dense)/\1/p")
  BLAS=$(echo "${RESULT}" | sed -n "s/[[:space:]][[:space:]]*BLAS GFLOPS    : \(..*\)/\1/p")
  echo "${MAT}${SEP}${PERF_N}${SEP}${PERF_R}${SEP}${PERF_B}${SEP}${SPARSE}${SEP}${DENSE}${SEP}${BLAS}"
done | tee -a "${TMPF}.csv"

PERF_B=0
export FSSPMDM_NTS=0
for MTX in "${MATS}"/p*/{pri,hex}/m{0,132,460}"${POSTFX}".mtx; do
  MAT=$(echo "${MTX}" | sed "s/^${MATX}\///" | sed 's/\(.*\)\..*/\1/' | sed "s/${POSTFX}$//")
  RESULT=$("${HERE}/pyfr_driver_asp_reg" "${MTX}" "${PERF_N}" "${PERF_R}" "${PERF_B}")
  SPARSE=$(echo "${RESULT}" | sed -n "s/[[:space:]][[:space:]]*LIBXSMM GFLOPS : \(..*\) (sparse)/\1/p")
  DENSE=$(echo "${RESULT}" | sed -n "s/[[:space:]][[:space:]]*LIBXSMM GFLOPS : \(..*\) (dense)/\1/p")
  BLAS=$(echo "${RESULT}" | sed -n "s/[[:space:]][[:space:]]*BLAS GFLOPS    : \(..*\)/\1/p")
  echo "${MAT}${SEP}${PERF_N}${SEP}${PERF_R}${SEP}${PERF_B}${SEP}${SPARSE}${SEP}${DENSE}${SEP}${BLAS}"
done | tee -a "${TMPF}.csv"

echo "MATRIX${SEP}N${SEP}NREP${SEP}BETA${SEP}SPARSE${SEP}DENSE${SEP}BLAS" >libxsmm.csv
sort -t"${SEP}" -k1 "${TMPF}.csv" >>libxsmm.csv

echo
echo "------------------------------------------------------------------"
echo "Gimmik"
echo "------------------------------------------------------------------"
echo "MATRIX${SEP}GFLOPS${SEP}MEMBW"
"${HERE}/gimmik" "${PERF_R}" | tee "${TMPF}.csv"
echo "MATRIX${SEP}GFLOPS${SEP}MEMBW" >gimmik.csv
sort -t"${SEP}" -k1 "${TMPF}.csv" >>gimmik.csv

cut -d"${SEP}" -f1,2 gimmik.csv | sed "1s/GFLOPS/GIMMIK/" \
| join --header -t"${SEP}" \
  libxsmm.csv \
  - \
>"${BASE}.csv"

if [ "$(command -v datamash)" ]; then
  echo
  echo "------------------------------------------------------------------"
  echo "Performance"
  echo "------------------------------------------------------------------"
  if [ "$(datamash geomean 2>&1 | grep invalid)" ]; then
    datamash --headers -t"${SEP}"    mean 5-8 <performance.csv
  else
    datamash --headers -t"${SEP}" geomean 5-8 <performance.csv
  fi
fi
