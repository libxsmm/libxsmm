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
# shellcheck disable=SC2034,SC2129

HERE=$(cd "$(dirname "$0")" && pwd -P)
CAT=$(command -v cat)
TR=$(command -v tr)

# setup thread affinity
export OMP_SCHEDULE=static OMP_PROC_BIND=TRUE

OUT_BLAZE=benchmark-blaze.txt
OUT_EIGEN=benchmark-eigen.txt
OUT_XSMM=benchmark-xsmm.txt
OUT_XBAT=benchmark-xbat.txt
OUT_BLAS=benchmark-blas.txt

SCRT=${HERE}/../../scripts/libxsmm_utilities.py

# MNK: comma separated numbers are on its own others are combined into triplets
RUNS1=$(${SCRT} -1 $((128*128*128)) 21 \
  2, 3, 4, 5, 8, 10, 15, 16, 20, 23, 24, 25, 28, 30, 32, 35, 36, 40, \
  5 7 13, \
  0 0)
RUNS2=$(${SCRT} -1 $((128*128*128)) 46 \
  4 5 7 9 13 25 26 28 32 45, \
  13 14 25 26 32, \
  5 32 13 24 26, \
  14 16 29, \
  14 32 29, \
  16 29 55, \
  32 29 55, \
  9 32 22, \
  4 10 15, \
  6 7 8, \
  23, \
  64, \
  78, \
  12, \
  6, \
  0 0)

if [ "$1" ]; then
  SIZE=$1
  shift
else
  SIZE=0
fi

if [ "$1" ]; then
  RUNS=RUNS$1
  shift
else
  RUNS=RUNS1
fi

${CAT} /dev/null >"${OUT_BLAZE}"
${CAT} /dev/null >"${OUT_EIGEN}"
${CAT} /dev/null >"${OUT_XSMM}"
${CAT} /dev/null >"${OUT_XBAT}"
${CAT} /dev/null >"${OUT_BLAS}"

NRUN=1
NMAX=$(echo ${!RUNS} | wc -w | tr -d " ")
for RUN in ${!RUNS} ; do
  MVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f1)
  NVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f2)
  KVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f3)
  echo "${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})... "
  echo -n "${MVALUE} ${NVALUE} ${KVALUE} "                                                >>"${OUT_BLAZE}"
  "${HERE}/magazine_blaze" "${SIZE}" "${MVALUE}" "${NVALUE}" "${KVALUE}" | ${TR} "\n" " " >>"${OUT_BLAZE}"
  echo                                                                                    >>"${OUT_BLAZE}"
  echo -n "${MVALUE} ${NVALUE} ${KVALUE} "                                                >>"${OUT_EIGEN}"
  "${HERE}/magazine_eigen" "${SIZE}" "${MVALUE}" "${NVALUE}" "${KVALUE}" | ${TR} "\n" " " >>"${OUT_EIGEN}"
  echo                                                                                    >>"${OUT_EIGEN}"
  echo -n "${MVALUE} ${NVALUE} ${KVALUE} "                                                >>"${OUT_XSMM}"
  "${HERE}/magazine_xsmm"  "${SIZE}" "${MVALUE}" "${NVALUE}" "${KVALUE}" | ${TR} "\n" " " >>"${OUT_XSMM}"
  echo                                                                                    >>"${OUT_XSMM}"
  echo -n "${MVALUE} ${NVALUE} ${KVALUE} "                                                >>"${OUT_XBAT}"
  "${HERE}/magazine_batch" "${SIZE}" "${MVALUE}" "${NVALUE}" "${KVALUE}" | ${TR} "\n" " " >>"${OUT_XBAT}"
  echo                                                                                    >>"${OUT_XBAT}"
  echo -n "${MVALUE} ${NVALUE} ${KVALUE} "                                                >>"${OUT_BLAS}"
  "${HERE}/magazine_blas"  "${SIZE}" "${MVALUE}" "${NVALUE}" "${KVALUE}" | ${TR} "\n" " " >>"${OUT_BLAS}"
  echo                                                                                    >>"${OUT_BLAS}"
  NRUN=$((NRUN+1))
done

