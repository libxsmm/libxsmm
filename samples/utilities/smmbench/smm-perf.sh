#!/usr/bin/env sh
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

CASE=0
if [ "$1" ]; then
  CASE=$1
  shift
fi

RUNS="2_2_2 4_4_4 4_6_9 5_5_5 5_5_13 5_13_5 5_13_13 6_6_6 8_8_8 10_10_10 12_12_12 13_5_5 13_5_7 13_5_13 13_13_5 13_13_13 13_13_26 \
  13_26_13 13_26_26 14_14_14 16_16_16 18_18_18 20_20_20 23_23_23 24_3_36 24_24_24 26_13_13 26_13_26 26_26_13 26_26_26 32_32_32 \
  40_40_40 48_48_48 56_56_56 64_64_64 72_72_72 80_80_80 88_88_88 96_96_96 104_104_104 112_112_112 120_120_120 128_128_128"

cat /dev/null > smm-blas.txt
cat /dev/null > smm-dispatched.txt
cat /dev/null > smm-inlined.txt
cat /dev/null > smm-specialized.txt

NRUN=1
NMAX=$(echo "${RUNS}" | wc -w | tr -d " ")
for RUN in ${RUNS} ; do
  MVALUE=$(echo "${RUN}" | cut --output-delimiter=' ' -d_ -f1)
  NVALUE=$(echo "${RUN}" | cut --output-delimiter=' ' -d_ -f2)
  KVALUE=$(echo "${RUN}" | cut --output-delimiter=' ' -d_ -f3)

  >&2 echo "Test ${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})"

  env LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH}" "${HERE}/blas"        "${CASE}" "${MVALUE}" "${NVALUE}" "${KVALUE}"     >>smm-blas.txt
  echo                                                                                                             >>smm-blas.txt

  env LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH}" "${HERE}/specialized" "${CASE}" "${MVALUE}" "${NVALUE}" "${KVALUE}"     >>smm-specialized.txt
  echo                                                                                                             >>smm-specialized.txt

  env LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH}" "${HERE}/dispatched"  "$((CASE/2))" "${MVALUE}" "${NVALUE}" "${KVALUE}" >>smm-dispatched.txt
  echo                                                                                                             >>smm-dispatched.txt

  env LD_LIBRARY_PATH=".:${LD_LIBRARY_PATH}" "${HERE}/inlined"     "$((CASE/2))" "${MVALUE}" "${NVALUE}" "${KVALUE}" >>smm-inlined.txt
  echo                                                                                                             >>smm-inlined.txt

  NRUN=$((NRUN + 1))
done
