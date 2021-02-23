#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
SCRT=${HERE}/../../scripts/libxsmm_utilities.py
FILE=eigen_smm-cp2k.txt

RUNS0=$(${SCRT} -1 $((64*64*64-0))   19  23, 6, 14 16 29, 5 16 13 24 26, 9 16 22, 32, 64, 78, 16 29 55                                                         0 0)
RUNS1=$(${SCRT} -1 $((64*64*64-0))   19  23, 6, 14 16 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55                                                         0 0)
RUNS2=$(${SCRT} -1 $((64*64*64-0))   20  23, 6, 14 16 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 12                                                     0 0)
RUNS3=$(${SCRT} -1 $((64*64*64-0))   26  23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12                                 0 0)
RUNS4=$(${SCRT} -1 $((64*64*64-1))   31  23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12, 13 26 28 32 45                 0 0)
RUNS5=$(${SCRT} -1 $((64*64*64-0))   31  23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12, 13 26 28 32 45                 0 0)
RUNS6=$(${SCRT} -1 $((80*80*80-0))   35  23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12, 13 26 28 32 45, 7 13 25 32     0 0)
RUNS7=$(${SCRT} -1 $((80*80*80-0))   35  23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45          0 0)
RUNS8=$(${SCRT} -1 $((80*80*80-0))   37  23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45, 4 10    0 0)
RUNS9=$(${SCRT} -1 $((80*80*80-0))   38  23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45, 4 10 15 0 0)
RUNS10=$(${SCRT} -1 $((128*128*128)) 41  23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45, 4 10 15, 6 7 8 0 0)
RUNS11=$(${SCRT} -1 $((128*128*128)) 46  23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45, 4 10 15, 6 7 8, 13 14 25 26 32 0 0)

CASE=6
if [ "$1" ]; then
  CASE=$1
  shift
fi

case "$1" in
  "-"*) RUNS=RUNS${1:1}; shift
  ;;
esac

if [ "" = "${!RUNS}" ]; then
  RUNS=RUNS11
fi

if [ "$1" ]; then
  SIZE=$1
  shift
else
  SIZE=0
fi
# NREPEAT not defined if not given per CLI
if [ "$1" ]; then
  NREPEAT=$1
fi
cat /dev/null > ${FILE}

NRUN=1
NMAX=$(echo ${!RUNS} | wc -w | tr -d " ")
for RUN in ${!RUNS} ; do
  MVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f1)
  NVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f2)
  KVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f3)
  >&2 echo -n "${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})... "
  ERROR=$({ ${HERE}/eigen_smm ${CASE} ${MVALUE} ${NVALUE} ${KVALUE} ${SIZE} ${NREPEAT} >> ${FILE}; } 2>&1)
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    echo "FAILED(${RESULT}) ${ERROR}"
    exit 1
  else
    echo "OK ${ERROR}"
  fi
  echo >> ${FILE}
  NRUN=$((NRUN+1))
done

