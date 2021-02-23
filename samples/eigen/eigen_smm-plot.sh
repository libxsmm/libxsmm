#!/usr/bin/env bash
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

SORT=$(command -v sort)
JOIN=$(command -v join)
GREP=$(command -v grep)
CAT=$(command -v cat)
CUT=$(command -v cut)
SED=$(command -v sed)
AWK=$(command -v awk)
RM=$(command -v rm)

VARIANT="LIBXSMM streamed (A,B)"

HERE=$(cd "$(dirname "$0")" && pwd -P)
FILE=${HERE}/eigen_smm-cp2k.txt

PERF=$(${GREP} -A2 "${VARIANT}" ${FILE} \
   | ${GREP} -e "performance" \
   | ${CUT} -d" " -f2 \
   | ${SORT} -n)

NUM=$(echo "${PERF}" | wc -l | tr -d " ")
MIN=$(echo ${PERF} | ${CUT} -d" " -f1)
MAX=$(echo ${PERF} | ${CUT} -d" " -f${NUM})

echo "num=${NUM}"
echo "min=${MIN}"
echo "max=${MAX}"

BC=$(command -v bc)
if [ "${BC}" ]; then
  AVG=$(echo "$(echo -n "scale=3;(${PERF})/${NUM}" | tr "\n" "+")" | ${BC})
  NUM2=$((NUM / 2))

  if [ "0" = "$((NUM % 2))" ]; then
    A=$(echo ${PERF} | ${CUT} -d" " -f${NUM2})
    B=$(echo ${PERF} | ${CUT} -d" " -f$((NUM2 + 1)))
    MED=$(echo "$(echo -n "scale=3;(${A} + ${B})/2")" | ${BC})
  else
    MED=$(echo ${PERF} | ${CUT} -d" " -f$((NUM2 + 1)))
  fi

  echo "avg=${AVG}"
  echo "med=${MED}"
fi

if [ -f /cygdrive/c/Program\ Files/gnuplot/bin/wgnuplot ]; then
  WGNUPLOT=/cygdrive/c/Program\ Files/gnuplot/bin/wgnuplot
  GNUPLOT=/cygdrive/c/Program\ Files/gnuplot/bin/gnuplot
elif [ -f /cygdrive/c/Program\ Files\ \(x86\)/gnuplot/bin/wgnuplot ]; then
  WGNUPLOT=/cygdrive/c/Program\ Files\ \(x86\)/gnuplot/bin/wgnuplot
  GNUPLOT=/cygdrive/c/Program\ Files\ \(x86\)/gnuplot/bin/gnuplot
else
  GNUPLOT=$(command -v gnuplot)
  WGNUPLOT=${GNUPLOT}
fi

GNUPLOT_MAJOR=0
GNUPLOT_MINOR=0
if [ -f "${GNUPLOT}" ]; then
  GNUPLOT_MAJOR=$("${GNUPLOT}" --version | ${SED} "s/.\+ \([0-9]\).\([0-9]\) .*/\1/")
  GNUPLOT_MINOR=$("${GNUPLOT}" --version | ${SED} "s/.\+ \([0-9]\).\([0-9]\) .*/\2/")
fi
GNUPLOT_VERSION=$((GNUPLOT_MAJOR * 10000 + GNUPLOT_MINOR * 100))

if [ "40600" -le "${GNUPLOT_VERSION}" ]; then
  # determine behavior of sort command
  export LC_ALL=C.UTF-8

  if [ "" = "$1" ]; then
    FILENAME=eigen_smm-cp2k.pdf
  else
    FILENAME=$1
    shift
  fi
  if [ "" = "$1" ]; then
    MULTI=1
  else
    MULTI=$1
    shift
  fi
  ${SED} \
    -e "/^m=/,/${VARIANT}/{//!d}" \
    -e "/${VARIANT}/d" \
    -e "/\.\.\./,/Finished/{//!d}" \
    -e "/Finished/d" \
    -e "/\.\.\./d" \
    -e "/^$/d" \
    ${FILE} \
  | ${SED} \
    -e "s/m=//" -e "s/n=//" -e "s/k=//" -e "s/ (..*) / /" \
    -e "s/size=//" \
    -e "/duration:/d" \
  | ${SED} \
    -e "N;s/ memory=..*\n..*//" \
    -e "N;s/\n\tperformance:\(..*\) GFLOPS\/s/\1/" \
    -e "N;s/\n\tbandwidth:\(..*\) GB\/s/\1/" \
  > "${HERE}/eigen_smm-cp2k.dat"

  if [ -f "${HERE}/eigen_smm-cp2k.set" ]; then
    ${JOIN} --nocheck-order \
      <(${CUT} "${HERE}/eigen_smm-cp2k.set" -d" " -f1-3 | ${SORT} -nk1) \
      <(${SORT} -nk1 "${HERE}/eigen_smm-cp2k.dat") \
    | ${AWK} \
      '{ if ($2==$4 && $3==$5) printf("%s %s %s %s %s %s\n", $1, $2, $3, $6, $7, $8) }' \
    | ${SORT} \
      -b -n -k1 -k2 -k3 \
    > "${HERE}/eigen_smm-plot-join.dat"
  else
    ${RM} "${HERE}/eigen_smm-plot-join.dat"
  fi

  env \
    GDFONTPATH=/cygdrive/c/Windows/Fonts \
    FILENAME=${FILENAME} \
    MULTI=${MULTI} \
  "${WGNUPLOT}" eigen_smm-cp2k.plt
fi

