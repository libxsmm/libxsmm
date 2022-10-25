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

SORT=$(command -v sort)
SED=$(command -v sed)
CUT=$(command -v cut)

VARIANT=Specialized

if [ "$1" ]; then
  VARIANT=$1
  shift
fi

HERE=$(cd "$(dirname "$0")" && pwd -P)
FILE=${HERE}/cp2k-perf.txt

GREP=$(command -v grep)
PERF=$(${GREP} -A1 -i "${VARIANT}" ${FILE} | \
  ${GREP} -e "performance" | \
  ${CUT} -d" " -f2 | \
  ${SORT} -n)

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

  if [ ! "$1" ]; then
    FILENAME=cp2k-$(echo ${VARIANT} | tr '[:upper:]' '[:lower:]').pdf
  else
    FILENAME=$1
    shift
  fi
  if [ ! "$1" ]; then
    MULTI=1
  else
    MULTI=$1
    shift
  fi
  ${GREP} -i -A2 \
    -e "^m=" -e "${VARIANT}" \
    ${FILE} | \
  ${SED} \
    -e "s/m=//" -e "s/n=//" -e "s/k=//" -e "s/ldc=[0-9][0-9]* //" -e "s/ (..*) / /" \
    -e "s/size=//" -e "s/batch=[0-9][0-9]* //" -e "s/memory=//" -e "s/ GB\/s//" \
    -e "/^..*\.\.\./Id" -e "/^$/d" -e "/--/d" | \
  ${SED} \
    -e "N;s/ MB\( (.P)\)*\n\tperformance://g" \
    -e "N;s/ GFLOPS\/s\n\tbandwidth://g" \
  > "${HERE}/cp2k-perf.dat"
  env \
    GDFONTPATH=/cygdrive/c/Windows/Fonts \
    FILENAME=${FILENAME} \
    MULTI=${MULTI} \
  "${WGNUPLOT}" cp2k-perf.plt
fi

