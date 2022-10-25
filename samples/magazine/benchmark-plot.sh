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
FIND=$(command -v find)
SORT=$(command -v sort)
JOIN=$(command -v join)
CUT=$(command -v cut)
SED=$(command -v sed)
AWK=$(command -v awk)
RM=$(command -v rm)


if [ ! "$1" ]; then
  KIND=xsmm
else
  KIND=$1
  shift
fi
if [ ! "$1" ]; then
  FILEEXT=pdf
else
  FILEEXT=$1
  shift
fi
if [ ! "$1" ]; then
  MULTI=1
else
  MULTI=$1
  shift
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
  if [ -f "${HERE}/benchmark.set" ]; then
    # determine behavior of sort command
    export LC_ALL=C.UTF-8
    ${JOIN} --nocheck-order \
      <(${CUT} "${HERE}/benchmark.set" -d" " -f1-3 | ${SORT} -nk1) \
      <(${SORT} -nk1 benchmark-${KIND}.txt) \
    | ${AWK} \
      '{ if ($2==$4 && $3==$5) printf("%s %s %s %s %s\n", $1, $2, $3, $6, $8) }' \
    | ${SORT} \
      -b -n -k1 -k2 -k3 \
    > benchmark-${KIND}.join
  fi
  env GDFONTPATH=/cygdrive/c/Windows/Fonts \
    FILEEXT=${FILEEXT} KIND=${KIND} MULTI=${MULTI} \
    "${WGNUPLOT}" "${HERE}/benchmark.plt"
  if [ "1" != "${MULTI}" ] && [ "pdf" != "${FILEEXT}" ] && [ "$(command -v mogrify)" ]; then
    ${FIND} . -name "benchmark*.${FILEEXT}" -type f -exec mogrify -trim -transparent-color white {} \;
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

