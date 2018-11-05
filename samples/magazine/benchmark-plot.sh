#!/bin/bash
#############################################################################
# Copyright (c) 2015-2018, Intel Corporation                                #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions        #
# are met:                                                                  #
# 1. Redistributions of source code must retain the above copyright         #
#    notice, this list of conditions and the following disclaimer.          #
# 2. Redistributions in binary form must reproduce the above copyright      #
#    notice, this list of conditions and the following disclaimer in the    #
#    documentation and/or other materials provided with the distribution.   #
# 3. Neither the name of the copyright holder nor the names of its          #
#    contributors may be used to endorse or promote products derived        #
#    from this software without specific prior written permission.          #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#############################################################################
# Hans Pabst (Intel Corp.)
#############################################################################

HERE=$(cd $(dirname $0); pwd -P)
FIND=$(which find)
SORT=$(which sort)
JOIN=$(which join)
CUT=$(which cut)
SED=$(which sed)
AWK=$(which awk)
RM=$(which rm)


if [ "" = "$1" ]; then
  KIND=xsmm
else
  KIND=$1
  shift
fi
if [ "" = "$1" ]; then
  FILEEXT=pdf
else
  FILEEXT=$1
  shift
fi
if [ "" = "$1" ]; then
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
  GNUPLOT=$(which gnuplot 2>/dev/null)
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
  if [ -f ${HERE}/benchmark.set ]; then
    ${JOIN} \
      <(${CUT} ${HERE}/benchmark.set -d" " -f1-3 | ${SORT} -k1) \
      <(${SORT} -k1 benchmark-${KIND}.txt) \
    | ${AWK} \
      '{ if ($2==$4 && $3==$5) printf("%s %s %s %s %s\n", $1, $2, $3, $6, $8) }' \
    | ${SORT} \
      -b -n -k1 -k2 -k3 \
    > benchmark-${KIND}.join
  fi
  env GDFONTPATH=/cygdrive/c/Windows/Fonts \
    FILEEXT=${FILEEXT} KIND=${KIND} MULTI=${MULTI} \
    "${WGNUPLOT}" ${HERE}/benchmark.plt
  if [ "1" != "${MULTI}" ] && [ "pdf" != "${FILEEXT}" ] && [ "" != "$(which mogrify 2>/dev/null)" ]; then
    ${FIND} . -name "benchmark*.${FILEEXT}" -type f -exec mogrify -trim -transparent-color white {} \;
  fi
fi

