#!/bin/sh
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

CASE=0
if [ "" != "$1" ]; then
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
NMAX=$(echo ${RUNS} | wc -w)
for RUN in ${RUNS} ; do
  MVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f1)
  NVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f2)
  KVALUE=$(echo ${RUN} | cut --output-delimiter=' ' -d_ -f3)

  >&2 echo "Test ${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})"

  env LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ${HERE}/blas.sh        ${CASE} ${MVALUE} ${NVALUE} ${KVALUE}     >> smm-blas.txt
  echo                                                                                                    >> smm-blas.txt

  env LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ${HERE}/specialized.sh ${CASE} ${MVALUE} ${NVALUE} ${KVALUE}     >> smm-specialized.txt
  echo                                                                                                    >> smm-specialized.txt

  env LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ${HERE}/dispatched.sh  $((CASE/2)) ${MVALUE} ${NVALUE} ${KVALUE} >> smm-dispatched.txt
  echo                                                                                                    >> smm-dispatched.txt

  env LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ${HERE}/inlined.sh     $((CASE/2)) ${MVALUE} ${NVALUE} ${KVALUE} >> smm-inlined.txt
  echo                                                                                                    >> smm-inlined.txt

  NRUN=$((NRUN + 1))
done

