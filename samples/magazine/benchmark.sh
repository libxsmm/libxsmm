#!/bin/sh
#############################################################################
# Copyright (c) 2018-2019, Intel Corporation                                #
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
ECHO=$(command -v echo)
CAT=$(command -v cat)
TR=$(command -v tr)

# setup thread affinity
export OMP_SCHEDULE=static OMP_PROC_BIND=TRUE

OUT_BLAZE=benchmark-blaze.txt
OUT_EIGEN=benchmark-eigen.txt
OUT_XSMM=benchmark-xsmm.txt
OUT_BLAS=benchmark-blas.txt

SCRT=${HERE}/../../scripts/libxsmm_utilities.py

# MNK: comma separated numbers are on its own others are combined into triplets
RUNS1=$(${SCRT} -1 $((128*128*128)) 11 \
  2, 3, 5, 10, 20, 30, \
  5 7 13, \
  23, 32 \
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

if [ "" != "$1" ]; then
  SIZE=$1
  shift
else
  SIZE=0
fi

if [ "" != "$1" ]; then
  RUNS=RUNS$1
  shift
else
  RUNS=RUNS1
fi

${CAT} /dev/null > ${OUT_BLAZE}
${CAT} /dev/null > ${OUT_EIGEN}
${CAT} /dev/null > ${OUT_XSMM}
${CAT} /dev/null > ${OUT_BLAS}

NRUN=1
NMAX=$(${ECHO} ${!RUNS} | wc -w | tr -d " ")
for RUN in ${!RUNS} ; do
  MVALUE=$(${ECHO} ${RUN} | cut --output-delimiter=' ' -d_ -f1)
  NVALUE=$(${ECHO} ${RUN} | cut --output-delimiter=' ' -d_ -f2)
  KVALUE=$(${ECHO} ${RUN} | cut --output-delimiter=' ' -d_ -f3)
  ${ECHO} "${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})... "
  ${ECHO} -n "${MVALUE} ${NVALUE} ${KVALUE} "                                   >> ${OUT_BLAZE}
  ${HERE}/magazine_blaze ${SIZE} ${MVALUE} ${NVALUE} ${KVALUE} | ${TR} "\n" " " >> ${OUT_BLAZE}
  ${ECHO}                                                                       >> ${OUT_BLAZE}
  ${ECHO} -n "${MVALUE} ${NVALUE} ${KVALUE} "                                   >> ${OUT_EIGEN}
  ${HERE}/magazine_eigen ${SIZE} ${MVALUE} ${NVALUE} ${KVALUE} | ${TR} "\n" " " >> ${OUT_EIGEN}
  ${ECHO}                                                                       >> ${OUT_EIGEN}
  ${ECHO} -n "${MVALUE} ${NVALUE} ${KVALUE} "                                   >> ${OUT_XSMM}
  ${HERE}/magazine_xsmm  ${SIZE} ${MVALUE} ${NVALUE} ${KVALUE} | ${TR} "\n" " " >> ${OUT_XSMM}
  ${ECHO}                                                                       >> ${OUT_XSMM}
  ${ECHO} -n "${MVALUE} ${NVALUE} ${KVALUE} "                                   >> ${OUT_BLAS}
  ${HERE}/magazine_blas  ${SIZE} ${MVALUE} ${NVALUE} ${KVALUE} | ${TR} "\n" " " >> ${OUT_BLAS}
  ${ECHO}                                                                       >> ${OUT_BLAS}
  NRUN=$((NRUN+1))
done

