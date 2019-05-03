#!/bin/bash
#############################################################################
# Copyright (c) 2015-2019, Intel Corporation                                #
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
DEPDIR=${HERE}/../../..

TMPF=$(${DEPDIR}/.mktmp.sh /tmp/.libxsmm_XXXXXX.out)
UNAME=$(command -v uname)
GREP=$(command -v grep)
SORT=$(command -v sort)
RM=$(command -v rm)
TR=$(command -v tr)

if [ "Darwin" != "$(${UNAME})" ]; then
  LIBEXT=so
else
  LIBEXT=dylib
fi
if [ "" != "$1" ]; then
  TEST=$1
  shift
else
  TEST=dgemm
fi

if [ -e ${HERE}/${TEST}-blas ]; then
  NAME=$(echo ${TEST} | ${TR} [:lower:] [:upper:])
  echo "============================="
  echo "Running ${NAME} (ORIGINAL BLAS)"
  echo "============================="
  { time ${HERE}/${TEST}-blas.sh "$@" 2>${TMPF}; } 2>&1 | ${GREP} real
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    echo -n "FAILED(${RESULT}) "; ${SORT} -u ${TMPF}
    ${RM} -f ${TMPF}
    exit ${RESULT}
  else
    echo -n "OK "; ${SORT} -u ${TMPF}
  fi
  echo

  if [ -e ${DEPDIR}/lib/libxsmmext.${LIBEXT} ]; then
    echo
    echo "============================="
    echo "Running ${NAME} (LD_PRELOAD)"
    echo "============================="
    { time \
      LD_LIBRARY_PATH=${DEPDIR}/lib:${LD_LIBRARY_PATH} LD_PRELOAD=${DEPDIR}/lib/libxsmmext.${LIBEXT} \
      DYLD_LIBRARY_PATH=${DEPDIR}/lib:${DYLD_LIBRARY_PATH} DYLD_INSERT_LIBRARIES=${DEPDIR}/lib/libxsmmext.${LIBEXT} \
      ${HERE}/${TEST}-blas.sh "$@" 2>${TMPF}; } 2>&1 | ${GREP} real
    RESULT=$?
    if [ 0 != ${RESULT} ]; then
      echo -n "FAILED(${RESULT}) "; ${SORT} -u ${TMPF}
      ${RM} -f ${TMPF}
      exit ${RESULT}
    else
      echo -n "OK "; ${SORT} -u ${TMPF}
    fi
    echo
  fi
fi

if [ -e ${HERE}/${TEST}-wrap ] && [ -e .state ] && \
   [ "" = "$(${GREP} 'BLAS=0' .state)" ];
then
  echo
  echo "============================="
  echo "Running ${NAME} (STATIC WRAP)"
  echo "============================="
  { time ${HERE}/${TEST}-wrap.sh "$@" 2>${TMPF}; } 2>&1 | ${GREP} real
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    echo -n "FAILED(${RESULT}) "; ${SORT} -u ${TMPF}
    ${RM} -f ${TMPF}
    exit ${RESULT}
  else
    echo -n "OK "; ${SORT} -u ${TMPF}
  fi
  echo
fi

${RM} -f ${TMPF}

