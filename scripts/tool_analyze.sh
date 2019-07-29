#!/bin/bash
#############################################################################
# Copyright (c) 2017-2019, Intel Corporation                                #
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

MAKE=$(command -v make)
GREP=$(command -v grep)
SORT=$(command -v sort)
CXX=$(command -v clang++)
CC=$(command -v clang)

if [ "" != "${MAKE}" ] && [ "" != "${CXX}" ] && [ "" != "${CC}" ] && \
   [ "" != "${GREP}" ] && [ "" != "${SORT}" ];
then
  HERE=$(cd $(dirname $0); pwd -P)
  cd ${HERE}/..
  ARG=$*
  if [ "" = "${ARG}" ]; then
    ARG=lib
  fi
  ${MAKE} CXX=${CXX} CC=${CC} FC= DBG=1 EFLAGS=--analyze ${ARG} 2> .analyze.log
  ISSUES=$(${GREP} -e "error:" -e "warning:" .analyze.log | ${GREP} -v "is never read" | ${SORT} -u)
  echo
  echo   "================================================================================"
  if [ "" = "${ISSUES}" ]; then
    echo "SUCCESS"
    echo "================================================================================"
  else
    echo "Errors (warnings)"
    echo "================================================================================"
    echo "${ISSUES}"
  fi
else
  echo "Error: missing prerequisites!"
  exit 1
fi

