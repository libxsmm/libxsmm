#!/bin/bash
#############################################################################
# Copyright (c) 2018, Intel Corporation                                     #
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

REPO=hfp
PROJECT=libxsmm
AHEADMIN=$1

if [ "" = "${AHEADMIN}" ] || [ "0" = "$((0<AHEADMIN))" ]; then
  AHEADMIN=1
fi

FORKS=$(wget -qO- https://api.github.com/repos/${REPO}/${PROJECT}/forks \
      | sed -n 's/ *\"login\": \"\(..*\)\".*/\1/p')
ABRANCH=$(wget -qO- https://api.github.com/repos/${REPO}/${PROJECT} \
      | sed -n 's/ *\"default_branch\": \"\(..*\)\".*/\1/p')
BBRANCH=${ABRANCH}

AHEAD=0
for FORK in ${FORKS}; do
  AHEADBY=$(wget -qO- https://api.github.com/repos/${REPO}/${PROJECT}/compare/${REPO}:${ABRANCH}...${FORK}:${BBRANCH} \
          | sed -n 's/ *\"ahead_by\": \"\(..*\)\".*/\1/p')
  if [ "0" != "$((AHEADMIN<=AHEADBY))" ]; then
    echo "Fork \"${FORK}\" is ahead by ${AHEADBY} fork(s)."
    AHEAD=$((AHEAD+1))
  fi
done

if [ "0" = "${AHEAD}" ]; then
  echo "None of the forks is ahead."
fi

